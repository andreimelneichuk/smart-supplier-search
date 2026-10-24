"""Объединённое LLM-обогащение каталога (enrich_catalog.py). Клиент LLM подменён, пауз нет."""
import json
import sys
import types

import pytest
from openai import APIError, RateLimitError

import db
import enrich_catalog as en


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "e.db")
    monkeypatch.setattr(en.time, "sleep", lambda s: None)
    c = db.connect()
    en.init_db(c)
    yield c
    c.close()


def add_supplier(con, sid, name="Завод", sub="Сыр", desc="Сыры и масло", city="Тула"):
    con.execute("INSERT INTO suppliers (id, name, url, subcategory, description, city, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)", (sid, name, f"https://x/producers/{sid}/s{sid}", sub, desc, city))
    con.commit()


def batch_rows(con):
    return con.execute("SELECT id, name, city, subcategory, description FROM suppliers ORDER BY id").fetchall()


class Raw:
    def __init__(self, content, headers=None):
        self._content = content
        self.headers = headers or {}

    def parse(self):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=self._content))])


class FakeClient:
    """Имитирует client.chat.completions.with_raw_response.create; очередь ответов или исключений."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        completions = types.SimpleNamespace(with_raw_response=types.SimpleNamespace(create=self.create))
        self.chat = types.SimpleNamespace(completions=completions)

    def create(self, model, messages, temperature, max_tokens):
        self.prompts.append(messages[0]["content"])
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def test_load_env_and_get_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        en.get_client()
    monkeypatch.setenv("OPENAI_API_KEY", "your_key")
    with pytest.raises(RuntimeError):
        en.get_client()
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_x")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    assert en.get_client() is not None


def test_init_db_creates_tables_idempotently(con):
    en.init_db(con)  # второй вызов не падает
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"enriched_catalog", "supplier_extra"} <= names


def test_clean_json_str_extracts_array():
    assert en.clean_json_str('ответ: [{"id": 1}] спасибо') == '[{"id": 1}]'
    assert en.clean_json_str("  без скобок  ") == "без скобок"


def test_build_search_keywords_lowercases_and_joins():
    kw = en.build_search_keywords({"primary_products": ["Сыр"], "specific_varieties": ["Моцарелла"],
                                   "processing": ["Копченый"]})
    assert kw == "сыр моцарелла копченый"
    assert en.build_search_keywords({}) == ""


def test_call_llm_returns_content_on_success():
    client = FakeClient([Raw("[]")])
    assert en.call_llm_with_retry(client, "p", "m") == "[]"


def test_call_llm_pauses_when_tokens_low(monkeypatch):
    waits = []
    monkeypatch.setattr(en.time, "sleep", lambda s: waits.append(s))
    client = FakeClient([Raw("[]", headers={"x-ratelimit-remaining-tokens": "100",
                                              "x-ratelimit-reset-tokens": "3.5s"})])
    assert en.call_llm_with_retry(client, "p", "m") == "[]"
    assert waits == [pytest.approx(5.0)]


def _rate_limit_error(msg):
    import httpx
    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(429, request=req)
    return RateLimitError(msg, response=resp, body=None)


def test_call_llm_waits_on_rate_limit_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(en.time, "sleep", lambda s: waits.append(s))
    client = FakeClient([_rate_limit_error("try again in 2.0s"), Raw("[1]")])
    assert en.call_llm_with_retry(client, "p", "m") == "[1]"
    assert waits == [pytest.approx(6.0)]


def test_call_llm_api_and_generic_errors_retry(monkeypatch):
    waits = []
    monkeypatch.setattr(en.time, "sleep", lambda s: waits.append(s))
    import httpx
    req = httpx.Request("POST", "https://x")
    client = FakeClient([APIError("сбой", request=req, body=None), RuntimeError("упало"), Raw("ok")])
    assert en.call_llm_with_retry(client, "p", "m") == "ok"
    assert waits == [6, 5]


def test_call_llm_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(en.time, "sleep", lambda s: None)
    client = FakeClient([RuntimeError("x")] * 2)
    assert en.call_llm_with_retry(client, "p", "m", max_retries=2) is None


def test_process_batch_saves_entities_and_extra(con):
    add_supplier(con, 1)
    add_supplier(con, 2, name="Тара")
    answer = json.dumps([
        {"id": 1, "is_food": True, "primary_products": ["сыр"], "specific_varieties": ["моцарелла"],
         "processing": ["копченый"], "dish_ingredients": [], "min_order": "от 100 кг",
         "price_range": "по запросу", "certificates": "ГОСТ", "delivery_terms": "ТК",
         "region_scope": "Вся Россия"},
        {"id": 2, "is_food": False},
    ])
    client = FakeClient([Raw(answer)])
    saved = en.process_batch(client, con, batch_rows(con), "m")
    assert saved == 2
    row = con.execute("SELECT is_food, primary_products, search_keywords FROM enriched_catalog WHERE supplier_id=1").fetchone()
    assert row[0] == 1
    assert json.loads(row[1]) == ["сыр"]
    assert row[2] == "сыр моцарелла копченый"
    assert con.execute("SELECT is_food FROM enriched_catalog WHERE supplier_id=2").fetchone()[0] == 0
    extra = con.execute("SELECT min_order, certificates FROM supplier_extra WHERE supplier_id=1").fetchone()
    assert tuple(extra) == ("от 100 кг", "ГОСТ")


def test_process_batch_fills_defaults_when_fields_missing(con):
    add_supplier(con, 1)
    client = FakeClient([Raw(json.dumps([{"id": 1}]))])
    en.process_batch(client, con, batch_rows(con), "m")
    extra = con.execute("SELECT min_order, delivery_terms FROM supplier_extra WHERE supplier_id=1").fetchone()
    assert extra[0] == "По запросу (от 1 коробки / партии)"
    assert extra[1] == "ТК по РФ, самовывоз"


def test_process_batch_skips_company_missing_from_answer(con):
    add_supplier(con, 1)
    client = FakeClient([Raw(json.dumps([{"id": 99}]))])
    assert en.process_batch(client, con, batch_rows(con), "m") == 0


def test_process_batch_empty_llm_answer(con):
    add_supplier(con, 1)
    client = FakeClient([Raw("")])
    assert en.process_batch(client, con, batch_rows(con), "m") == 0


def test_process_batch_bad_json(con, capsys):
    add_supplier(con, 1)
    client = FakeClient([Raw("[{не json")])
    assert en.process_batch(client, con, batch_rows(con), "m") == 0
    assert "Ошибка парсинга" in capsys.readouterr().out


def test_process_batch_prompt_contains_company_fields(con):
    add_supplier(con, 5, name="Ромашка", sub=None, city=None, desc="описание")
    client = FakeClient([Raw("[]")])
    en.process_batch(client, con, batch_rows(con), "m")
    assert "ID: 5 | Компания: Ромашка | Город: РФ | Подкатегория: —" in client.prompts[0]


def test_main_nothing_to_do(con, monkeypatch, capsys):
    monkeypatch.setattr(en, "get_client", lambda: FakeClient([]))
    monkeypatch.setattr(en.db, "connect", lambda: con)
    en.main()
    assert "уже полностью обогащены" in capsys.readouterr().out


def test_main_processes_pending_companies(con, monkeypatch, capsys):
    add_supplier(con, 1)
    add_supplier(con, 2)
    monkeypatch.setattr(en, "BATCH_SIZE", 1)
    monkeypatch.setattr(en.db, "connect", lambda: con)
    client = FakeClient([Raw(json.dumps([{"id": 1}])), Raw(json.dumps([{"id": 2}]))])
    monkeypatch.setattr(en, "get_client", lambda: client)
    en.main()
    out = capsys.readouterr().out
    assert "ОБОГАЩЕНИЕ ЗАВЕРШЕНО" in out
    assert con.execute("SELECT COUNT(*) FROM enriched_catalog").fetchone()[0] == 2


def test_module_load_env_reads_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ENRICH_TEST_K='v'\n", encoding="utf-8")
    monkeypatch.setattr(en, "HERE", tmp_path)
    monkeypatch.delenv("ENRICH_TEST_K", raising=False)
    en.load_env()
    import os
    assert os.environ["ENRICH_TEST_K"] == "v"
    monkeypatch.delenv("ENRICH_TEST_K", raising=False)

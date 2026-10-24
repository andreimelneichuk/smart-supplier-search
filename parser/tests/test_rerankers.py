"""LLM-реранкер (llm_rerank.py) и локальный cross-encoder (cross_encoder.py). Модели и сеть подменены."""
import sys
import types

import numpy as np
import pytest

import cross_encoder
import llm_rerank


def doc(name, sub="Колбасы", desc="описание"):
    return {"name": name, "subcategory": sub, "description": desc}


# --- llm_rerank ---

@pytest.fixture
def llm_cache(tmp_path, monkeypatch):
    path = tmp_path / "llm.db"
    monkeypatch.setattr(llm_rerank, "_cache", lambda p=None: _open(path))
    return path


def _open(path):
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS g (k TEXT PRIMARY KEY, grade INTEGER)")
    return con


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# c\nLLM_TEST_KEY_A='x'\nBAD\n", encoding="utf-8")
    monkeypatch.setattr(llm_rerank, "HERE", tmp_path)
    monkeypatch.delenv("LLM_TEST_KEY_A", raising=False)
    llm_rerank._load_env()
    import os
    assert os.environ["LLM_TEST_KEY_A"] == "x"
    monkeypatch.delenv("LLM_TEST_KEY_A", raising=False)


def test_line_and_prompt_contain_fields():
    line = llm_rerank._line(0, doc("Завод", sub="Сыр", desc="сыры\nмясо"))
    assert "товар: Сыр" in line and "компания: Завод" in line and "\n" not in line
    assert "—" in llm_rerank._line(1, {"name": "X"})
    p = llm_rerank._prompt("сыр", [doc("Завод")])
    assert "Запрос: «сыр»" in p and "1. товар" in p


@pytest.mark.parametrize("text, n, expected", [
    ("[3, 2, 0]", 3, [3, 2, 0]),
    ("ответ: [1,2]", 2, [1, 2]),
    ("[3, 2]", 3, None),        # длина не совпала
    ("[5]", 1, None),           # вне шкалы 0..3
    ("[1.5]", 1, None),         # не целое
    ("без массива", 1, None),
    ("[x]", 1, None),           # не парсится
])
def test_parse_grades(text, n, expected):
    assert llm_rerank._parse(text, n) == expected


def test_key_depends_on_query_and_card():
    a = llm_rerank._key("сыр", doc("A"))
    assert a == llm_rerank._key("сыр", doc("A"))
    assert a != llm_rerank._key("масло", doc("A"))


def test_call_without_key_raises(monkeypatch):
    monkeypatch.setattr(llm_rerank, "_load_env", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm_rerank._call("p")
    monkeypatch.setenv("OPENAI_API_KEY", "your_key_here")
    with pytest.raises(RuntimeError):
        llm_rerank._call("p")


def test_call_uses_openai_client(monkeypatch):
    monkeypatch.setattr(llm_rerank, "_load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_test")
    seen = {}

    class Msg:
        content = "[3]"

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    class Completions:
        @staticmethod
        def create(model, temperature, messages):
            seen["model"] = model
            return Resp()

    class OpenAI:
        def __init__(self, api_key, base_url):
            seen["key"] = api_key
            self.chat = types.SimpleNamespace(completions=Completions())

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    assert llm_rerank._call("p") == "[3]"
    assert seen["key"] == "gsk_test"


def test_grade_calls_llm_and_caches(llm_cache, monkeypatch):
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return "[3, 1]"

    monkeypatch.setattr(llm_rerank, "_call", fake_call)
    docs = [doc("A"), doc("B")]
    assert llm_rerank.grade("сыр", docs) == [3, 1]
    assert llm_rerank.grade("сыр", docs) == [3, 1]
    assert len(calls) == 1  # второй раз — из кэша


def test_grade_invalid_answer_gives_none(llm_cache, monkeypatch):
    monkeypatch.setattr(llm_rerank, "_call", lambda p: "мусор")
    assert llm_rerank.grade("сыр", [doc("A")]) == [None]


def test_grade_empty_list(llm_cache, monkeypatch):
    monkeypatch.setattr(llm_rerank, "_call", lambda p: pytest.fail("вызова не должно быть"))
    assert llm_rerank.grade("сыр", []) == []


def test_grade_batches_large_lists(llm_cache, monkeypatch):
    monkeypatch.setattr(llm_rerank, "BATCH", 2)
    sizes = []

    def fake_call(prompt):
        n = prompt.count("товар:")
        sizes.append(n)
        return "[" + ", ".join(["2"] * n) + "]"

    monkeypatch.setattr(llm_rerank, "_call", fake_call)
    docs = [doc(f"Ф{i}") for i in range(5)]
    assert llm_rerank.grade("сыр", docs) == [2] * 5
    assert sizes == [2, 2, 1]


def test_rerank_blends_top_n_and_keeps_rest(llm_cache, monkeypatch):
    monkeypatch.setattr(llm_rerank, "_call", lambda p: "[3, 0]")
    docs = [doc("A"), doc("B"), doc("C")]
    base = [0.2, 0.9, 0.1]
    out = llm_rerank.rerank("сыр", docs, base, top_n=2, weight=0.5)
    # топ-2 по базе: B (0.9) и A (0.2), LLM получает их в этом порядке: B -> 3, A -> 0; C остаётся как есть
    assert out[2] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.5 * 0.9 + 0.5 * 3 / 3)
    assert out[0] == pytest.approx(0.5 * 0.2 + 0.5 * 0 / 3)


def test_rerank_keeps_base_when_llm_fails(llm_cache, monkeypatch):
    monkeypatch.setattr(llm_rerank, "_call", lambda p: "мусор")
    base = [0.4, 0.6]
    assert llm_rerank.rerank("сыр", [doc("A"), doc("B")], base) == base


# --- cross_encoder ---

class FakeTextEncoder:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def rerank(self, query, docs):
        self.calls.append(list(docs))
        return [float(len(d)) for d in docs]


@pytest.fixture
def ce(monkeypatch):
    monkeypatch.setattr(cross_encoder, "_model", None)
    monkeypatch.setattr(cross_encoder, "_failed_at", None)
    monkeypatch.setattr(cross_encoder, "_score_cache", {})
    return cross_encoder


def _install_fastembed(monkeypatch, cls):
    pkg = types.ModuleType("fastembed")
    rr = types.ModuleType("fastembed.rerank")
    ce_mod = types.ModuleType("fastembed.rerank.cross_encoder")
    ce_mod.TextCrossEncoder = cls
    monkeypatch.setitem(sys.modules, "fastembed", pkg)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", rr)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", ce_mod)


def test_get_reranker_loads_once(ce, monkeypatch):
    _install_fastembed(monkeypatch, FakeTextEncoder)
    first = ce.get_reranker()
    assert ce.get_reranker() is first
    assert first.name == ce.MODEL_NAME


def test_get_reranker_failure_then_retry_window(ce, monkeypatch):
    class Broken:
        def __init__(self, name):
            raise OSError("нет модели")

    _install_fastembed(monkeypatch, Broken)
    with pytest.raises(OSError):
        ce.get_reranker()
    # в течение RETRY_AFTER повторной загрузки нет
    with pytest.raises(RuntimeError):
        ce.get_reranker()
    # после окна — пробуем снова
    monkeypatch.setattr(ce, "_failed_at", ce.time.time() - ce.RETRY_AFTER - 1)
    _install_fastembed(monkeypatch, FakeTextEncoder)
    assert ce.get_reranker().name == ce.MODEL_NAME


def test_format_doc_for_rerank_roles():
    text = ce_format({"subcategory": "Сыр", "name": "Завод", "description": "a\nb"})
    assert text == "Товар: Сыр. Компания: Завод. Описание: a b"
    assert "Товар: —" in ce_format({})


def ce_format(d):
    return cross_encoder.format_doc_for_rerank(d)


def test_format_doc_truncates_description():
    text = ce_format({"description": "я" * 1000})
    assert len(text) < 400


def test_sigmoid_range_and_clamp():
    assert cross_encoder.sigmoid(0) == pytest.approx(0.5)
    assert cross_encoder.sigmoid(1000) == pytest.approx(cross_encoder.sigmoid(20))
    assert 0 < cross_encoder.sigmoid(-1000) < 1e-6


def test_rerank_candidates_empty(ce):
    assert cross_encoder.rerank_candidates("q", [], []) == []


def test_rerank_candidates_sorts_and_caches(ce, monkeypatch):
    model = FakeTextEncoder("m")
    monkeypatch.setattr(ce, "get_reranker", lambda: model)
    docs = [doc("Короткая"), doc("Очень длинное название компании")]
    out = ce.rerank_candidates("сыр", docs, [0, 1])
    assert out[0][0] == 1  # длиннее — выше по нашей заглушке
    again = ce.rerank_candidates("сыр", docs, [0, 1])
    assert again == out
    assert len(model.calls) == 1  # второй раз ничего не считаем


def test_rerank_candidates_clears_cache_when_full(ce, monkeypatch):
    model = FakeTextEncoder("m")
    monkeypatch.setattr(ce, "get_reranker", lambda: model)
    monkeypatch.setattr(ce, "_CACHE_MAX", 1)
    ce._score_cache.update({("x", str(i)): 0.0 for i in range(3)})
    ce.rerank_candidates("сыр", [doc("A")], [0])
    assert len(ce._score_cache) == 1


def test_rerank_scores_blends_top_n(ce, monkeypatch):
    model = FakeTextEncoder("m")
    monkeypatch.setattr(ce, "get_reranker", lambda: model)
    docs = [doc("A"), doc("Очень длинная компания"), doc("B")]
    base = np.array([0.1, 0.5, 0.9])
    out = ce.rerank_scores("сыр", docs, base, top_n=2, weight=0.5)
    # топ-2 по базе — индексы 2 и 1; индекс 0 вне топ-2 и не меняется
    assert out[0] == pytest.approx(0.1)
    assert out[2] != pytest.approx(0.9)
    assert out[1] != pytest.approx(0.5)


def test_rerank_scores_empty_and_zero_top(ce):
    assert cross_encoder.rerank_scores("q", [], np.zeros(0)).shape == (0,)
    base = np.array([0.3, 0.4])
    out = cross_encoder.rerank_scores("q", [doc("A"), doc("B")], base, top_n=0)
    assert np.allclose(out, base)

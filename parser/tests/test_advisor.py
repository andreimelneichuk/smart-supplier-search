"""LLM-советник (advisor.py): проверка ответов по карточке, паспорт, сравнение. LLM подменён — сеть не трогаем."""
import json
import sqlite3

import pytest
from openai import OpenAIError

import advisor
import db
import live


# --- подменённый клиент OpenAI ---

class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResp:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, answers):
        self.answers = answers  # список: строка-ответ или исключение
        self.models = []

    def create(self, model, messages, **kw):
        self.models.append(model)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return FakeResp(a)


class FakeClient:
    def __init__(self, answers):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(answers)


def llm_by_prompt(rules):
    """call_llm, который отвечает по подстроке промпта: rules = [(подстрока, ответ), ...]."""
    def call(client, prompt, max_tokens=800):
        for needle, answer in rules:
            if needle in prompt:
                return answer(prompt) if callable(answer) else answer
        return ""
    return call


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "adv.db")
    c = db.connect()
    yield c
    c.close()


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(advisor, "get_client", lambda: None)
    monkeypatch.setattr(advisor, "fetch_full_page_text", lambda url: "")


def sup(id, name, sub="Колбасы", desc="Колбасы и сосиски", city="Казань", **kw):
    base = {"id": id, "name": name, "subcategory": sub, "description": desc, "city": city, "url": "",
            "categories": [], "phones": [], "emails": [], "products_count": 10, "slug": f"s{id}",
            "primary_products": [], "distance_km": None, "score": 0.5, "contacts_at": None}
    base.update(kw)
    return base


# --- вспомогательные функции ---

def test_load_env_reads_dotenv_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# комментарий\nOPENAI_MODEL_TEST_X='abc'\n\nNOVALUE\n", encoding="utf-8")
    monkeypatch.setattr(advisor, "HERE", tmp_path)
    monkeypatch.setattr(advisor, "HERE", tmp_path)
    monkeypatch.delenv("OPENAI_MODEL_TEST_X", raising=False)
    advisor.load_env()
    import os
    assert os.environ["OPENAI_MODEL_TEST_X"] == "abc"
    os.environ["OPENAI_MODEL_TEST_X"] = "keep"
    advisor.load_env()
    assert os.environ["OPENAI_MODEL_TEST_X"] == "keep"
    monkeypatch.delenv("OPENAI_MODEL_TEST_X", raising=False)


def test_get_client_without_key_is_none(monkeypatch):
    monkeypatch.setattr(advisor, "load_env", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert advisor.get_client() is None
    monkeypatch.setenv("OPENAI_API_KEY", "your_key_here")
    assert advisor.get_client() is None


def test_get_client_with_key_builds_client(monkeypatch):
    monkeypatch.setattr(advisor, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    c = advisor.get_client()
    assert c is not None
    assert str(c.base_url).startswith("https://example.test/v1")


def test_get_client_without_base_url(monkeypatch):
    monkeypatch.setattr(advisor, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert advisor.get_client() is not None


def test_call_llm_falls_back_on_error_and_empty():
    client = FakeClient([OpenAIError("лимит"), "", "ответ модели"])
    assert advisor.call_llm(client, "промпт") == "ответ модели"
    assert len(client.chat.completions.models) == 3


def test_call_llm_all_models_fail_returns_empty():
    client = FakeClient([OpenAIError("a")] * len(advisor.FALLBACK_MODELS))
    assert advisor.call_llm(client, "промпт") == ""


def test_call_llm_gpt_oss_gets_bigger_budget(monkeypatch):
    seen = {}

    class C:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens):
                    seen[model] = max_tokens
                    return FakeResp("ok")

    monkeypatch.setattr(advisor, "FALLBACK_MODELS", ["openai/gpt-oss-20b"])
    advisor.call_llm(C(), "p", max_tokens=100)
    assert seen["openai/gpt-oss-20b"] == 1600


@pytest.mark.parametrize("raw, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 2}\n```', {"a": 2}),
    ('Вот ответ: {"a": 3} спасибо', {"a": 3}),
    ('{"a": [1, 2,], }', {"a": [1, 2]}),
    ("совсем не json", None),
    ("", None),
])
def test_parse_json_from_llm(raw, expected):
    assert advisor.parse_json_from_llm(raw) == expected


def test_is_known_filters_placeholders():
    assert advisor.is_known("от 100 кг")
    assert not advisor.is_known("не указано")
    assert not advisor.is_known("—")
    assert not advisor.is_known("")
    assert not advisor.is_known(None)


def test_norm_yo_and_thousands_separator():
    assert advisor._norm("Ёлка 1 500 шт") == "елка 1500 шт"


def test_unsupported_numbers_and_documents():
    source = "Колбасы варёные. Минимальный заказ 100 кг. Имеется ГОСТ 32094."
    assert advisor.unsupported("от 100 кг", source) is False
    assert advisor.unsupported("от 200 кг", source) is True  # числа нет в карточке
    assert advisor.unsupported("сертификат ХАССП", source) is True  # документа нет
    assert advisor.unsupported("ГОСТ 32094", source) is False


def test_ground_string_and_list():
    source = "Минимальный заказ 100 кг"
    assert advisor.ground("от 100 кг", source) == "от 100 кг"
    assert advisor.ground("от 999 кг", source) == advisor.NOT_SPECIFIED
    assert advisor.ground("не указано", source) == "не указано"
    assert advisor.ground(["100 кг", "999 кг", 5], source) == ["100 кг"]
    assert advisor.ground(12, source) == 12


def test_ground_sentences_drops_unsupported_sentences():
    source = "Работаем с 2010 года. Минимальный заказ 100 кг."
    text = "Компания опытная. Минимальный заказ 100 кг. Есть ХАССП и ГОСТ 9999."
    out = advisor.ground_sentences(text, source)
    assert "ХАССП" not in out and "100 кг" in out
    assert advisor.ground_sentences(None, source) is None


def test_match_supplier_by_name_and_brand():
    suppliers = [sup(1, "Мясокомбинат Север"), sup(2, "ТМ «Вкусный Дом»", city="Казань")]
    assert advisor._match_supplier("Мясокомбинат Север", suppliers)["id"] == 1
    assert advisor._match_supplier("Мясокомбинат Север (Казань)", suppliers)["id"] == 1
    assert advisor._match_supplier("Вкусный Дом ООО", suppliers)["id"] == 2
    assert advisor._match_supplier("Неизвестная фирма", suppliers) is None


def test_product_in_card_detects_noun_and_ignores_city():
    s = sup(1, "Сыроварня", sub="Сыр", desc="сыры и масло", primary_products=["сыр"])
    assert advisor.product_in_card(s, "сыр моцарелла", None) is False
    assert advisor.product_in_card(s, "сыр", "Казань") is True
    assert advisor.product_in_card(s, "", None) is None
    s2 = {"name": "Завод", "subcategory": "", "description": "", "primary_products": "колбасы"}
    assert advisor.product_in_card(s2, "колбаса", None) is True


def test_ground_list_drops_unsupported_risks():
    assert advisor.ground(["ГОСТ 1", "ок"], "ок") == ["ок"]


def test_check_advice_rejects_unknown_company():
    suppliers = [sup(1, "A"), sup(2, "B")]
    assert advisor.check_advice({"best_choice": {"name": "Чужая"}}, suppliers, "src") is None


def test_check_advice_keeps_grounded_fields_and_flags_product():
    suppliers = [sup(1, "Мясокомбинат Север", desc="колбасы 100 кг"), sup(2, "Завод Юг", desc="сосиски")]
    data = {
        "best_choice": {"name": "Мясокомбинат Север", "reason": "Колбасы. Работает с ХАССП."},
        "alternative": {"name": "Завод Юг", "reason": "Резерв."},
        "risks": ["Уточнить 100 кг", "Нет данных про 999 кг"],
        "outreach_script": "Здравствуйте",
    }
    out = advisor.check_advice(data, suppliers, "Мясокомбинат Север: колбасы 100 кг", query="колбасы", city=None)
    assert out["best_choice"]["name"] == "Мясокомбинат Север"
    assert "ХАССП" not in out["best_choice"]["reason"]
    assert out["risks"] == ["Уточнить 100 кг"]
    assert out["best_choice"]["product_in_card"] is True
    assert out["alternative"]["name"] == "Завод Юг"


def test_check_advice_same_company_alternative_is_dropped():
    a = sup(1, "ТМ «Вкусный Дом»")
    b = sup(2, "ТМ «Вкусный Дом» филиал")
    data = {"best_choice": {"name": "ТМ «Вкусный Дом»"}, "alternative": {"name": "ТМ «Вкусный Дом» филиал"}}
    out = advisor.check_advice(data, [a, b], "src")
    assert out["alternative"]["name"] == "—"


def test_check_advice_no_alternative_given():
    suppliers = [sup(1, "A")]
    out = advisor.check_advice({"best_choice": {"name": "A"}}, suppliers, "src")
    assert out["alternative"]["reason"].startswith("Другой")


# --- LLM-вызовы через подменённый call_llm ---

def test_extract_extra_info_without_client_returns_defaults():
    assert advisor.extract_extra_info(None, sup(1, "A")) == advisor.DEFAULT_EXTRA


def test_extract_extra_info_grounds_values(monkeypatch):
    answer = json.dumps({"min_order": "от 100 кг", "price_range": "350 ₽/кг", "certificates": "ГОСТ 1",
                         "delivery_terms": "по РФ", "region_scope": ""})
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: answer)
    out = advisor.extract_extra_info(object(), sup(1, "A", desc="минимальный заказ 100 кг, доставка по РФ"))
    assert out["min_order"] == "от 100 кг"
    assert out["price_range"] == advisor.NOT_SPECIFIED  # цены в тексте нет
    assert out["certificates"] == advisor.NOT_SPECIFIED
    assert out["region_scope"] == advisor.NOT_SPECIFIED


def test_extract_extra_info_bad_answer_falls_back(monkeypatch):
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: "не json")
    assert advisor.extract_extra_info(object(), sup(1, "A")) == advisor.DEFAULT_EXTRA


def test_extract_extra_info_llm_error_falls_back(monkeypatch):
    def boom(c, p, max_tokens=0):
        raise OpenAIError("сбой")
    monkeypatch.setattr(advisor, "call_llm", boom)
    assert advisor.extract_extra_info(object(), sup(1, "A")) == advisor.DEFAULT_EXTRA


def test_enrich_suppliers_extra_saves_for_missing_min_order(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    suppliers = [{"id": 1, "name": "A", "min_order": None, "description": "мин. заказ 5 кг"}]
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "extract_extra_info", lambda c, s: {**advisor.DEFAULT_EXTRA, "min_order": "5 кг"})
    advisor.enrich_suppliers_extra(con, suppliers)
    assert suppliers[0]["min_order"] == "5 кг"
    assert con.execute("SELECT min_order FROM supplier_extra").fetchone()[0] == "5 кг"


def test_enrich_suppliers_extra_skips_done_and_handles_worker_error(con, monkeypatch):
    monkeypatch.setattr(advisor, "get_client", lambda: None)
    done = [{"id": 1, "min_order": "10 кг"}]
    advisor.enrich_suppliers_extra(con, done)  # все уже обогащены — ничего не делаем

    def bad(c, s):
        raise ValueError("битый ответ")
    todo = [{"id": 2, "name": "B", "min_order": None}]
    monkeypatch.setattr(advisor, "extract_extra_info", bad)
    advisor.enrich_suppliers_extra(con, todo)
    assert todo[0].get("min_order") is None or todo[0].get("min_order") == advisor.NOT_SPECIFIED


def test_recommend_best_supplier_empty_list():
    assert advisor.recommend_best_supplier([], "колбаса", "Казань") == {}


def test_recommend_best_supplier_without_llm_builds_fallback(no_llm):
    suppliers = [sup(1, "Лидер", distance_km=10, score=0.9), sup(2, "Резерв", distance_km=500)]
    out = advisor.recommend_best_supplier(suppliers, "колбаса", "Казань")
    assert out["best_choice"]["name"] == "Лидер"
    assert out["alternative"]["name"] == "Резерв"
    assert len(out["risks"]) == 2
    assert "колбаса" in out["outreach_script"]


def test_recommend_best_supplier_single_company_fallback(no_llm):
    out = advisor.recommend_best_supplier([sup(1, "Один")], None, None)
    assert out["alternative"]["name"] == "—"


def test_recommend_best_supplier_uses_llm_answer(monkeypatch):
    answer = json.dumps({"best_choice": {"name": "Мясокомбинат Север", "reason": "Колбасы."},
                         "alternative": {"name": "Завод Юг", "reason": "Резерв."},
                         "risks": ["Проверить 100 кг"], "outreach_script": "Здравствуйте"})
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: answer)
    suppliers = [sup(1, "Мясокомбинат Север", distance_km=5), sup(2, "Завод Юг", distance_km=40)]
    out = advisor.recommend_best_supplier(suppliers, "колбасы", "Казань")
    assert out["best_choice"]["name"] == "Мясокомбинат Север"
    assert out["outreach_script"] == "Здравствуйте"


def test_recommend_best_supplier_llm_invalid_name_falls_back(monkeypatch):
    answer = json.dumps({"best_choice": {"name": "Фантазия"}})
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: answer)
    out = advisor.recommend_best_supplier([sup(1, "Лидер")], "x", None)
    assert out["best_choice"]["name"] == "Лидер"


def test_recommend_best_supplier_llm_error_falls_back(monkeypatch):
    def boom(c, p, max_tokens=0):
        raise OpenAIError("нет")
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", boom)
    out = advisor.recommend_best_supplier([sup(1, "Лидер")], "x", None)
    assert out["best_choice"]["name"] == "Лидер"


def test_select_candidates_empty_and_no_llm(no_llm):
    assert advisor.select_candidates_via_llm([], "x", None)["selected_ids"] == []
    cands = [sup(1, "A", score=0.9), sup(2, "B"), sup(3, "C")]
    out = advisor.select_candidates_via_llm(cands, "колбаса", None)
    assert out["selected_ids"] == [1, 2, 3]


def test_select_candidates_parses_llm_ids_and_dedupes(monkeypatch):
    cands = [sup(1, "Север"), sup(2, "Юг"), sup(3, "ТМ «Двойник»", city="Казань"),
             sup(4, "ТМ «Двойник» филиал", city="Казань")]
    answer = json.dumps({"selected_ids": ["2", 1, 1, 3, 4, 999, "x"]})
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: answer)
    out = advisor.select_candidates_via_llm(cands, "колбаса", "Казань")
    assert out["selected_ids"] == [2, 1, 3]
    assert "Отобраны" in out["summary"]


def test_select_candidates_no_valid_ids_falls_back(monkeypatch):
    cands = [sup(1, "A"), sup(2, "B")]
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: json.dumps({"selected_ids": [777]}))
    out = advisor.select_candidates_via_llm(cands, "колбаса", None)
    assert set(out["selected_ids"]) == {1, 2}
    assert "не нашёл" in out["summary"]


def test_select_candidates_no_json_falls_back(monkeypatch):
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: "ничего")
    out = advisor.select_candidates_via_llm([sup(1, "A")], "колбаса", None)
    assert out["selected_ids"] == [1]


def test_select_candidates_llm_error_falls_back(monkeypatch):
    def boom(c, p, max_tokens=0):
        raise OpenAIError("сбой")
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "call_llm", boom)
    out = advisor.select_candidates_via_llm([sup(1, "A")], "колбаса", None)
    assert out["selected_ids"] == [1]


def test_fetch_full_page_text_extracts_main_block(monkeypatch):
    html = ("<html><body><nav>меню</nav><div class='content'><script>x</script>"
            "<p>Колбасы   варёные</p></div><footer>f</footer></body></html>")
    monkeypatch.setattr(live, "_fetch", lambda url: html)
    text = advisor.fetch_full_page_text("https://x")
    assert text == "Колбасы варёные"


def test_fetch_full_page_text_no_html(monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert advisor.fetch_full_page_text("https://x") == ""


def test_fetch_full_page_text_bad_markup_returns_empty(monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: "<html></html>")
    assert advisor.fetch_full_page_text("https://x") == ""


def test_passport_without_llm_saves_template(con, no_llm):
    s = sup(1, "Мясокомбинат", url="", min_order="10 кг")
    db.upsert(con, {"id": 1, "name": "Мясокомбинат", "url": "https://x/producers/1/m", "categories": []})
    out = advisor.get_or_create_b2b_passport(con, s)
    assert out["passport"]["commercial_terms"]["min_order"] == "10 кг"
    assert "warning" not in out
    assert db.get_passport(con, 1) is not None


def test_passport_uses_cache_unless_forced(con, no_llm, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    db.save_passport(con, 1, {"cached": True}, "текст")
    s = sup(1, "A")
    assert advisor.get_or_create_b2b_passport(con, s)["passport"] == {"cached": True}
    out = advisor.get_or_create_b2b_passport(con, s, force_refresh=True)
    assert "commercial_terms" in out["passport"]  # без LLM пересобрали шаблон


def test_passport_llm_failure_warns_and_keeps_cache(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "fetch_full_page_text", lambda url: "")
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: "не json")
    out = advisor.get_or_create_b2b_passport(con, sup(1, "A", desc="Колбасы"))
    assert "warning" in out
    assert "не собран" in out["warning"]
    assert db.get_passport(con, 1) is None  # шаблон не кэшируем


def test_passport_llm_failure_returns_old_cache(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    db.save_passport(con, 1, {"commercial_terms": {"min_order": "старый"}}, "т")
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "fetch_full_page_text", lambda url: "")

    def boom(c, p, max_tokens=0):
        raise OpenAIError("сбой")
    monkeypatch.setattr(advisor, "call_llm", boom)
    out = advisor.get_or_create_b2b_passport(con, sup(1, "A"), force_refresh=True)
    assert out["passport"]["commercial_terms"]["min_order"] == "старый"


def test_passport_llm_answer_is_grounded_and_cached(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    answer = {
        "commercial_terms": {"min_order": "100 кг", "packaging": "", "price_estimate": "350 ₽", "payment_terms": ""},
        "compliance_and_quality": {"standards": "ГОСТ 1", "certifications": "ХАССП", "shelf_life": "",
                                   "product_specs": "Колбасы"},
        "logistics": {"delivery_type": "по РФ", "storage_temp": ""},
        "buyer_verdict": {"green_flags": ["Колбасы"], "red_flags": [], "negotiation_questions": []},
    }
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "fetch_full_page_text", lambda url: "полный текст " * 20)
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: json.dumps(answer))
    s = sup(1, "A", desc="Колбасы, минимальный заказ 100 кг", primary_products=["колбасы"],
            specific_varieties=["варёная"], dish_ingredients=["пицца"])
    out = advisor.get_or_create_b2b_passport(con, s)
    p = out["passport"]
    assert p["commercial_terms"]["min_order"] == "100 кг"
    assert p["commercial_terms"]["price_estimate"] == advisor.NOT_SPECIFIED  # 350 ₽ нет в тексте
    assert p["compliance_and_quality"]["standards"] == advisor.NOT_SPECIFIED  # ГОСТ 1 нет в тексте
    assert p["logistics"]["storage_temp"] == advisor.NOT_SPECIFIED  # пустое дополнено шаблоном
    assert db.get_passport(con, 1) is not None
    assert con.execute("SELECT min_order FROM supplier_extra").fetchone()[0] == "100 кг"


def test_passport_llm_bad_sections_are_repaired(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://x/producers/1/a", "categories": []})
    answer = {"commercial_terms": "строка вместо словаря", "buyer_verdict": {"red_flags": [], "green_flags": [],
                                                                         "negotiation_questions": []}}
    monkeypatch.setattr(advisor, "get_client", lambda: object())
    monkeypatch.setattr(advisor, "fetch_full_page_text", lambda url: "")
    monkeypatch.setattr(advisor, "call_llm", lambda c, p, max_tokens=0: json.dumps(answer))
    out = advisor.get_or_create_b2b_passport(con, sup(1, "A"))
    assert isinstance(out["passport"]["commercial_terms"], dict)
    assert "logistics" in out["passport"]


def test_sign_letter_fills_known_and_drops_unknown_placeholders():
    text = ("Здравствуйте! Меня зовут [Ваше имя], я представляю компанию [Название вашей компании]. "
            "С уважением, [Ваше имя], [Должность], [Контактный телефон].")
    out = advisor.sign_letter(text, "Алексей (Старший закупщик)", "ООО «ФудТрейд»")
    assert "[" not in out
    assert "Меня зовут Алексей" in out and "компанию ООО «ФудТрейд»" in out
    assert out.endswith("С уважением, Алексей, старший закупщик.")


def test_speculative_risk_is_dropped():
    risk = "Специализация на БАД может означать, что мёд не будет полностью натуральным."
    assert advisor.speculative(risk)
    assert not advisor.speculative("Не указана минимальная партия и условия доставки.")


def test_quoted_does_not_double_quotes():
    assert advisor.quoted("Компания «Воял»") == "Компания «Воял»"
    assert advisor.quoted("Воял") == "«Воял»"


def test_kappa_and_recall_aware_ndcg():
    import eval_agreement, eval_search
    assert eval_agreement.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(eval_agreement.kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
    # одна найденная из трёх релевантных: раньше было 1.0, теперь пропуски штрафуются
    assert eval_search.ndcg([1, 0, 0]) == 1.0
    assert eval_search.ndcg([1, 0, 0], [1, 1, 1]) < 0.5

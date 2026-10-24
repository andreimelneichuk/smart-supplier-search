"""Ядро ранжирования (ranking.py): _rank во всех режимах, штрафы, близость, склейка компаний."""
import numpy as np
import pytest

import cross_encoder
import llm_rerank
import ranking
import search


def sup(id, name, sub="", desc="", products=None, city="Казань", **kw):
    base = {"id": id, "name": name, "subcategory": sub, "description": desc, "city": city,
            "categories": [], "phones": [], "emails": [], "products_count": 10, "slug": f"s{id}",
            "primary_products": products or [], "search_keywords": "", "is_food": None}
    base.update(kw)
    return base


COORDS = {"Казань": (55.79, 49.12), "Москва": (55.75, 37.62), "Самара": (53.20, 50.15)}


def coords(city, region=None):
    return COORDS.get(city)


CATS = {}


@pytest.fixture
def sample():
    return [
        sup(1, "Мясокомбинат Север", "Колбасы", "колбасы и сосиски", ["колбасы"], city="Казань"),
        sup(2, "Птицефабрика Восток", "Мясо птицы", "индейка и курица", ["индейка", "курица"], city="Самара"),
        sup(3, "Сыроварня Альпы", "Сыр", "сыр моцарелла", ["моцарелла"], city="Москва"),
        sup(4, "Оборудование Плюс", "Оборудование для производства колбас", "", city="Москва"),
    ]


# --- _rank в разных режимах ---

@pytest.mark.parametrize("model", ["lexical", "bm25"])
def test_rank_lexical_and_bm25_put_matching_company_first(sample, model):
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    out = ranking._rank(sample, parsed, CATS, None, model=model)
    assert out[0]["id"] == 1
    assert all("score" in r and "parts" in r for r in out)


def test_rank_with_city_adds_distance_and_geo_part(sample):
    parsed = {"terms": ["колбасы"], "city": "Казань", "raw": "колбасы Казань"}
    out = ranking._rank(sample, parsed, CATS, coords, model="lexical")
    top = out[0]
    assert top["id"] == 1 and top["distance_km"] == 0
    assert "geo" in top["parts"]


def test_rank_city_with_no_coords_gives_none_distance(sample):
    parsed = {"terms": ["колбасы"], "city": "Тверь", "raw": "колбасы Тверь"}
    out = ranking._rank(sample, parsed, CATS, coords, model="lexical")
    assert all(r["distance_km"] is None for r in out)


def test_rank_without_terms_returns_all_ranked_by_size_and_geo(sample):
    parsed = {"terms": [], "city": None, "raw": ""}
    out = ranking._rank(sample, parsed, CATS, None, model="lexical")
    assert len(out) == len(sample)  # оборудование без фильтра не отсекается по релевантности


def test_rank_skips_non_food_flagged_by_llm(sample):
    sample[0]["is_food"] = 0
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    ids = [r["id"] for r in ranking._rank(sample, parsed, CATS, None, model="lexical")]
    assert 1 not in ids


def test_rank_bm25_cross_uses_reranker_scores(sample, monkeypatch):
    monkeypatch.setattr(search, "cross_encoder_rerank",
                        lambda q, docs, idx: [(i, 5.0 if docs[i]["id"] == 3 else -5.0) for i in idx])
    monkeypatch.setattr(cross_encoder, "sigmoid", lambda x: 1 / (1 + np.exp(-x)))
    parsed = {"terms": ["сыр", "моцарелла"], "city": None, "raw": "сыр моцарелла"}
    out = ranking._rank(sample, parsed, CATS, None, model="bm25_cross")
    assert out[0]["id"] == 3


def test_rank_bm25_cross_without_candidates_skips_reranker(sample, monkeypatch):
    called = []
    monkeypatch.setattr(search, "cross_encoder_rerank", lambda *a: called.append(a) or [])
    parsed = {"terms": ["несуществующийтовар"], "city": None, "raw": "x"}
    out = ranking._rank(sample, parsed, CATS, None, model="bm25_cross")
    assert out == [] and called == []


def test_rank_llm_mode_uses_grades(sample, monkeypatch):
    def grade(raw, cands):
        return [3 if c["id"] == 2 else 0 for c in cands]
    monkeypatch.setattr(llm_rerank, "grade", grade)
    parsed = {"terms": ["индейка"], "city": None, "raw": "мясо индейки"}
    out = ranking._rank(sample, parsed, CATS, None, model="llm")
    assert out[0]["id"] == 2


def test_rank_llm_mode_none_grades_are_ignored(sample, monkeypatch):
    monkeypatch.setattr(llm_rerank, "grade", lambda raw, cands: [None] * len(cands))
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    assert ranking._rank(sample, parsed, CATS, None, model="llm") == []


def test_rank_llm_mode_empty_candidates(sample, monkeypatch):
    monkeypatch.setattr(llm_rerank, "grade", lambda raw, cands: [])
    parsed = {"terms": ["несуществующийтовар"], "city": None, "raw": "x"}
    assert ranking._rank(sample, parsed, CATS, None, model="llm") == []


# --- полный rank с синонимами ---

def test_rank_expands_synonyms_and_dedupes(monkeypatch):
    suppliers = [sup(1, "Тушёнка Мясная", "Мясные консервы", "тушёнка говяжья", ["мясные консервы"]),
                 sup(2, "Завод", "Тушёное мясо", "", ["тушеное мясо"])]
    parsed = {"terms": ["тушенка"], "city": None, "raw": "тушенка"}
    out = ranking.rank(suppliers, parsed, CATS, None, model="lexical")
    assert {r["id"] for r in out} >= {1}


def test_rank_single_variant_path(sample):
    parsed = {"terms": ["сыр"], "city": None, "raw": "сыр"}
    assert ranking.rank(sample, parsed, CATS, None, model="lexical")[0]["id"] == 3


# --- штрафы и признаки ---

def test_head_only_floor_for_modifier_query(monkeypatch):
    # «хлеб ржаной»: пекарня без слова «ржаной» не должна выпасть из выдачи
    bakery = sup(1, "Хлебозавод", "Хлеб", "хлеб", ["хлеб"])
    parsed = {"terms": ["хлеб", "ржаной"], "city": None, "raw": "хлеб ржаной"}
    out = ranking._rank([bakery], parsed, CATS, None, model="lexical")
    assert [r["id"] for r in out] == [1]


def test_non_food_clash_downweights_equipment_for_food_query():
    equip = sup(1, "Заводик", "Оборудование для колбас", "", [])
    rel = ranking.relevance(equip, ["колбасы"], CATS)
    assert ranking.non_food_clash(equip, {"колбаса"}) is True
    assert ranking.non_food_clash(equip, {"оборудование"}) is False
    assert rel >= 0


def test_not_food_detects_obvious_non_food():
    assert ranking.not_food("Чехлы для iPhone")
    assert not ranking.not_food("Колбасы варёные")
    assert not ranking.not_food(None)


def test_derived_clash_flags_snacks_for_raw_meat():
    snack = sup(1, "Снеки", "Мясные снеки", "", [])
    assert ranking.derived_clash(snack, {"мясо", "индейка"}) is True
    assert ranking.derived_clash(snack, {"снек"}) is False
    plain = sup(2, "Мясо", "Мясо птицы", "", [])
    assert ranking.derived_clash(plain, {"мясо"}) is False


def test_is_main_product_checks_specialization_and_name_only():
    s = sup(1, "Фабрика Сыра", "Сыр", "молоко и масло в описании", ["сыр"])
    assert ranking.is_main_product(s, "сыр") is True
    assert ranking.is_main_product(s, "молоко") is False
    assert ranking.is_main_product(s, "сыр", with_name=False) is True


def test_secondary_product_is_penalised(sample):
    # «колбасы» есть в описании сыроварни, но не в специализации — её балл ниже, чем у колбасного завода
    secondary = sup(9, "Сыроварня", "Сыр", "колбасы тоже делаем", [])
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    out = ranking._rank(sample[:1] + [secondary], parsed, CATS, None, model="lexical")
    scores = {r["id"]: r["parts"]["relevance"] for r in out}
    assert scores[1] > scores.get(9, 0)


def test_parent_category_rescues_poultry_factory_for_turkey_query(sample):
    parsed = {"terms": ["мясо", "индейки"], "city": None, "raw": "мясо индейки"}
    out = ranking._rank(sample, parsed, CATS, None, model="lexical")
    assert 2 in [r["id"] for r in out]


def test_apposition_parent_for_cheese_query(sample):
    parsed = {"terms": ["сыр", "моцарелла"], "city": None, "raw": "сыр моцарелла"}
    out = ranking._rank(sample, parsed, CATS, None, model="lexical")
    assert out[0]["id"] == 3


def test_site_bonus_raises_relevance_for_site_hits(sample):
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы", "site_ids": {1}}
    base = ranking._rank(sample, {**parsed, "site_ids": set()}, CATS, None, model="lexical")
    boosted = ranking._rank(sample, parsed, CATS, None, model="lexical")
    rel = lambda out: {r["id"]: r["parts"]["relevance"] for r in out}
    assert rel(boosted)[1] >= rel(base)[1]


def test_completeness_and_size_parts(sample):
    sample[0]["phones"] = ["+7 843 000-00-01"]
    sample[0]["emails"] = ["a@b.ru"]
    sample[0]["contacts_at"] = 1.0
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    top = ranking._rank(sample, parsed, CATS, None, model="lexical")[0]
    assert top["parts"]["completeness"] == 1.0
    assert 0 < top["parts"]["size"] <= 1.0


def test_unknown_contacts_are_not_penalised(sample):
    parsed = {"terms": ["колбасы"], "city": None, "raw": "колбасы"}
    top = ranking._rank(sample, parsed, CATS, None, model="lexical")[0]
    assert top["parts"]["completeness"] == 0.5


def test_nearest_reference_for_geo_score(sample):
    parsed = {"terms": ["сыр"], "city": "Москва", "raw": "сыр Москва"}
    out = ranking._rank(sample, parsed, CATS, coords, model="lexical")
    assert out[0]["distance_km"] == 0


# --- вспомогательные функции ---

def test_relevance_stem_empty_terms_is_one():
    assert ranking.relevance_stem(sup(1, "x"), [], CATS) == 1.0


def test_relevance_stem_weights_by_field():
    s = sup(1, "Колбасный цех", "Колбасы", "сосиски", [])
    assert ranking.relevance_stem(s, ["колбас"], CATS) == pytest.approx(ranking.W_NAME)
    assert ranking.relevance_stem(s, ["сосиск"], CATS) == pytest.approx(ranking.W_DESC)
    assert ranking.relevance_stem(s, ["рыба"], CATS) == 0.0


def test_reliability_grows_with_contacts_and_description():
    bare = sup(1, "x", desc="")
    full = sup(2, "y", desc="x" * 700, phones=["1"], emails=["a@b"])
    assert ranking.reliability(full, 1.0) > ranking.reliability(bare, 0.0)
    assert ranking.reliability(full, 1.0) == pytest.approx(0.35 + 0.25 + 0.25 + 0.15)


def test_brand_and_same_company_by_brand_in_same_city():
    a = sup(1, "ТМ «Вкусный Дом»", city="Казань")
    b = sup(2, "ТМ «Вкусный Дом» филиал", city="Казань")
    c = sup(3, "ТМ «Вкусный Дом»", city="Самара")
    assert ranking._brand(a) == ranking._brand(b)
    assert ranking.same_company(a, b) is True
    assert ranking.same_company(a, c) is False


def test_same_company_by_email_and_empty_brand():
    a = sup(1, "A", emails=["x@y.ru"])
    b = sup(2, "B", emails=["x@y.ru"])
    assert ranking.same_company(a, b) is True
    assert ranking._company_keys(sup(3, "")) == set()


def test_sort_results_distance_and_size_and_reliability():
    res = [
        {"id": 1, "distance_km": 100, "products_count": 5, "reliability": 0.2, "parts": {"relevance": 0.9}},
        {"id": 2, "distance_km": 50, "products_count": 50, "reliability": 0.9, "parts": {"relevance": 0.9}},
    ]
    assert [r["id"] for r in ranking.sort_results(res, "Ближе к городу")] == [2, 1]
    assert [r["id"] for r in ranking.sort_results(res, "Надёжнее")] == [2, 1]
    assert [r["id"] for r in ranking.sort_results(res, "Шире ассортимент")] == [2, 1]
    assert ranking.sort_results(res, "Лучшее совпадение") is res


def test_correct_terms_skips_short_and_known_words():
    suppliers = [{"subcategory": "Мясо", "primary_products": []}]
    assert ranking.correct_terms(["мяс"], suppliers) is None

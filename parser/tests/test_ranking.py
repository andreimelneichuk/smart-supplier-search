"""Разбор запроса, близость, сортировки, дедупликация и исправление опечаток (ranking.py)."""
import pytest

import ranking

CITIES = ["Казань", "Москва", "Санкт-Петербург", "Екатеринбург"]


def test_city_from_query_is_taken_out_of_terms():
    parsed = ranking.parse_query("сыр моцарелла Казань", CITIES)
    assert parsed["city"] == "Казань"
    assert parsed["terms"] == ["сыр", "моцарелла"]


def test_city_alias_and_declined_form():
    assert ranking.parse_query("мёд в Питере", CITIES)["city"] == "Санкт-Петербург"
    assert ranking.parse_query("сыр в Москве", CITIES)["city"] == "Москва"


def test_query_without_city_keeps_all_terms():
    parsed = ranking.parse_query("мясо индейки", CITIES)
    assert parsed["city"] is None
    assert parsed["terms"] == ["мясо", "индейки"]


def test_distance_to_same_city_is_zero_and_unknown_is_none():
    coords = lambda city, region: {"Казань": (55.79, 49.12), "Москва": (55.75, 37.62)}.get(city)
    assert ranking.distance_to({"city": "Казань"}, "Казань", (55.79, 49.12), coords) == 0.0
    assert ranking.distance_to({"city": "Москва"}, None, None, coords) is None
    assert ranking.distance_to({"city": "Москва"}, "Казань", None, coords) is None


def test_distance_to_other_city_is_positive_km():
    coords = lambda city, region: {"Казань": (55.79, 49.12), "Москва": (55.75, 37.62)}.get(city)
    d = ranking.distance_to({"city": "Москва"}, "Казань", (55.79, 49.12), coords)
    assert 700 < d < 760  # Казань — Москва по прямой ~720 км


def test_sort_puts_strong_relevance_first():
    results = [
        {"name": "near_weak", "distance_km": 5, "parts": {"relevance": 0.2}},
        {"name": "far_strong", "distance_km": 900, "parts": {"relevance": 0.9}},
    ]
    ordered = ranking.sort_results(results, "Ближе к городу")
    assert [r["name"] for r in ordered] == ["far_strong", "near_weak"]


def test_sort_by_distance_treats_unknown_as_last():
    results = [
        {"name": "unknown", "distance_km": None, "parts": {"relevance": 0.9}},
        {"name": "known", "distance_km": 300, "parts": {"relevance": 0.9}},
    ]
    assert [r["name"] for r in ranking.sort_results(results, "Ближе к городу")] == ["known", "unknown"]


def test_sort_best_match_keeps_order():
    results = [{"name": "a", "parts": {}}, {"name": "b", "parts": {}}]
    assert ranking.sort_results(results, "Лучшее совпадение") is results


def test_dedupe_merges_by_shared_phone_and_email():
    suppliers = [
        {"id": 1, "phones": ["+7 (843) 555-01-02"], "emails": []},
        {"id": 2, "phones": ["8 843 555 01 02"], "emails": []},  # тот же номер, другой формат
        {"id": 3, "phones": [], "emails": ["info@example.ru"]},
        {"id": 4, "phones": [], "emails": ["info@example.ru"]},
        {"id": 5, "phones": ["+7 900 000 00 00"], "emails": []},
    ]
    assert [s["id"] for s in ranking.dedupe(suppliers)] == [1, 3, 5]


def test_correct_terms_fixes_typo_from_catalog_vocabulary():
    suppliers = [{"subcategory": "Сыр моцарелла", "primary_products": ["моцарелла"]}] * 3
    assert ranking.correct_terms(["мацарела"], suppliers) == ["моцарелла"]


def test_correct_terms_returns_none_when_nothing_to_fix():
    suppliers = [{"subcategory": "Сыр моцарелла", "primary_products": []}]
    assert ranking.correct_terms(["моцарелла"], suppliers) is None


def test_query_variants_include_original_and_synonym():
    variants = ranking.query_variants(["тушенка"])
    assert ["тушенка"] in variants
    assert any("мясные" in v for v in variants)


@pytest.mark.parametrize("weights_sum", [ranking.WEIGHTS])
def test_weights_sum_to_one(weights_sum):
    assert sum(weights_sum.values()) == pytest.approx(1.0)

"""Лексический поиск и RRF (search.py) на небольшом наборе документов и на реальной базе."""
import numpy as np
import pytest

import db
import ranking
import search


def doc(name, subcategory="", description="", products=None):
    return {"name": name, "subcategory": subcategory, "description": description,
            "search_keywords": "", "primary_products": products or [], "categories": "",
            "city": "Казань", "phones": [], "emails": [], "products_count": 1}


def test_product_in_subcategory_beats_description_mention():
    docs = [
        doc("Фабрика А", subcategory="Колбасы", description="колбаса и другая продукция"),
        doc("Ферма Б", subcategory="Молоко", description="иногда делаем колбасу"),
    ]
    scores = search.lexical_scores(docs, ["колбаса"], {})
    assert scores[0] > scores[1]


def test_unrelated_document_scores_zero():
    docs = [doc("Молочный завод", subcategory="Молоко")]
    assert search.lexical_scores(docs, ["креветки"], {})[0] == 0


def test_rrf_rewards_documents_ranked_high_by_both_lists():
    a = np.array([0.9, 0.1, 0.5])
    b = np.array([0.8, 0.1, 0.2])
    fused = search.rrf(a, b)
    assert fused[0] > fused[2] > fused[1]


def test_rrf_returns_one_score_per_document():
    assert search.rrf(np.zeros(4), np.zeros(4)).shape == (4,)


@pytest.fixture(scope="module")
def real_suppliers():
    con = db.connect()
    rows = [dict(r) for r in con.execute("SELECT * FROM suppliers").fetchall()]
    for r in rows:
        r["primary_products"] = []
        r["search_keywords"] = r.get("search_keywords") or ""
    return rows


@pytest.mark.parametrize("query, expected_word", [
    ("моцарелла", "моцарелл"),
    ("мёд", "мед"),
])
def test_reference_queries_on_real_base_find_the_product(real_suppliers, query, expected_word):
    """Эталонные запросы: в топ-10 по лексике должна быть компания, у которой товар в названии или специализации."""
    terms = query.split()
    scores = search.lexical_scores(real_suppliers, terms, {})
    top = np.argsort(-scores)[:10]
    hits = [real_suppliers[i] for i in top if scores[i] > 0]
    assert hits, f"по «{query}» ничего не нашлось"
    assert any(expected_word in (h.get("subcategory") or "").lower().replace("ё", "е") or
               expected_word in (h.get("name") or "").lower().replace("ё", "е") for h in hits)

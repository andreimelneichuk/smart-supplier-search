"""Оставшиеся ветки search.py: веса слов, эмбеддинги с кэшем, BM25-кэш, переранжирование."""
import numpy as np
import pytest

import cross_encoder
import search


def doc(id, name, sub="", desc="", cats=None):
    return {"id": id, "name": name, "subcategory": sub, "description": desc, "search_keywords": "",
            "categories": cats or [], "primary_products": []}


# --- веса слова в поле ---

def test_field_match_without_head_weighting_is_full():
    assert search._field_match("курица", [("курица", "NOUN", "nom")], False) == 1.0


def test_field_match_compound_word_gets_compound_factor():
    toks = [("хлебокомбинат", "NOUN", "nom")]
    assert search._field_match("хлеб", toks, True) == pytest.approx(search.COMPOUND_FACTOR)


def test_field_match_head_word_is_full_weight():
    assert search._field_match("сыр", [("сыр", "NOUN", "nom")], True) == 1.0


def test_field_match_accessory_in_accusative_is_weak():
    toks = [("приправа", "NOUN", "nom"), ("курица", "NOUN", "acc")]
    assert search._field_match("курица", toks, True) == search.ACCESSORY_FACTOR


def test_field_match_non_food_head_is_weak():
    toks = [("аромат", "NOUN", "nom"), ("курица", "NOUN", "gen")]
    assert search._field_match("курица", toks, True) == search.ACCESSORY_FACTOR


def test_field_match_genitive_is_almost_full():
    toks = [("мясо", "NOUN", "nom"), ("курица", "NOUN", "gen")]
    assert search._field_match("курица", toks, True) == search.GENITIVE_FACTOR


def test_field_match_modifier_is_partial():
    toks = [("сыр", "NOUN", "nom"), ("курица", "ADJF", "nom")]
    assert search._field_match("курица", toks, True) == search.MODIFIER_FACTOR


def test_field_match_missing_word_is_zero():
    assert search._field_match("рыба", [("сыр", "NOUN", "nom")], True) == 0.0


def test_lexical_score_uses_weighted_fields_and_full_match_bonus():
    d = doc(1, "Колбасный цех", sub="Колбасы", desc="сосиски")
    assert search.lexical_score(d, ["колбаса"], {}) > 0
    assert search.lexical_score(d, ["рыба"], {}) == 0


def test_lexical_score_with_category_names():
    d = doc(1, "Завод", cats=["c1"])
    assert search.lexical_score(d, search.query_lemmas(["консервы"]), {"c1": "Консервы"}) > 0


def test_query_lemmas_and_content_lemmas_handle_empty():
    assert search.query_lemmas([]) == []
    assert search.content_lemmas(None) == []
    assert "колбаса" in search.query_lemmas(["колбасы"])


# --- эмбеддинги ---

class FakeModel:
    """Детерминированные векторы по первой букве текста — без загрузки настоящей модели."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            v[ord(t[0]) % 4] = 1.0
            v[(ord(t[0]) + 1) % 4] += 0.5
            yield v


@pytest.fixture
def fake_emb(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(search, "_model", model)
    monkeypatch.setattr(search, "EMB_CACHE", tmp_path / "emb.npz")
    return model


def test_get_model_loads_once(monkeypatch):
    created = []

    class TE:
        def __init__(self, name):
            created.append(name)

    import sys
    import types
    fake_mod = types.ModuleType("fastembed")
    fake_mod.TextEmbedding = TE
    monkeypatch.setitem(sys.modules, "fastembed", fake_mod)
    monkeypatch.setattr(search, "_model", None)
    search._get_model()
    search._get_model()
    assert created == [search.MODEL]
    monkeypatch.setattr(search, "_model", None)


def test_doc_text_truncates_description():
    text = search.doc_text(doc(1, "Имя", sub="Сыр", desc="x" * 500))
    assert text.startswith("Сыр. Имя. ")
    assert len(text) < 260


def test_doc_matrix_caches_embeddings_on_disk(fake_emb):
    docs = [doc(1, "Колбасы"), doc(2, "Сыр")]
    m1 = search.doc_matrix(docs)
    calls_after_first = fake_emb.calls
    m2 = search.doc_matrix(docs)  # второй раз — из файла кэша
    assert fake_emb.calls == calls_after_first
    assert np.allclose(m1, m2)
    assert search.EMB_CACHE.exists()


def test_doc_matrix_embeds_only_new_documents(fake_emb):
    search.doc_matrix([doc(1, "Колбасы")])
    before = fake_emb.calls
    search.doc_matrix([doc(1, "Колбасы"), doc(2, "Сыр")])
    assert fake_emb.calls == before + 1


def test_dense_cosine_returns_one_score_per_doc(fake_emb):
    scores = search.dense_cosine([doc(1, "Колбасы"), doc(2, "Сыр")], "колбаса")
    assert scores.shape == (2,)
    assert scores[0] >= scores[1] - 1e-6 or scores[0] <= 1.0


# --- BM25 с кэшем ---

def test_get_bm25_reuses_index_for_same_content():
    docs = [doc(1, "Колбасы", sub="Колбасы"), doc(2, "Сыр", sub="Сыр")]
    idx1 = search.get_bm25(docs, {})
    idx2 = search.get_bm25(list(docs), {})
    assert idx1 is idx2


def test_get_bm25_rebuilds_when_description_changes():
    docs = [doc(1, "Колбасы", sub="Колбасы")]
    idx1 = search.get_bm25(docs, {})
    docs2 = [dict(docs[0], description="Сыр тоже")]
    assert search.get_bm25(docs2, {}) is not idx1


def test_bm25_scores_orders_matching_document_first():
    docs = [doc(1, "Колбасный цех", sub="Колбасы"), doc(2, "Сыроварня", sub="Сыр")]
    scores = search.bm25_scores(docs, ["колбасы"], {})
    assert scores[0] > scores[1]


# --- переранжирование ---

def test_cross_encoder_rerank_delegates(monkeypatch):
    captured = {}

    def fake(query, docs, idx):
        captured["args"] = (query, len(docs), idx)
        return [(idx[0], 0.9)]

    monkeypatch.setattr(cross_encoder, "rerank_candidates", fake)
    out = search.cross_encoder_rerank("колбаса", [doc(1, "A"), doc(2, "B")], [1])
    assert out == [(1, 0.9)]
    assert captured["args"] == ("колбаса", 2, [1])


def test_rrf_with_ties_and_single_list():
    out = search.rrf(np.array([1.0, 1.0, 0.0]))
    assert out.shape == (3,)
    assert out[2] < out[0]

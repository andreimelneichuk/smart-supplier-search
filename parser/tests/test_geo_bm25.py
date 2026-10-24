"""Координаты и справочник городов (geo.py) и индекс BM25 (bm25.py)."""
import csv
import sqlite3

import numpy as np
import pytest

import bm25
import geo

FIELDS = ["region", "region_type", "city", "settlement", "geo_lat", "geo_lon", "population"]
ROWS = [
    {"region": "Пермский", "region_type": "край", "city": "Пермь", "settlement": "", "geo_lat": "58.01",
     "geo_lon": "56.22", "population": "1000000"},
    {"region": "Московская", "region_type": "обл", "city": "Серпухов", "settlement": "", "geo_lat": "54.91",
     "geo_lon": "37.41", "population": "100000"},
    {"region": "Ямало-Ненецкий", "region_type": "АО", "city": "Салехард", "settlement": "", "geo_lat": "66.53",
     "geo_lon": "66.61", "population": "40000"},
    {"region": "Тульская", "region_type": "обл", "city": "Ленино", "settlement": "", "geo_lat": "",
     "geo_lon": "", "population": "0"},
    {"region": "Карелия", "region_type": "Респ", "city": "", "settlement": "Пряжа", "geo_lat": "61.8",
     "geo_lon": "33.5", "population": "100"},
    {"region": "Ханты", "region_type": "АО", "city": "Ханты", "settlement": "", "geo_lat": "61.0",
     "geo_lon": "69.0", "population": "5"},
]


@pytest.fixture
def csv_db(tmp_path, monkeypatch):
    path = tmp_path / "city.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ROWS)
    monkeypatch.setattr(geo, "CSV_PATH", path)
    monkeypatch.setattr(geo, "_static", None)
    monkeypatch.setattr(geo, "_regions", [])
    monkeypatch.setattr(geo.time, "sleep", lambda s: None)
    con = sqlite3.connect(tmp_path / "geo.db")
    con.row_factory = sqlite3.Row
    yield con
    con.close()


def test_norm_lowercases_and_replaces_yo():
    assert geo.norm("  Ёлка ") == "елка"
    assert geo.norm(None) == ""


def test_load_static_builds_city_index_and_regions(csv_db):
    st = geo._load_static()
    assert "пермь" in st
    assert ("Пермский край" in dict(geo._regions).values())
    assert "москва" in st  # ручные координаты федеральных городов
    assert geo._load_static() is st  # второй вызов — из памяти


def test_region_type_variants_produce_full_names(csv_db):
    geo._load_static()
    names = dict(geo._regions).values()
    assert "Московская область" in names
    assert "Ямало-Ненецкий автономный округ" in names
    assert "Республика Карелия" in names


def test_region_nominative_converts_genitive_to_nominative(csv_db):
    assert geo.region_nominative("Пермского края") == "Пермский край"
    assert geo.region_nominative(None) is None
    assert geo.region_nominative("Марсианская") == "Марсианская"


def test_region_matches_and_stem():
    assert geo._region_matches("пермский", "Пермского края") is True
    assert geo._region_matches("", "что угодно") is False
    assert geo._region_matches("Москва", "Московская область") is False
    assert geo._stem("Московская") == "Московск"
    assert geo._stem("Пермь") == "Пермь"


def test_cached_coords_without_city_is_none(csv_db):
    assert geo.cached_coords(csv_db, None) is None


def test_cached_coords_offline_without_region_takes_biggest_namesake(csv_db):
    lat, lon = geo.cached_coords(csv_db, "Пермь")
    assert lat == pytest.approx(58.01)


def test_cached_coords_region_must_match_namesake(csv_db):
    assert geo.cached_coords(csv_db, "Серпухов", "Московская область") is not None
    assert geo.cached_coords(csv_db, "Серпухов", "Пермского края") is None


def test_cached_coords_moscow_region_equals_city(csv_db):
    assert geo.cached_coords(csv_db, "Москва", "Москва") is not None


def test_cached_coords_uses_geo_cache_table(csv_db):
    geo._ensure_cache(csv_db)
    csv_db.execute("INSERT INTO geo_cache VALUES (?,?,?)", (geo._cache_key("Нигде", "Тула"), 54.2, 37.6))
    assert geo.cached_coords(csv_db, "Нигде", "Тула") == (54.2, 37.6)
    csv_db.execute("INSERT INTO geo_cache VALUES (?,?,?)", (geo._cache_key("Пусто", None), None, None))
    assert geo.cached_coords(csv_db, "Пусто") is None


def test_cache_key_contains_region_prefix():
    assert geo._cache_key("Тула", "Тульская область") == "тула|тульская область"
    assert geo._cache_key("Тула", None) == "тула"


def test_geocode_returns_cached_offline_hit(csv_db, monkeypatch):
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: pytest.fail("сеть не нужна"))
    assert geo.geocode(csv_db, "Пермь") is not None


def test_geocode_queries_nominatim_and_caches(csv_db, monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return [{"lat": "44.0", "lon": "40.0"}]

    calls = []
    monkeypatch.setattr(geo.requests, "get", lambda url, params, headers, timeout: calls.append(params) or Resp())
    assert geo.geocode(csv_db, "Новгородок", "Тульская область") == (44.0, 40.0)
    assert calls[0]["q"].startswith("Новгородок, Тульская область, Россия")
    # повторный вызов берёт результат из кэша и сеть не трогает
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: pytest.fail("кэш должен сработать"))
    assert geo.geocode(csv_db, "Новгородок", "Тульская область") == (44.0, 40.0)


def test_geocode_not_found_is_cached_as_none(csv_db, monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return []

    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: Resp())
    assert geo.geocode(csv_db, "Ничейный") is None
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: pytest.fail("уже пробовали"))
    assert geo.geocode(csv_db, "Ничейный") is None


def test_geocode_http_error_not_cached(csv_db, monkeypatch):
    class Resp:
        status_code = 503

    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: Resp())
    assert geo.geocode(csv_db, "Сбойный") is None
    assert csv_db.execute("SELECT COUNT(*) FROM geo_cache").fetchone()[0] == 0


def test_geocode_network_error_returns_none(csv_db, monkeypatch):
    def boom(*a, **k):
        raise geo.requests.RequestException("нет сети")

    monkeypatch.setattr(geo.requests, "get", boom)
    assert geo.geocode(csv_db, "Сеть") is None


def test_geocode_waits_between_requests(csv_db, monkeypatch):
    waits = []
    monkeypatch.setattr(geo.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(geo, "_last_request", geo.time.time())

    class Resp:
        status_code = 200

        def json(self):
            return []

    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: Resp())
    geo.geocode(csv_db, "Ждалка")
    assert waits and waits[0] > 0


def test_distance_km_moscow_spb_and_symmetry():
    d = geo.distance_km((55.7558, 37.6173), (59.9343, 30.3351))
    assert 620 < d < 650
    assert geo.distance_km((1, 2), (1, 2)) == pytest.approx(0)


def test_warm_prints_and_geocodes_missing_pairs(tmp_path, monkeypatch, csv_db, capsys):
    con = sqlite3.connect(tmp_path / "w.db")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE suppliers (city TEXT, region TEXT)")
    con.execute("INSERT INTO suppliers VALUES ('Далёкий', 'Тульская область')")
    con.execute("INSERT INTO suppliers VALUES ('Пермь', 'Пермский край')")
    monkeypatch.setattr(geo, "geocode", lambda c, city, region=None: None)
    geo.warm(con)
    out = capsys.readouterr().out
    assert "Пар город+регион: 2" in out
    assert "Найдено: 0 из 1" in out or "Найдено: 0 из 2" in out


# --- BM25 ---

def doc(name, sub="", desc="", keywords="", cats=None):
    return {"name": name, "subcategory": sub, "description": desc, "search_keywords": keywords,
            "categories": cats or []}


def test_tokenize_lemmas_drops_stopwords_and_prepositions():
    assert bm25.tokenize_lemmas(None) == []
    toks = bm25.tokenize_lemmas("Колбасы для завода")
    assert "колбаса" in toks
    assert "для" not in toks


def test_is_compound_of_detects_compound_words():
    assert bm25.is_compound_of("хлебокомбинат", "хлеб")
    assert not bm25.is_compound_of("сырьё", "сыр")  # короткая основа не расширяется
    assert not bm25.is_compound_of("хлеб", "хлеб")


def test_bm25_index_scores_relevant_document_higher():
    docs = [doc("Хлебозавод", sub="Хлеб", desc="ржаной хлеб"),
            doc("Молочный", sub="Молоко", desc="молоко и сливки")]
    idx = bm25.BM25Index(docs, {})
    scores = idx.score(["хлеб"])
    assert scores[0] > scores[1] == 0
    assert idx.score([]).sum() == 0


def test_bm25_compound_terms_match_with_lower_weight():
    docs = [doc("Хлебокомбинат", sub="Хлебобулочные изделия"), doc("Сыр", sub="Сыр")]
    idx = bm25.BM25Index(docs, {})
    s = idx.score(["хлеб"])
    assert s[0] > 0 and s[1] == 0


def test_bm25_unknown_term_scores_zero_and_idf_positive():
    docs = [doc("A", sub="Сыр"), doc("B", sub="Масло")]
    idx = bm25.BM25Index(docs, {})
    assert idx.score(["несуществующее"]).sum() == 0
    assert idx.idf("сыр") > 0


def test_bm25_uses_category_names():
    docs = [doc("A", cats=["cat-1"])]
    idx = bm25.BM25Index(docs, {"cat-1": "Консервы"})
    assert idx.score(bm25.tokenize_lemmas("Консервы"))[0] > 0


def test_bm25_empty_collection_has_safe_averages():
    idx = bm25.BM25Index([], {})
    assert idx.N == 0
    assert idx.score(["сыр"]).shape == (0,)
    assert isinstance(idx.score(["сыр"]), np.ndarray)

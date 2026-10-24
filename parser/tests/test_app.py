"""Smoke-тесты интерфейса app.py через streamlit.testing.v1.AppTest.

Модели поиска и сеть подменены, база — временная с несколькими карточками.
"""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import cross_encoder
import db
import live
import search

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeEmbedder:
    def embed(self, texts):
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            v[sum(map(ord, t)) % 8] = 1.0
            yield v


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(search, "_model", FakeEmbedder())
    monkeypatch.setattr(search, "EMB_CACHE", tmp_path / "emb.npz")
    monkeypatch.setattr(cross_encoder, "get_reranker", lambda: None)
    monkeypatch.setattr(live, "search_live", lambda con, q, **kw: 0)
    monkeypatch.setattr(live, "enrich_contacts", lambda con, ids: None, raising=False)
    monkeypatch.setattr(live, "fetch_products", lambda url: [], raising=False)
    con = db.connect()
    # таблицу создаёт enrich_catalog.init_db, а не db.connect: повторяем её схему, как в реальной базе
    con.execute("CREATE TABLE IF NOT EXISTS enriched_catalog (supplier_id INTEGER PRIMARY KEY, "
                "is_food INTEGER NOT NULL DEFAULT 1, primary_products TEXT NOT NULL DEFAULT '[]', "
                "specific_varieties TEXT NOT NULL DEFAULT '[]', processing TEXT NOT NULL DEFAULT '[]', "
                "dish_ingredients TEXT NOT NULL DEFAULT '[]', search_keywords TEXT, raw_json TEXT, "
                "enriched_at REAL NOT NULL DEFAULT 0)")
    for sid, name, sub, desc, city in [
        (1, "Сыроварня Север", "Сыр", "Моцарелла и сулугуни оптом", "Тула"),
        (2, "Молочный двор", "Молоко", "Молоко и сливки", "Казань"),
        (3, "Мясокомбинат Юг", "Колбасы", "Колбасы варёные копчёные", "Краснодар"),
    ]:
        con.execute("INSERT INTO suppliers (id, name, url, subcategory, description, city, added_at, categories) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1700000000, '[]')",
                    (sid, name, f"https://x/producers/{sid}/s{sid}", sub, desc, city))
    con.commit()
    con.close()
    return tmp_path


def run_app(timeout=60):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    return at


def test_app_starts_without_exceptions(app_env):
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]


def test_app_shows_sidebar_with_db_caption(app_env):
    at = run_app()
    assert not at.exception
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "3" in captions  # в подписи сайдбара есть число карточек из базы


def test_app_search_by_query_shows_result(app_env, monkeypatch):
    monkeypatch.setattr(search, "_model", FakeEmbedder())
    at = run_app()
    assert not at.exception
    at.text_input(key="q").input("сыр").run()
    assert not at.exception
    all_text = " ".join(m.value for m in at.markdown)
    assert "Сыроварня Север" in all_text

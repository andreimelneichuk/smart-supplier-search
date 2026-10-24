"""Служебные скрипты: dbstats.py (покрытие полей) и backfill_regions.py (дозаполнение регионов)."""
import sqlite3

import pytest

import backfill_regions
import db
import dbstats
import geo
import productcenter_parser as pc


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    c = db.connect()
    yield c
    c.close()


def test_has_text_treats_empty_placeholders_as_missing():
    assert dbstats._has_text("+7 900")
    for v in (None, "", "[]", "{}", "null", "None", "   "):
        assert not dbstats._has_text(v) or v == "   "


def test_price_and_min_order_regexes():
    assert dbstats.PRICE_RE.search("Цена 300 ₽")
    assert dbstats.MIN_ORDER_RE.search("минимальный заказ 10 кг")
    assert dbstats.MIN_ORDER_RE.search("от 50 кг")
    assert not dbstats.MIN_ORDER_RE.search("ничего")


ENRICHED_DDL = """
    CREATE TABLE IF NOT EXISTS enriched_catalog (
        supplier_id INTEGER PRIMARY KEY,
        is_food INTEGER NOT NULL DEFAULT 1,
        primary_products TEXT NOT NULL DEFAULT '[]'
    )"""


def test_coverage_empty_base_is_all_zero(con):
    con.execute(ENRICHED_DDL)
    fields, fresh = dbstats.coverage(con)
    assert set(fields.values()) == {0.0}
    assert fresh == {"всего": 0, "последнее добавление": None, "контакты обновлялись": 0}


def test_coverage_counts_fields_and_freshness(con):
    con.execute(ENRICHED_DDL)
    con.execute(
        "INSERT INTO suppliers (id, name, url, city, phones, emails, description, products_count, categories, "
        "contacts_at, added_at) VALUES (1, 'A', 'u1', 'Казань', '[\"+7\"]', '[]', ?, 5, '[\"c\"]', 1700000000, 1700000000)",
        ("Цена указана. Минимальный заказ 10 кг. " + "x" * 100,))
    con.execute(
        "INSERT INTO suppliers (id, name, url, city, phones, emails, description, products_count, categories, "
        "contacts_at, added_at) VALUES (2, 'B', 'u2', NULL, '[]', '[]', '', 0, '[]', NULL, 0)")
    con.execute("INSERT INTO enriched_catalog (supplier_id, primary_products) VALUES (1, '[\"сыр\"]')")
    con.commit()

    fields, fresh = dbstats.coverage(con)
    assert fields["город"] == 0.5
    assert fields["телефон"] == 0.5
    assert fields["e-mail"] == 0.0
    assert fields["описание (от 100 знаков)"] == 0.5
    assert fields["ассортимент (число товаров)"] == 0.5
    assert fields["основные товары (LLM-разметка)"] == 0.5
    assert fields["контакты проверены с сайта"] == 0.5
    assert fields["упомянута цена (текст карточки)"] == 0.5
    assert fields["упомянут минимальный заказ (текст)"] == 0.5
    assert fresh["всего"] == 2
    assert fresh["контакты обновлялись"] == 1
    assert fresh["последнее добавление"] is not None


def test_coverage_freshness_empty_added(con):
    con.execute(ENRICHED_DDL)
    con.execute("INSERT INTO suppliers (id, name, url, added_at) VALUES (1, 'A', 'u1', 0)")
    con.commit()
    _, fresh = dbstats.coverage(con)
    assert fresh["последнее добавление"] is None


def test_dbstats_main_prints_table(con, monkeypatch, capsys):
    con.execute(ENRICHED_DDL)
    con.execute("INSERT INTO suppliers (id, name, url, city, added_at) VALUES (1, 'A', 'u1', 'Тула', 1700000000)")
    con.commit()
    monkeypatch.setattr(dbstats.db, "connect", lambda: con)
    dbstats.main()
    out = capsys.readouterr().out
    assert "| поле | доля карточек |" in out
    assert "Всего карточек: 1" in out


# --- backfill_regions ---

def test_backfill_main_runs_all_three_steps(con, monkeypatch, capsys):
    con.execute("INSERT INTO suppliers (id, name, url, region_slug, region, added_at) "
                "VALUES (1, 'A', 'u1', 'tula', NULL, 0)")
    con.commit()
    monkeypatch.setattr(backfill_regions.db, "connect", lambda: con)
    monkeypatch.setattr(backfill_regions.time, "sleep", lambda s: None)
    monkeypatch.setattr(pc, "CATEGORIES", ["sir"])
    monkeypatch.setattr(backfill_regions.live, "_fetch", lambda url: "<html>")
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: [
        {"id": 2, "name": "B", "url": "https://x/producers/2/b", "categories": []}])
    monkeypatch.setattr(backfill_regions.db, "region_name", lambda c, s, f: "Тульская область")
    warmed = []
    monkeypatch.setattr(geo, "warm", lambda c: warmed.append(True))

    backfill_regions.main()

    out = capsys.readouterr().out
    assert "1/3" in out and "2/3" in out and "3/3" in out
    assert "tula -> Тульская область" in out
    assert warmed == [True]
    assert "из 2 компаний" in out
    assert con.execute("SELECT COUNT(*) FROM suppliers WHERE id = 2").fetchone()[0] == 1


def test_backfill_reports_missing_listing(con, monkeypatch, capsys):
    monkeypatch.setattr(backfill_regions.db, "connect", lambda: con)
    monkeypatch.setattr(backfill_regions.time, "sleep", lambda s: None)
    monkeypatch.setattr(pc, "CATEGORIES", ["sir"])
    monkeypatch.setattr(backfill_regions.live, "_fetch", lambda url: None)
    monkeypatch.setattr(geo, "warm", lambda c: None)
    backfill_regions.main()
    assert "нет ответа: sir" in capsys.readouterr().out

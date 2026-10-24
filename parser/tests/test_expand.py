"""Масштабное расширение базы (expand_catalog.py). Сеть и пауза подменены, база — временная."""
import pytest

import db
import expand_catalog as ex
import live
import productcenter_parser as pc


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "x.db")
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    c = db.connect()
    yield c
    c.close()


def item(i):
    return {"id": i, "name": f"Компания {i}", "url": f"https://x/producers/{i}/c{i}", "categories": []}


def test_expand_category_stops_after_two_empty_responses(con, monkeypatch):
    calls = []
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: calls.append(url) or None)
    assert ex.expand_category_pages(con, "miaso-203", "Мясо", 10) == 0
    assert len(calls) == 2


def test_expand_category_stops_when_no_cards(con, monkeypatch):
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: "<html>")
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: [])
    assert ex.expand_category_pages(con, "miaso-203", "Мясо", 10) == 0


def test_expand_category_adds_new_and_follows_pages(con, monkeypatch):
    urls = []
    pages = {1: [item(1), item(2)], 2: [item(2), item(3)]}

    def fetch(url, timeout):
        urls.append(url)
        return "<html>"

    monkeypatch.setattr(pc, "fetch", fetch)
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: pages[len(urls)])
    added = ex.expand_category_pages(con, "miaso-203", "Мясо", 2)
    assert added == 3  # id 2 повторяется — не считается новой
    assert urls[0].endswith("/catalog-miaso-203")
    assert urls[1].endswith("/catalog-miaso-203/page-2")
    assert con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 3


def test_expand_regions_tags_region_and_stops_on_short_page(con, monkeypatch):
    monkeypatch.setattr(ex, "TARGET_REGIONS", [("tula-1", "Тульская область")])
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: "<html>")
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: [item(7), item(8)])  # меньше 40 — последняя страница
    total = ex.expand_regions(con)
    assert total == 2
    assert con.execute("SELECT region_slug FROM suppliers WHERE id = 7").fetchone()[0] == "tula-1"


def test_expand_regions_breaks_on_no_response(con, monkeypatch):
    monkeypatch.setattr(ex, "TARGET_REGIONS", [("tula-1", "Тульская")])
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: None)
    assert ex.expand_regions(con) == 0


def test_expand_regions_breaks_on_empty_listing(con, monkeypatch):
    monkeypatch.setattr(ex, "TARGET_REGIONS", [("tula-1", "Тульская")])
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: "<html>")
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: [])
    assert ex.expand_regions(con) == 0


def test_expand_regions_full_page_continues(con, monkeypatch):
    monkeypatch.setattr(ex, "TARGET_REGIONS", [("tula-1", "Тульская")])
    monkeypatch.setattr(pc, "fetch", lambda url, timeout: "<html>")
    full = [item(i) for i in range(40)]
    pages = [full, []]
    monkeypatch.setattr(pc, "parse_listing", lambda html, slug: pages.pop(0))
    assert ex.expand_regions(con) == 40


def test_expand_targeted_queries_sums_and_reports_errors(con, monkeypatch, capsys):
    def search(c, q, pages):
        if q == TWO_BAD:
            raise RuntimeError("сеть упала")
        return 2

    TWO_BAD = ex.TARGET_QUERIES[1]
    monkeypatch.setattr(ex, "TARGET_QUERIES", ex.TARGET_QUERIES[:3])
    monkeypatch.setattr(live, "search_live", search)
    total = ex.expand_targeted_queries(con)
    assert total == 4
    assert "Ошибка при поиске" in capsys.readouterr().out


def test_main_runs_all_stages(con, monkeypatch, capsys):
    monkeypatch.setattr(ex.db, "connect", lambda: con)
    seen = []
    monkeypatch.setattr(ex, "expand_targeted_queries", lambda c: seen.append("q") or 1)
    monkeypatch.setattr(ex, "expand_regions", lambda c: seen.append("r") or 2)
    monkeypatch.setattr(ex, "CATEGORY_LIMITS", {"miaso-203": 1})
    monkeypatch.setattr(pc, "CATEGORIES", {"miaso-203": "Мясо"})
    monkeypatch.setattr(ex, "expand_category_pages", lambda c, s, n, m: seen.append("c") or 3)
    ex.main()
    assert seen == ["q", "r", "c"]
    out = capsys.readouterr().out
    assert "РАСШИРЕНИЕ ЗАВЕРШЕНО" in out
    assert "Добавлено по разделам: +3" in out

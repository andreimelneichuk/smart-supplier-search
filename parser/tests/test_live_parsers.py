"""Живой поиск на сайте (live.py) и парсеры каталогов (productcenter_parser.py, agroserver_parser.py).
Сеть не трогаем: HTTP подменён, HTML — маленькие фикстуры по структуре страниц."""
import json
import sys
from urllib.parse import unquote

import pytest

import agroserver_parser as agro
import db
import live
import productcenter_parser as pc

PRODUCT_SEARCH = """
<div class="card_item product">
  <div class="ii_company" title="Производитель: Мясокомбинат Север">
    <a href="/producers/1/miasokombinat-sever">x</a></div>
  <div class="text"><a class="link" href="/products/9/kolbasa">Колбаса варёная</a></div>
  <div class="ii_city"><a href="/r-altaiskii-krai-155/x">Барнаул</a></div>
  <div class="item_descriptor">Мясные изделия</div>
</div>
<div class="card_item product">
  <div class="ii_company"><a href="/producers/1/miasokombinat-sever">x</a></div>
  <div class="text"><a class="link" href="/products/10/sosiski">Сосиски</a></div>
</div>
<div class="card_item product">
  <div class="text"><a class="link" href="/products/11/nothing">Без компании</a></div>
</div>
"""

FIRM_SEARCH = """
<div class="card_item firm">
  <div class="text"><a class="link" href="/producers/2/syrovarnia">Сыроварня</a></div>
  <div class="ii_city"><a href="/r-tulskaia-70/x">Тула</a></div>
  <div class="item_descriptor">Сыры</div>
  <div class="ii_goods">12</div>
</div>
<div class="card_item firm">
  <div class="text"><a class="link" href="/producers/3/bez-goods">Без товаров</a></div>
  <div class="ii_goods">много</div>
</div>
<div class="card_item firm">
  <div class="text"><a class="link" href="/producers/1/miasokombinat-sever">Уже есть</a></div>
</div>
<div class="card_item firm">
  <div class="text"><span>без ссылки</span></div>
</div>
"""

PRODUCT_CARDS = """
<div class="card_item product">
  <a class="link" href="/products/5/kolbasa">Колбаса</a>
  <div class="item_price"><b>250</b></div>
  <div class="image"><img src="/img/k.jpg"></div>
</div>
<div class="card_item product">
  <a class="link" href="/products/5/kolbasa">Дубль</a>
</div>
<div class="card_item product">
  <a class="link" href="https://other.ru/products/6/x">Внешняя</a>
  <div class="image"><img src="https://img.ru/a.jpg"></div>
</div>
<div class="card_item product">
  <span>без ссылки</span>
</div>
"""


@pytest.fixture(autouse=True)
def reset_live_state(monkeypatch):
    monkeypatch.setattr(live, "_fails", 0)
    monkeypatch.setattr(live, "_blocked_until", 0.0)
    monkeypatch.setattr(live, "_search_cache", {})
    monkeypatch.setattr(live.time, "sleep", lambda s: None)
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    monkeypatch.setattr(agro.time, "sleep", lambda s: None)


class Resp:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text
        self.encoding = None


def fake_get(responses):
    def get(url, timeout=None):
        r = responses(url) if callable(responses) else responses
        if isinstance(r, Exception):
            raise r
        return r
    return get


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "live.db")
    c = db.connect()
    yield c
    c.close()


# --- live: HTTP и автопауза ---

def test_fetch_returns_html_and_resets_failures(monkeypatch):
    monkeypatch.setattr(live.pc.session, "get", fake_get(Resp(200, "<p>ok</p>")))
    live._fails = 2
    assert live._fetch("https://x") == "<p>ok</p>"
    assert live._fails == 0


def test_fetch_non_200_counts_as_failure(monkeypatch):
    monkeypatch.setattr(live.pc.session, "get", fake_get(Resp(404)))
    assert live._fetch("https://x") is None
    assert live._fails == 1


def test_fetch_network_error_is_logged_and_counted(monkeypatch):
    import requests
    monkeypatch.setattr(live.pc.session, "get", fake_get(requests.RequestException("обрыв")))
    assert live._fetch("https://x") is None
    assert live._fails == 1


def test_three_failures_block_live_requests(monkeypatch):
    import requests
    monkeypatch.setattr(live.pc.session, "get", fake_get(requests.RequestException("нет")))
    for _ in range(live.BLOCK_AFTER):
        live._fetch("https://x")
    assert live.is_blocked()
    monkeypatch.setattr(live.pc.session, "get", fake_get(Resp(200, "ok")))
    assert live._fetch("https://x") is None  # на паузе сеть не трогаем


def test_block_expires(monkeypatch):
    monkeypatch.setattr(live, "_blocked_until", 0.0)
    assert live.is_blocked() is False


def test_requests_quote_encodes_cyrillic():
    assert live.requests_quote("мёд") == "%D0%BC%D1%91%D0%B4"


# --- live: разбор выдачи ---

def test_parse_search_groups_products_by_company():
    firms = live.parse_search(PRODUCT_SEARCH)
    assert len(firms) == 1
    f = firms[0]
    assert f["slug"] == "miasokombinat-sever"
    assert f["name"] == "Мясокомбинат Север"
    assert f["region_slug"] == "altaiskii-krai-155"
    assert f["found_products"] == ["Колбаса варёная", "Сосиски"]
    assert "Сосиски" in f["description"]
    assert f["city"] == "Барнаул"


def test_parse_search_adds_firm_cards_and_skips_known_and_broken():
    firms = live.parse_search(FIRM_SEARCH)
    slugs = {f["slug"] for f in firms}
    assert slugs == {"syrovarnia", "bez-goods", "miasokombinat-sever"}  # без товаров-карточек компания тоже новая
    syr = next(f for f in firms if f["slug"] == "syrovarnia")
    assert syr["products_count"] == 12
    assert syr["from_search"] is True
    bez = next(f for f in firms if f["slug"] == "bez-goods")
    assert bez["products_count"] is None
    assert bez["region_slug"] is None


def test_parse_search_empty_html():
    assert live.parse_search("<html></html>") == []


def test_parse_product_cards_extracts_links_and_prices():
    cards = live._parse_product_cards(PRODUCT_CARDS)
    urls = [c["url"] for c in cards]
    assert "https://productcenter.ru/products/5/kolbasa" in urls
    assert "https://other.ru/products/6/x" in urls
    first = cards[0]
    assert first["price"] == "от 250 ₽"
    assert first["image"] == "https://productcenter.ru/img/k.jpg"


def test_resolve_producer_url_finds_company_page(monkeypatch):
    html = '<a href="/producers/1/miasokombinat-sever">Производитель</a>'
    monkeypatch.setattr(live, "_fetch", lambda url: html)
    item = {"url": "https://productcenter.ru/products/9/kolbasa", "slug": "miasokombinat-sever"}
    assert live._resolve_producer_url(item) == "https://productcenter.ru/producers/1/miasokombinat-sever"


def test_resolve_producer_url_none_without_html_or_link(monkeypatch):
    item = {"url": "https://x/products/1", "slug": "firm"}
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live._resolve_producer_url(item) is None
    monkeypatch.setattr(live, "_fetch", lambda url: "<p>нет ссылки</p>")
    assert live._resolve_producer_url(item) is None


def test_fetch_products_full_list_and_limit(monkeypatch):
    listing = '<a href="/products/producer-1-all">Смотреть всё</a>' + PRODUCT_CARDS
    first = '<a href="/products/producer-1-all">Смотреть всё</a>'

    def fake(url):
        if url.endswith("/producers/1/x"):
            return first
        return PRODUCT_CARDS

    monkeypatch.setattr(live, "_fetch", fake)
    item = {"url": "https://productcenter.ru/producers/1/x", "slug": "x"}
    got = live.fetch_products(item, limit=1)
    assert len(got) == 1
    assert got[0]["name"] == "Колбаса"


def test_fetch_products_resolves_product_link_and_handles_failure(monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live.fetch_products({"url": "https://x/products/1", "slug": "s"}) == []
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live.fetch_products({"url": "https://productcenter.ru/producers/1/x"}) == []


def test_fetch_products_without_full_link_uses_first_page(monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: PRODUCT_CARDS)
    got = live.fetch_products({"url": "https://productcenter.ru/producers/1/x"})
    assert len(got) >= 1


def test_fetch_products_resolves_then_loads(monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        if url.endswith("/products/1/x"):
            return '<a href="/producers/4/firma">ок</a>'
        return PRODUCT_CARDS

    monkeypatch.setattr(live, "_fetch", fake)
    item = {"url": "https://productcenter.ru/products/1/x", "slug": "firma"}
    got = live.fetch_products(item)
    assert got and calls[0].endswith("/products/1/x")


# --- live: поиск с записью в БД и кэшем ---

def search_pages(url):
    if "q=" in url and "%D0%BA%D0%BE%D0%BB%D0%B1%D0%B0%D1%81%D0%B0" in url and "page=" not in url:
        return PRODUCT_SEARCH
    return "<html></html>"


def test_search_live_adds_companies_and_fills_found(con, monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: PRODUCT_SEARCH if "q=" in url else None)
    found = set()
    new = live.search_live(con, "колбаса", pages=1, found=found)
    assert new == 1
    assert len(found) == 1
    assert con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 1


def test_search_live_second_call_uses_cache(con, monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        return PRODUCT_SEARCH

    monkeypatch.setattr(live, "_fetch", fake)
    live.search_live(con, "колбаса", pages=1)
    n_calls = len(calls)
    found = set()
    assert live.search_live(con, "колбаса", pages=1, found=found) == 0
    assert len(calls) == n_calls
    assert found  # компании из кэша снова попадают в found


def test_search_live_with_city_and_head_word(con, monkeypatch):
    seen = []

    def fake(url):
        seen.append(url)
        return PRODUCT_SEARCH if "%D0%B1%D0%B0%D1%80%D0%BD%D0%B0%D1%83%D0%BB" in url else "<html></html>"

    monkeypatch.setattr(live, "_fetch", fake)
    live.search_live(con, "колбаса варёная", pages=1, city="Барнаул")
    seen = [unquote(u) for u in seen]
    assert any("колбаса варёная Барнаул" in u for u in seen)  # сначала «товар + город»
    assert any("q=колбаса" in u and "Барнаул" not in u for u in seen)  # потом главное слово


def test_search_live_skips_non_food_and_stops_on_empty(con, monkeypatch):
    monkeypatch.setattr(live.ranking, "not_food", lambda text: True)
    monkeypatch.setattr(live, "_fetch", lambda url: PRODUCT_SEARCH if "page=" not in url else None)
    assert live.search_live(con, "колбаса", pages=3) == 0


def test_search_live_empty_query(con, monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live.search_live(con, "", pages=1) == 0


def test_search_live_broad_query_does_not_count_as_hit(con, monkeypatch):
    # по фразе сайт ничего не отдал, по главному слову — компания; она не считается попаданием по фразе
    monkeypatch.setattr(live, "_fetch",
                        lambda url: PRODUCT_SEARCH if unquote(url).endswith("q=колбаса") else None)
    found = set()
    assert live.search_live(con, "колбаса копчёная", pages=1, found=found) == 1
    assert found == set()


def test_search_live_stops_paging_when_no_items(con, monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live.search_live(con, "молоко", pages=2) == 0


# --- live: контакты ---

def test_enrich_contacts_saves_phones_and_emails(con, monkeypatch):
    db.upsert(con, {"id": 1, "name": "A", "url": "https://productcenter.ru/producers/1/a",
                    "categories": [], "phones": []})
    db.upsert(con, {"id": 2, "name": "B", "url": "https://productcenter.ru/producers/2/b",
                    "categories": [], "phones": []})
    items = [dict(r) for r in con.execute("SELECT * FROM suppliers").fetchall()]
    for it in items:
        it["contacts_at"] = None
    html = "Тел. +7 843 000-00-01 почта sales@example.ru"

    monkeypatch.setattr(live, "_fetch", lambda url: html)
    assert live.enrich_contacts(con, items) == 2
    row = con.execute("SELECT phones, emails, contacts_at FROM suppliers WHERE id=1").fetchone()
    assert json.loads(row["phones"]) == ["+7 843 000-00-01"]
    assert row["contacts_at"] is not None


def test_enrich_contacts_skips_fresh_and_empty(con, monkeypatch):
    import time as t
    fresh = {"id": 5, "contacts_at": t.time(), "url": "https://x/producers/5/y", "slug": "y"}
    monkeypatch.setattr(live, "_fetch", lambda url: pytest.fail("свежие контакты не нужны"))
    assert live.enrich_contacts(con, [fresh]) == 0
    assert live.enrich_contacts(con, []) == 0


def test_enrich_contacts_failed_page_not_counted(con, monkeypatch):
    item = {"id": 7, "url": "https://productcenter.ru/producers/7/z", "slug": "z", "contacts_at": None}
    db.upsert(con, {"id": 7, "name": "Z", "url": item["url"], "categories": [], "phones": []})
    monkeypatch.setattr(live, "_fetch", lambda url: None)
    assert live.enrich_contacts(con, [item]) == 0


def test_enrich_contacts_resolves_product_items(con, monkeypatch):
    item = {"id": 8, "url": "https://productcenter.ru/products/8/q", "slug": "q", "contacts_at": None}
    db.upsert(con, {"id": 8, "name": "Q", "url": item["url"], "categories": [], "phones": []})
    monkeypatch.setattr(live, "_fetch", lambda url: '<a href="/producers/8/q">ок</a> +7 900 111-22-33')
    assert live.enrich_contacts(con, [item]) == 1


def test_fetch_one_handles_parse_error(monkeypatch):
    monkeypatch.setattr(live, "_fetch", lambda url: "<html>")
    monkeypatch.setattr(live.pc, "parse_details", lambda html: (_ for _ in ()).throw(ValueError("битая")))
    assert live._fetch_one({"id": 3, "url": "https://x/producers/3/a"}) == (3, None, None)


# --- productcenter_parser ---

LISTING = """
<div class="card_item firm">
  <div class="text"><a class="link" href="/producers/10/molzavod">Молочный завод</a></div>
  <div class="item_descriptor">Сыры и молоко</div>
  <div class="ii_city">Казань</div>
  <div class="ii_category" title="Категория: Молочные продукты, сыры">
    <a href="/r-tatarstan-16/x">Кат</a></div>
  <div class="ii_goods"><a>Товаров 42</a></div>
</div>
<div class="card_item firm"><span>без ссылки</span></div>
"""

DETAILS = """
<html><body>Звоните: +7 (843) 555-01-02 или 8 900 123-45-67,
почта: info@molzavod.ru, sales@molzavod.ru
Голый номер 89001234567 не считаем.</body></html>
"""


def test_pc_clean_collapses_whitespace():
    assert pc.clean("  а \n\n  б ") == "а б"


def test_pc_parse_listing_fields():
    items = pc.parse_listing(LISTING, "molochnyie-produkty-316")
    assert len(items) == 1
    it = items[0]
    assert it["id"] == 10
    assert it["name"] == "Молочный завод"
    assert it["subcategory"] == "Молочные продукты"
    assert it["region_slug"] == "tatarstan-16"
    assert it["products_count"] == 42
    assert it["city"] == "Казань"


def test_pc_parse_listing_without_optional_parts():
    html = '<div class="card_item firm"><div class="text"><a class="link" href="/producers/11/x">X</a></div></div>'
    it = pc.parse_listing(html, "c")[0]
    assert it["id"] == 11
    assert it["region_slug"] is None
    assert it["subcategory"] is None
    assert it["products_count"] is None


def test_pc_parse_details_phones_and_emails():
    d = pc.parse_details(DETAILS)
    assert "+7 (843) 555-01-02" in d["phones"] or any("555" in p for p in d["phones"])
    assert len(d["phones"]) <= 3
    assert d["emails"] == ["info@molzavod.ru", "sales@molzavod.ru"]
    assert "89001234567" not in d["phones"]  # голые цифры без разделителей не считаем


def test_pc_fetch_retries_and_returns_text(monkeypatch):
    monkeypatch.setattr(pc.session, "get", fake_get(Resp(200, "<p>да</p>")))
    assert pc.fetch("https://x") == "<p>да</p>"


def test_pc_fetch_gives_up_after_retries(monkeypatch):
    import requests
    monkeypatch.setattr(pc.session, "get", fake_get(requests.RequestException("сеть")))
    assert pc.fetch("https://x", retries=2) is None


def test_pc_fetch_rate_limit_status(monkeypatch):
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(pc.session, "get", fake_get(Resp(429)))
    assert pc.fetch("https://x", retries=1) is None
    assert 10 in slept


def test_pc_scrape_category_pages_and_details(monkeypatch):
    pages = {
        "https://productcenter.ru/producers/catalog-miaso-203": LISTING,
        "https://productcenter.ru/producers/catalog-miaso-203/page-2": "<html></html>",
    }
    monkeypatch.setattr(pc, "fetch", lambda url: pages.get(url))
    pages["https://productcenter.ru/producers/10/molzavod"] = DETAILS
    res = pc.scrape_category("miaso-203", limit=5, details=True)
    assert len(res) == 1
    assert res[0]["emails"] == ["info@molzavod.ru", "sales@molzavod.ru"]


def test_pc_scrape_category_failed_first_page(monkeypatch):
    monkeypatch.setattr(pc, "fetch", lambda url: None)
    assert pc.scrape_category("x", limit=3, details=False) == []


def test_pc_scrape_category_details_missing_page(monkeypatch):
    monkeypatch.setattr(pc, "fetch", lambda url: LISTING if "page-" not in url and "catalog" in url and
                        url.endswith("catalog-x") else (None if "page-" not in url else None))
    res = pc.scrape_category("x", limit=3, details=True)
    assert len(res) == 1
    assert "phones" not in res[0]


def test_pc_main_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "scrape_category",
                        lambda slug, limit, details: [
                            {"id": 10, "category": slug, "name": "A", "url": "u"},
                        ] if slug == "a" else [
                            {"id": 10, "category": slug, "name": "A", "url": "u"},
                            {"id": 20, "category": slug, "name": "B", "url": "v"},
                        ])
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["pc", "-c", "a", "b", "-n", "2", "-o", str(out)])
    pc.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["count"] == 2
    ten = next(i for i in data["items"] if i["id"] == 10)
    assert sorted(ten["categories"]) == ["a", "b"]


# --- agroserver_parser ---

AGRO_LINE = """
<div class="line" id="p1">
  <div class="th"><a href="/offer/1/muka">Мука пшеничная</a></div>
  <div class="price">от 3 700 руб / кг</div>
  <div class="geo">Казань</div>
  <div class="text">Высший сорт</div>
  <a class="personal_org_menu">ООО Мука</a>
  <a href="/user/77/" >Профиль</a>
  <a class="href_to_tel">+7 843 111-22-33</a>
  <span class="verify"></span>
  <div class="date">вчера</div>
</div>
<div class="line" id="p2"><div class="th"><span>без ссылки</span></div></div>
"""


def test_agro_parse_price_variants():
    p = agro.parse_price("от 3 700 руб /  кг")
    assert p["price_value"] == 3700.0
    assert p["price_unit"] == "кг"
    assert agro.parse_price("договорная")["price_value"] is None
    assert agro.parse_price("1,5 руб / шт")["price_value"] == 1.5


def test_agro_parse_listing_fields():
    items = agro.parse_listing(AGRO_LINE, "bakaleya")
    assert len(items) == 1
    it = items[0]
    assert it["title"] == "Мука пшеничная"
    assert it["url"] == "https://agroserver.ru/offer/1/muka"
    assert it["city"] == "Казань"
    assert it["company"] == "ООО Мука"
    assert it["company_url"] == "https://agroserver.ru/user/77/"
    assert it["phones"] == ["+7 843 111-22-33"]
    assert it["verified"] is True
    assert it["posted"] == "вчера"


def test_agro_next_page_url():
    html = '<a href="/bakaleya/2/">2</a>'
    assert agro.next_page_url(html, "https://agroserver.ru/bakaleya/1/") == "https://agroserver.ru/bakaleya/2/"
    assert agro.next_page_url("<p></p>", "https://agroserver.ru/bakaleya/") is None


def test_agro_fetch_retries(monkeypatch):
    import requests
    monkeypatch.setattr(agro.session, "get", fake_get(Resp(200, "<p>ок</p>")))
    assert agro.fetch("https://x") == "<p>ок</p>"
    monkeypatch.setattr(agro.session, "get", fake_get(requests.RequestException("нет")))
    assert agro.fetch("https://x", retries=2) is None
    monkeypatch.setattr(agro.session, "get", fake_get(Resp(500)))
    assert agro.fetch("https://x", retries=1) is None


def test_agro_scrape_category_follows_pages(monkeypatch):
    pages = {
        "https://agroserver.ru/bakaleya/": AGRO_LINE + '<a href="/bakaleya/2/">2</a>',
        "https://agroserver.ru/bakaleya/2/": AGRO_LINE.replace("p1", "p3").replace("/offer/1/muka", "/offer/9/x"),
    }
    monkeypatch.setattr(agro, "fetch", lambda url: pages.get(url))
    res = agro.scrape_category("bakaleya", limit=10)
    assert len(res) == 2
    assert agro.scrape_category("bakaleya", limit=1)[0]["title"] == "Мука пшеничная"


def test_agro_scrape_category_stops_without_html(monkeypatch):
    monkeypatch.setattr(agro, "fetch", lambda url: None)
    assert agro.scrape_category("x", limit=5) == []


def test_agro_main_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(agro, "scrape_category", lambda slug, limit: [{"title": slug}])
    out = tmp_path / "agro.json"
    monkeypatch.setattr(sys, "argv", ["agro", "-c", "a", "b", "-o", str(out)])
    agro.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["count"] == 2

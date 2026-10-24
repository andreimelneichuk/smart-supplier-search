"""SQLite-хранилище (db.py) на временной базе."""
import json

import pytest

import db


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    c = db.connect()
    yield c
    c.close()


def card(**kw):
    base = {"id": 1, "name": "Мясокомбинат Север", "url": "https://productcenter.ru/producers/1/sever",
            "subcategory": "Колбасы", "city": "Казань", "description": "Колбасы и сосиски",
            "products_count": 12, "categories": ["miaso-203"], "phones": ["+7 843 000-00-01"],
            "emails": ["sever@example.ru"]}
    base.update(kw)
    return base


def test_slug_of_takes_last_segment():
    assert db.slug_of("https://productcenter.ru/producers/40417/italian-pizza/") == "italian-pizza"


def test_connect_creates_schema_and_tables(con):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"suppliers", "regions", "supplier_extra", "supplier_notes",
            "supplier_note_history", "supplier_passports"} <= names


def test_connect_migrates_old_database(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    import sqlite3
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE suppliers (id INTEGER PRIMARY KEY, name TEXT, url TEXT, subcategory TEXT, city TEXT, "
                "description TEXT, products_count INTEGER, categories TEXT, phones TEXT, emails TEXT, "
                "contacts_at REAL, added_at REAL)")
    old.execute("INSERT INTO suppliers (id, name, url, categories, phones, emails, added_at) "
                "VALUES (1, 'Старый', 'https://x.ru/producers/9/old-firm', '[]', '[]', '[]', 0)")
    old.commit()
    old.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    c = db.connect()
    row = c.execute("SELECT slug FROM suppliers WHERE id=1").fetchone()
    assert row["slug"] == "old-firm"
    cols = {r["name"] for r in c.execute("PRAGMA table_info(suppliers)")}
    assert {"region_slug", "region"} <= cols
    c.close()


def test_upsert_new_card_returns_true_and_stores_fields(con):
    assert db.upsert(con, card()) is True
    row = con.execute("SELECT * FROM suppliers").fetchone()
    assert row["slug"] == "sever"
    assert json.loads(row["phones"]) == ["+7 843 000-00-01"]
    assert row["contacts_at"] is not None  # телефоны уже были в карточке


def test_upsert_new_card_without_phones_has_no_contacts_time(con):
    item = card()
    del item["phones"]
    item.pop("emails")
    db.upsert(con, item)
    assert con.execute("SELECT contacts_at FROM suppliers").fetchone()[0] is None


def test_upsert_uses_category_singular_when_no_list(con):
    item = card(categories=None, category="bakalieia-208")
    db.upsert(con, item)
    assert json.loads(con.execute("SELECT categories FROM suppliers").fetchone()[0]) == ["bakalieia-208"]


def test_upsert_existing_company_merges_categories_and_updates_card(con):
    db.upsert(con, card())
    db.upsert(con, card(id=2, categories=["konsiervy-615"], name="Север (новое название)", description="Новое"))
    rows = con.execute("SELECT name, categories FROM suppliers").fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "Север (новое название)"
    assert json.loads(rows[0]["categories"]) == ["konsiervy-615", "miaso-203"]


def test_upsert_from_search_keeps_card_and_prepends_found_products(con):
    db.upsert(con, card())
    res = db.upsert(con, card(id=2, from_search=True, categories=["ryba"], description="Колбасы и сосиски",
                              found_products=["Сосиски Докторские", "Колбасы"]))
    assert res is False
    desc = con.execute("SELECT name, description FROM suppliers").fetchone()
    assert desc["name"] == "Мясокомбинат Север"  # карточку из поиска не перезаписываем
    assert desc["description"].startswith("Товары: Сосиски Докторские. ")  # «Колбасы» уже есть в описании


def test_upsert_from_search_without_new_products_keeps_description(con):
    db.upsert(con, card(description="Сосиски"))
    db.upsert(con, card(id=2, product_url="https://x/p/2", found_products=["сосиски"], description="Сосиски"))
    assert con.execute("SELECT description FROM suppliers").fetchone()[0] == "Сосиски"


def test_search_hit_uses_product_url_and_negative_id(con):
    db.upsert(con, card(id=77, product_url="https://productcenter.ru/products/5/sosiski", slug="sosiski-firma"))
    row = con.execute("SELECT id, url FROM suppliers").fetchone()
    assert row["id"] == -77
    assert row["url"] == "https://productcenter.ru/products/5/sosiski"


def test_region_slug_is_applied_from_regions_table(con):
    con.execute("INSERT INTO regions (slug, name) VALUES ('altaiskii-krai-155', 'Алтайский край')")
    db.upsert(con, card(region_slug="altaiskii-krai-155"))
    row = con.execute("SELECT region_slug, region FROM suppliers").fetchone()
    assert row["region_slug"] == "altaiskii-krai-155"
    assert row["region"] == "Алтайский край"


def test_existing_company_gets_region_on_repeat_upsert(con):
    con.execute("INSERT INTO regions (slug, name) VALUES ('tula-70', 'Тульская область')")
    db.upsert(con, card())
    db.upsert(con, card(id=2, region_slug="tula-70"))
    assert con.execute("SELECT region FROM suppliers").fetchone()[0] == "Тульская область"


def test_set_url_and_save_contacts(con):
    db.upsert(con, card())
    sid = con.execute("SELECT id FROM suppliers").fetchone()[0]
    db.set_url(con, sid, "https://new.example/x")
    db.save_contacts(con, sid, ["+7 900 000 00 00"], [])
    row = con.execute("SELECT url, phones, emails, contacts_at FROM suppliers").fetchone()
    assert row["url"] == "https://new.example/x"
    assert json.loads(row["phones"]) == ["+7 900 000 00 00"]
    assert json.loads(row["emails"]) == []
    assert row["contacts_at"] is not None


def test_save_note_first_time_writes_history(con):
    db.upsert(con, card())
    sid = 1
    db.save_note(con, sid, "Ждём прайс", "Попросили КП", author="Анна")
    hist = db.get_note_history(con, sid)
    assert len(hist) == 1
    assert hist[0]["author"] == "Анна"
    assert hist[0]["text"] == "Попросили КП"
    assert hist[0]["status"] == "Ждём прайс"


def test_save_note_same_text_and_status_adds_no_history(con):
    db.upsert(con, card())
    db.save_note(con, 1, "Ждём прайс", "Текст")
    db.save_note(con, 1, "Ждём прайс", "Текст")
    assert len(db.get_note_history(con, 1)) == 1


def test_save_note_status_only_change_adds_history_without_text(con):
    db.upsert(con, card())
    db.save_note(con, 1, "Ждём прайс", "Текст")
    db.save_note(con, 1, "Договорились", "Текст")
    hist = db.get_note_history(con, 1)
    assert len(hist) == 2
    assert hist[0]["status"] == "Договорились"
    assert hist[0]["text"] == ""


def test_notes_are_separated_by_organization(con):
    db.upsert(con, card())
    db.save_note(con, 1, "Отказ", "Нет", org_name="Компания А")
    assert db.get_note_history(con, 1, org_name="Компания Б") == []
    assert len(db.get_note_history(con, 1, org_name="Компания А")) == 1


def test_passport_roundtrip_and_bad_json(con):
    db.upsert(con, card())
    assert db.get_passport(con, 1) is None
    db.save_passport(con, 1, {"logistics": {"x": "Казань"}}, "полный текст")
    got = db.get_passport(con, 1)
    assert got["passport"] == {"logistics": {"x": "Казань"}}
    assert got["full_text"] == "полный текст"
    con.execute("UPDATE supplier_passports SET passport_json='{битый'")
    assert db.get_passport(con, 1) is None


def test_clean_str_handles_all_types():
    assert db._clean_str(None) is None
    assert db._clean_str(["a", "b"]) == "a, b"
    assert db._clean_str({"value": "10 шт"}) == "10 шт"
    assert db._clean_str({"x": 1}) == '{"x": 1}'
    assert db._clean_str(5) == "5"


def test_save_extra_inserts_then_updates(con):
    db.upsert(con, card())
    db.save_extra(con, 1, {"min_order": ["100 кг"], "price_range": {"value": "от 300 ₽"}})
    db.save_extra(con, 1, {"min_order": "200 кг"})
    row = con.execute("SELECT min_order, price_range FROM supplier_extra").fetchone()
    assert row["min_order"] == "200 кг"
    assert row["price_range"] is None


def test_all_suppliers_joins_enriched_extra_and_notes(con):
    db.upsert(con, card())
    con.execute("""CREATE TABLE enriched_catalog (supplier_id INTEGER, is_food INTEGER, primary_products TEXT,
        specific_varieties TEXT, processing TEXT, dish_ingredients TEXT, search_keywords TEXT)""")
    con.execute("INSERT INTO enriched_catalog VALUES (1, 1, '[\"колбасы\"]', '[]', '[]', '[]', 'колбаса')")
    db.save_extra(con, 1, {"min_order": "100 кг"})
    db.save_note(con, 1, "Отказ", "Нет")
    row = db.all_suppliers(con)[0]
    assert row["primary_products"] == ["колбасы"]
    assert row["min_order"] == "100 кг"
    assert row["user_status"] == "Отказ"
    assert row["categories"] == ["miaso-203"]
    assert row["phones"] == ["+7 843 000-00-01"]


def test_all_suppliers_without_optional_tables_and_defaults(con):
    con.execute("DROP TABLE supplier_notes")
    db.upsert(con, card(categories=[], phones=[], emails=[]))
    con.execute("UPDATE suppliers SET categories='', phones='not-json'")
    row = db.all_suppliers(con)[0]
    assert row["categories"] == []
    assert row["phones"] == []
    assert row["user_status"] == "Не связывались"
    assert row["user_note"] == ""


def test_all_suppliers_bad_enriched_json_falls_back_to_empty(con):
    db.upsert(con, card())
    con.execute("""CREATE TABLE enriched_catalog (supplier_id INTEGER, is_food INTEGER, primary_products TEXT,
        specific_varieties TEXT, processing TEXT, dish_ingredients TEXT, search_keywords TEXT)""")
    con.execute("INSERT INTO enriched_catalog VALUES (1, 1, '{плохо', '[]', '[]', '[]', '')")
    assert db.all_suppliers(con)[0]["primary_products"] == []


def test_import_json_and_init_from_json_if_empty(con, tmp_path, monkeypatch):
    data = {"items": [card(id=1), card(id=2, name="Другой", url="https://productcenter.ru/producers/2/drugoi",
                                       phones=[], emails=[])]}
    path = tmp_path / "suppliers.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert db.import_json(con, path) == 2
    # база уже не пустая — init ничего не делает
    monkeypatch.setattr(db, "HERE", tmp_path)
    db.init_from_json_if_empty(con)
    assert con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 2


def test_init_from_json_loads_when_empty(con, tmp_path, monkeypatch):
    data = {"items": [card()]}
    (tmp_path / "suppliers.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(db, "HERE", tmp_path)
    db.init_from_json_if_empty(con)
    assert con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 1


def test_init_from_json_without_file_does_nothing(con, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "HERE", tmp_path)
    db.init_from_json_if_empty(con)
    assert con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 0


def test_region_name_from_cache_and_from_site(con):
    calls = []

    def fetch(url):
        calls.append(url)
        return "<h1 class='x'>Производители Алтайский край</h1>"

    db.upsert(con, card(region_slug="altaiskii-krai-155"))
    assert db.region_name(con, "altaiskii-krai-155", fetch) == "Алтайский край"
    assert db.region_name(con, "altaiskii-krai-155", fetch) == "Алтайский край"
    assert len(calls) == 1  # второй раз берём из кэша
    assert con.execute("SELECT region FROM suppliers").fetchone()[0] == "Алтайский край"


def test_region_name_none_when_site_fails_or_has_no_heading(con):
    assert db.region_name(con, "x-1", lambda url: None) is None
    assert db.region_name(con, "x-1", lambda url: "<p>нет заголовка</p>") is None


def test_resolve_regions_fills_missing_names(con):
    db.upsert(con, card(region_slug="tula-70"))
    db.upsert(con, card(id=2, name="Б", url="https://productcenter.ru/producers/2/b", region_slug="ryazan-62"))
    n = db.resolve_regions(
        con, lambda url: "<h1>Производители " + ("Тульская область" if "tula" in url else "Рязанская область") + "</h1>")
    assert n == 2
    assert db.resolve_regions(con, lambda url: None) == 0

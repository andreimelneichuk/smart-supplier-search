"""SQLite-хранилище поставщиков."""
import json
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "suppliers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    subcategory TEXT,
    city TEXT,
    description TEXT,
    products_count INTEGER,
    categories TEXT NOT NULL DEFAULT '[]',   -- JSON-список слагов
    phones TEXT NOT NULL DEFAULT '[]',
    emails TEXT NOT NULL DEFAULT '[]',
    contacts_at REAL,                        -- когда подтянули контакты (unix)
    slug TEXT,                               -- слаг компании из URL: ключ для склейки дублей
    region_slug TEXT,                        -- слаг региона из ссылок сайта (altaiskii-krai-155)
    region TEXT,                             -- русское название региона (для геокодирования)
    added_at REAL NOT NULL
);
"""


def slug_of(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=60.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=60000;")
    con.executescript(SCHEMA)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(suppliers)")}
    if "slug" not in cols:  # миграция старой БД
        con.execute("ALTER TABLE suppliers ADD COLUMN slug TEXT")
        for r in con.execute("SELECT id, url FROM suppliers").fetchall():
            con.execute("UPDATE suppliers SET slug=? WHERE id=?", (slug_of(r["url"]), r["id"]))
        con.commit()
    for col in ("region_slug", "region"):
        if col not in cols:
            con.execute(f"ALTER TABLE suppliers ADD COLUMN {col} TEXT")
    con.execute("CREATE TABLE IF NOT EXISTS regions (slug TEXT PRIMARY KEY, name TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_slug ON suppliers(slug)")
    con.execute("""
    CREATE TABLE IF NOT EXISTS supplier_extra (
        supplier_id INTEGER PRIMARY KEY,
        min_order TEXT,
        price_range TEXT,
        certificates TEXT,
        delivery_terms TEXT,
        region_scope TEXT,
        enriched_at REAL NOT NULL
    )""")
    cols_notes = {r["name"] for r in con.execute("PRAGMA table_info(supplier_notes)")}
    if "org_name" not in cols_notes:
        con.execute("DROP TABLE IF EXISTS supplier_notes")
        con.execute("""
        CREATE TABLE supplier_notes (
            supplier_id INTEGER,
            org_name TEXT NOT NULL DEFAULT 'ООО «ФудСервис»',
            status TEXT NOT NULL DEFAULT 'Не связывались',
            note TEXT NOT NULL DEFAULT '',
            updated_by TEXT NOT NULL DEFAULT 'Менеджер',
            updated_at REAL NOT NULL,
            PRIMARY KEY (supplier_id, org_name)
        )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS supplier_note_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id INTEGER NOT NULL,
        org_name TEXT NOT NULL DEFAULT 'ООО «ФудСервис»',
        author TEXT NOT NULL,
        status TEXT,
        text TEXT NOT NULL,
        created_at REAL NOT NULL
    )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS supplier_passports (
        supplier_id INTEGER PRIMARY KEY,
        passport_json TEXT NOT NULL,
        full_text TEXT,
        created_at REAL NOT NULL
    )""")
    return con


def upsert(con: sqlite3.Connection, item: dict) -> bool:
    """Добавляет карточку или дополняет существующую (ключ — слаг компании).
    Карточки из поиска (есть product_url) уже известную компанию не перезаписывают.
    Возвращает True, если запись новая."""
    slug = item.get("slug") or slug_of(item["url"])
    cats = item.get("categories") or ([item["category"]] if item.get("category") else [])
    row = con.execute("SELECT id, categories FROM suppliers WHERE slug=?", (slug,)).fetchone()
    if row:
        merged = sorted(set(json.loads(row["categories"])) | set(cats))
        if item.get("region_slug"):  # регион дописываем всегда, если он известен
            con.execute("UPDATE suppliers SET region_slug=?, region=COALESCE((SELECT name FROM regions WHERE slug=?), region) "
                        "WHERE id=?", (item["region_slug"], item["region_slug"], row["id"]))
        if "product_url" in item or item.get("from_search"):
            con.execute("UPDATE suppliers SET categories=? WHERE id=?", (json.dumps(merged), row["id"]))
            # товары, по которым сайт нашёл компанию, добавляем в описание: иначе наш поиск о них не знает
            desc = con.execute("SELECT description FROM suppliers WHERE id=?", (row["id"],)).fetchone()[0] or ""
            extra = [p for p in item.get("found_products", []) if p.lower() not in desc.lower()]
            if extra:  # в начало: и карточка, и переранжирование читают первые ~200 символов
                con.execute("UPDATE suppliers SET description=? WHERE id=?",
                            ("Товары: " + "; ".join(extra) + ". " + desc, row["id"]))
        else:
            con.execute(
                "UPDATE suppliers SET name=?, url=?, subcategory=?, city=?, description=?, "
                "products_count=?, categories=? WHERE id=?",
                (item["name"], item["url"], item.get("subcategory"), item.get("city"),
                 item.get("description"), item.get("products_count"), json.dumps(merged), row["id"]),
            )
        return False
    # для найденных поиском ссылка на компанию битая (404): пока храним ссылку на товар
    url = item.get("product_url") or item["url"]
    # id из поиска — это id товара и может совпасть с id чужой компании, поэтому он отрицательный
    sid = -abs(item["id"]) if "product_url" in item else item["id"]
    con.execute(
        "INSERT INTO suppliers (id,name,url,subcategory,city,description,products_count,"
        "categories,phones,emails,contacts_at,added_at,slug,region_slug,region) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,(SELECT name FROM regions WHERE slug=?))",
        (sid, item["name"], url, item.get("subcategory"), item.get("city"),
         item.get("description"), item.get("products_count"), json.dumps(cats),
         json.dumps(item.get("phones", [])), json.dumps(item.get("emails", [])),
         time.time() if "phones" in item else None, time.time(), slug,
         item.get("region_slug"), item.get("region_slug")),
    )
    return True


def set_url(con: sqlite3.Connection, supplier_id: int, url: str):
    con.execute("UPDATE suppliers SET url=? WHERE id=?", (url, supplier_id))
    con.commit()


def save_contacts(con: sqlite3.Connection, supplier_id: int, phones: list, emails: list):
    con.execute("UPDATE suppliers SET phones=?, emails=?, contacts_at=? WHERE id=?",
                (json.dumps(phones), json.dumps(emails), time.time(), supplier_id))
    con.commit()


def save_note(con: sqlite3.Connection, supplier_id: int, status: str, note: str,
              org_name: str = "ООО «ФудСервис»", author: str = "Менеджер"):
    now = time.time()
    prev = con.execute("SELECT status, note FROM supplier_notes WHERE supplier_id=? AND org_name=?",
                       (supplier_id, org_name)).fetchone()
    con.execute("""
    INSERT INTO supplier_notes (supplier_id, org_name, status, note, updated_by, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(supplier_id, org_name) DO UPDATE SET
        status = excluded.status,
        note = excluded.note,
        updated_by = excluded.updated_by,
        updated_at = excluded.updated_at
    """, (supplier_id, org_name, status, note, author, now))
    # в историю — только то, что изменилось: прежняя заметка при смене статуса не дублируется
    new_text = note.strip() if not prev or note.strip() != (prev["note"] or "").strip() else ""
    status_changed = (prev["status"] if prev else "Не связывались") != status
    if new_text or status_changed:
        con.execute("""
        INSERT INTO supplier_note_history (supplier_id, org_name, author, status, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (supplier_id, org_name, author, status, new_text, now))
    con.commit()


def get_note_history(con: sqlite3.Connection, supplier_id: int, org_name: str = "ООО «ФудСервис»") -> list[dict]:
    rows = con.execute("""
    SELECT author, status, text, created_at
    FROM supplier_note_history
    WHERE supplier_id = ? AND org_name = ?
    ORDER BY created_at DESC
    """, (supplier_id, org_name)).fetchall()
    return [dict(r) for r in rows]


def get_passport(con: sqlite3.Connection, supplier_id: int) -> dict | None:
    row = con.execute("SELECT passport_json, full_text FROM supplier_passports WHERE supplier_id=?", (supplier_id,)).fetchone()
    if row:
        try:
            return {"passport": json.loads(row["passport_json"]), "full_text": row["full_text"]}
        except json.JSONDecodeError:
            return None
    return None


def save_passport(con: sqlite3.Connection, supplier_id: int, passport: dict, full_text: str):
    con.execute("""
    INSERT OR REPLACE INTO supplier_passports (supplier_id, passport_json, full_text, created_at)
    VALUES (?, ?, ?, ?)
    """, (supplier_id, json.dumps(passport, ensure_ascii=False), full_text, time.time()))
    con.commit()



def _clean_str(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val)
    if isinstance(val, dict):
        if "value" in val:
            return str(val["value"])
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def save_extra(con: sqlite3.Connection, supplier_id: int, data: dict):
    con.execute("""
    INSERT INTO supplier_extra (supplier_id, min_order, price_range, certificates, delivery_terms, region_scope, enriched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(supplier_id) DO UPDATE SET
        min_order = excluded.min_order,
        price_range = excluded.price_range,
        certificates = excluded.certificates,
        delivery_terms = excluded.delivery_terms,
        region_scope = excluded.region_scope,
        enriched_at = excluded.enriched_at
    """, (supplier_id,
          _clean_str(data.get("min_order")),
          _clean_str(data.get("price_range")),
          _clean_str(data.get("certificates")),
          _clean_str(data.get("delivery_terms")),
          _clean_str(data.get("region_scope")),
          time.time()))
    con.commit()


def all_suppliers(con: sqlite3.Connection, org_name: str = "ООО «ФудСервис»") -> list[dict]:
    has_enriched = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='enriched_catalog'").fetchone() is not None
    has_extra = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='supplier_extra'").fetchone() is not None
    has_notes = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='supplier_notes'").fetchone() is not None

    select_fields = ["s.*"]
    joins = []
    params = []

    if has_enriched:
        select_fields.append("e.is_food, e.primary_products, e.specific_varieties, e.processing, e.dish_ingredients, e.search_keywords")
        joins.append("LEFT JOIN enriched_catalog e ON s.id = e.supplier_id")

    if has_extra:
        select_fields.append("x.min_order, x.price_range, x.certificates, x.delivery_terms, x.region_scope")
        joins.append("LEFT JOIN supplier_extra x ON s.id = x.supplier_id")

    if has_notes:
        select_fields.append("n.status AS user_status, n.note AS user_note, n.updated_by AS user_note_author, n.updated_at AS user_note_time")
        joins.append("LEFT JOIN supplier_notes n ON s.id = n.supplier_id AND n.org_name = ?")
        params.append(org_name)

    sql = f"SELECT {', '.join(select_fields)} FROM suppliers s {' '.join(joins)}"

    out = []
    for r in con.execute(sql, params):
        d = dict(r)
        for k in ("categories", "phones", "emails"):
            if k in d and d[k]:
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    d[k] = []
            else:
                d[k] = []
        if has_enriched:
            for ek in ("primary_products", "specific_varieties", "processing", "dish_ingredients"):
                if ek in d and d[ek]:
                    try:
                        d[ek] = json.loads(d[ek])
                    except json.JSONDecodeError:
                        d[ek] = []
        if "user_status" not in d or not d["user_status"]:
            d["user_status"] = "Не связывались"
        if "user_note" not in d or not d["user_note"]:
            d["user_note"] = ""
        out.append(d)
    return out


def import_json(con: sqlite3.Connection, path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    n = sum(upsert(con, i) for i in data["items"])
    con.commit()
    return n


def init_from_json_if_empty(con: sqlite3.Connection):
    if con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 0:
        f = HERE / "suppliers.json"
        if f.exists():
            import_json(con, f)


def region_name(con: sqlite3.Connection, slug: str, fetch) -> str | None:
    """Русское название региона по слагу: кэш в таблице regions, иначе страница региона на сайте.
    fetch(url) -> html | None."""
    import re
    row = con.execute("SELECT name FROM regions WHERE slug=?", (slug,)).fetchone()
    if row:
        return row["name"]
    html = fetch(f"https://productcenter.ru/producers/r-{slug}")
    if not html:
        return None
    m = re.search(r'<h1[^>]*>\s*Производители\s+([^<]+)', html)
    if not m:
        return None
    name = m.group(1).strip()
    con.execute("INSERT OR REPLACE INTO regions (slug, name) VALUES (?,?)", (slug, name))
    con.execute("UPDATE suppliers SET region=? WHERE region_slug=?", (name, slug))
    con.commit()
    return name


def resolve_regions(con: sqlite3.Connection, fetch) -> int:
    """Достраивает названия для всех слагов регионов, которых ещё нет в кэше."""
    slugs = [r[0] for r in con.execute(
        "SELECT DISTINCT region_slug FROM suppliers WHERE region_slug IS NOT NULL AND region IS NULL")]
    return sum(region_name(con, s, fetch) is not None for s in slugs)

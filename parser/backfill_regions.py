"""Дозаполняет регионы у уже собранных компаний и геокодирует пары «город + регион».

    python backfill_regions.py
Шаги: 1) перепроход по категориям из CATEGORIES (регион берётся из ссылок сайта);
      2) русские названия регионов со страниц регионов; 3) геокодирование недостающих городов.
"""
import time

import db
import geo
import live
import productcenter_parser as pc

DELAY = 1.0


def main():
    con = db.connect()

    print("1/3 регионы из категорий…")
    for slug in pc.CATEGORIES:
        html = live._fetch(f"{pc.BASE}/producers/catalog-{slug}")
        if not html:
            print(f"  ! нет ответа: {slug}")
            continue
        for item in pc.parse_listing(html, slug):
            db.upsert(con, item)
        con.commit()
        time.sleep(DELAY)

    print("2/3 названия регионов…")
    slugs = [r[0] for r in con.execute(
        "SELECT DISTINCT region_slug FROM suppliers WHERE region_slug IS NOT NULL AND region IS NULL")]
    for s in slugs:
        name = db.region_name(con, s, live._fetch)
        print(f"  {s} -> {name}")
        time.sleep(DELAY)

    print("3/3 геокодирование…")
    geo.warm(con)

    total, with_region = con.execute(
        "SELECT COUNT(*), COUNT(region) FROM suppliers").fetchone()
    print(f"Готово: регион есть у {with_region} из {total} компаний")


if __name__ == "__main__":
    main()

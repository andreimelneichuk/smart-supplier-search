"""Парсер каталога производителей/поставщиков с productcenter.ru.

Использование:
    python productcenter_parser.py                                   # категории по умолчанию
    python productcenter_parser.py -c bakalieia-208 -n 30 -o suppliers.json
    python productcenter_parser.py --details                         # + страницы производителей

Слаг категории — хвост URL вида /producers/catalog-<слаг>.
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

BASE = "https://productcenter.ru"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
}
CATEGORIES = {  # слаг -> название (слаги берутся из URL /producers/catalog-<слаг>)
    "produkty-pitaniia-45": "Продукты питания",
    "bakalieia-208": "Бакалея",
    "konditierskiie-izdieliia-427": "Кондитерские изделия",
    "konsiervy-615": "Консервы",
    "molochnyie-produkty-316": "Молочные продукты",
    "miaso-203": "Мясо",
    "napitki-201": "Напитки",
    "ovoshchi-773": "Овощи",
    "oriekhi-799": "Орехи",
    "ryba-morieprodukty-317": "Рыба, морепродукты",
    "frukty-204": "Фрукты",
    "khliebobulochnyie-izdieliia-297": "Хлебобулочные изделия",
    "iaitso-516": "Яйцо",
    "dietskoie-pitaniie-207": "Детское питание",
    "griby-788": "Грибы",
    "iagody-3951": "Ягоды",
    "sniekovaia-produktsiia-451": "Снековая продукция",
    "sousy-610": "Соусы",
    "pishchievyie-ingriediienty-426": "Пищевые ингредиенты",
    "pchielovodstvo-412": "Пчеловодство",
    "sielskoie-khoziaistvo-46": "Сельское хозяйство",
}
DEFAULT_CATEGORIES = list(CATEGORIES)
DELAY = 2.0  # секунд между запросами

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str, retries: int = 3, timeout: int = 25) -> str | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            print(f"  ! {r.status_code} {url}")
            if r.status_code in (403, 429):
                time.sleep(10)
        except requests.RequestException as e:
            print(f"  ! {e} ({url})")
        time.sleep(DELAY * (attempt + 1))
    return None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_listing(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for card in soup.select("div.card_item.firm"):
        link = card.select_one("div.text a.link")
        if not link:
            continue
        m = re.search(r"/producers/(\d+)/", link["href"])
        desc = card.select_one("div.item_descriptor")
        city = card.select_one("div.ii_city")
        cat = card.select_one("div.ii_category")
        goods = card.select_one("div.ii_goods a")
        cat_title = cat.get("title", "") if cat else ""
        cat_name = re.sub(r"^Категория:\s*", "", cat_title).split(",")[0].strip()
        goods_n = re.search(r"\d+", goods.get_text()) if goods else None
        rm = re.search(r"/r-([a-z0-9-]+-\d+)/", cat.select_one("a[href]")["href"]) \
            if cat and cat.select_one("a[href]") else None
        items.append({
            "id": int(m.group(1)) if m else None,
            "region_slug": rm.group(1) if rm else None,
            "name": clean(link.get_text()),
            "url": BASE + link["href"],
            "category": category,
            "subcategory": cat_name or None,
            "city": clean(city.get_text()) if city else None,
            "description": clean(desc.get_text()) if desc else None,
            "products_count": int(goods_n.group(0)) if goods_n else None,
        })
    return items


def parse_details(html: str) -> dict:
    """Контакты и подробности со страницы производителя (что найдётся)."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")
    # только отформатированные номера (+7 …, 8 …, со скобками/дефисами/пробелами), без голых 11 цифр
    phones = sorted(set(m for m in re.findall(
        r"(?:\+7|8)[\s-]*\(?\d{3}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}", text) if re.search(r"[\s()+-]", m)))
    emails = sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)))
    return {
        "phones": [clean(p) for p in phones][:3],
        "emails": emails[:3],
    }


def scrape_category(slug: str, limit: int, details: bool) -> list[dict]:
    results, seen = [], set()
    page = 1
    while len(results) < limit:
        url = f"{BASE}/producers/catalog-{slug}" + (f"/page-{page}" if page > 1 else "")
        print(f"[{slug}] стр. {page}: {url}")
        html = fetch(url)
        if not html:
            break
        new = [i for i in parse_listing(html, slug) if i["url"] not in seen]
        if not new:
            break
        for i in new:
            seen.add(i["url"])
            results.append(i)
        page += 1
        time.sleep(DELAY)
    results = results[:limit]
    if details:
        for n, item in enumerate(results, 1):
            print(f"  детали {n}/{len(results)}: {item['name']}")
            html = fetch(item["url"])
            if html:
                item.update(parse_details(html))
            time.sleep(DELAY)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("-n", "--limit", type=int, default=30, help="производителей на категорию")
    ap.add_argument("-o", "--output", default="suppliers.json")
    ap.add_argument("--details", action="store_true", help="открывать страницу каждого производителя")
    args = ap.parse_args()

    by_id: dict = {}
    for slug in args.categories:
        for item in scrape_category(slug, args.limit, args.details):
            if item["id"] in by_id:  # компания в нескольких категориях
                by_id[item["id"]]["categories"].append(slug)
            else:
                item["categories"] = [item.pop("category")]
                by_id[item["id"]] = item
    data = list(by_id.values())

    out = {
        "source": BASE,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(data),
        "category_names": {k: v for k, v in CATEGORIES.items() if k in args.categories},
        "items": data,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(data)} записей -> {args.output}")


if __name__ == "__main__":
    main()

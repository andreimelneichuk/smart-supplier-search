"""Парсер объявлений поставщиков с agroserver.ru.

Использование:
    python agroserver_parser.py                       # категории по умолчанию
    python agroserver_parser.py -c bakaleya griby -n 20 -o ../data/suppliers.json

Собирает карточки со страниц категорий (название, цена, город, описание,
компания, телефоны, ссылка) и сохраняет в JSON.
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

BASE = "https://agroserver.ru"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; supplier-mvp-bot/0.1)"}
DEFAULT_CATEGORIES = ["bakaleya", "frukty-yagody", "chay-kofe-kakao", "griby"]
DELAY = 1.5  # секунд между запросами — не нагружаем сайт

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                return r.text
            print(f"  ! {r.status_code} {url}")
        except requests.RequestException as e:
            print(f"  ! {e} ({url})")
        time.sleep(DELAY * (attempt + 1))
    return None


def clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def parse_price(raw: str) -> dict:
    """'от 3 700 руб /  кг' -> {price_value: 3700.0, price_unit: 'кг', ...}"""
    raw = clean(raw)
    m = re.search(r"([\d\s]+(?:[.,]\d+)?)\s*руб\s*/\s*(\S+)", raw)
    value = unit = None
    if m:
        value = float(m.group(1).replace(" ", "").replace("\xa0", "").replace(",", "."))
        unit = m.group(2)
    return {"price_raw": raw, "price_value": value, "price_unit": unit}


def parse_listing(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for line in soup.select("div.line[id^=p]"):
        a = line.select_one("div.th a")
        if not a:
            continue
        price = line.select_one("div.price")
        geo = line.select_one("div.geo")
        text = line.select_one("div.text")
        org = line.select_one("a.personal_org_menu")
        user_link = line.select_one('a[href^="/user/"][href$="/"]')
        phones = [clean(p.get_text()) for p in line.select("a.href_to_tel")]
        date = line.select_one("div.date")
        items.append({
            "title": clean(a.get_text()),
            "url": BASE + a["href"],
            "category": category,
            **parse_price(price.get_text() if price else ""),
            "city": clean(geo.get_text()) if geo else None,
            "description": clean(text.get_text()) if text else None,
            "company": clean(org.get_text()) if org else None,
            "company_url": BASE + user_link["href"] if user_link else None,
            "verified": line.select_one("span.verify") is not None,
            "phones": phones,
            "posted": clean(date.get_text()) if date else None,
        })
    return items


def next_page_url(html: str, current: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    # ссылка на следующую страницу — тот же путь категории, но с /N+1/ на конце
    m = re.search(r"/(\d+)/$", current)
    page = int(m.group(1)) if m else 1
    for a in soup.select("a[href]"):
        if a["href"].rstrip("/").endswith(f"/{page + 1}") and a["href"].startswith(
            "/" + current.replace(BASE, "").strip("/").split("/")[0]
        ):
            return BASE + a["href"]
    return None


def scrape_category(slug: str, limit: int) -> list[dict]:
    url = f"{BASE}/{slug}/"
    results: list[dict] = []
    seen = set()
    while url and len(results) < limit:
        print(f"[{slug}] {url}")
        html = fetch(url)
        if not html:
            break
        for item in parse_listing(html, slug):
            if item["url"] not in seen:
                seen.add(item["url"])
                results.append(item)
        url = next_page_url(html, url)
        time.sleep(DELAY)
    return results[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("-n", "--limit", type=int, default=15, help="объявлений на категорию")
    ap.add_argument("-o", "--output", default="suppliers.json")
    args = ap.parse_args()

    data = []
    for slug in args.categories:
        data.extend(scrape_category(slug, args.limit))

    out = {
        "source": BASE,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(data),
        "items": data,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(data)} записей -> {args.output}")


if __name__ == "__main__":
    main()

"""Подгрузка данных с productcenter.ru по запросу: поиск на сайте и контакты производителей."""
import time
from concurrent.futures import ThreadPoolExecutor

import db
import productcenter_parser as pc

CONTACTS_TTL = 14 * 24 * 3600  # контакты считаем свежими две недели
LIVE_TIMEOUT = 8               # живые запросы не должны вешать интерфейс
BLOCK_AFTER = 3                # столько сбоев подряд — считаем, что сайт нас притормозил
BLOCK_SECONDS = 300

_fails = 0
_blocked_until = 0.0


def is_blocked() -> bool:
    return time.time() < _blocked_until


def _fetch(url: str) -> str | None:
    """fetch с короткими таймаутами и «автоматом»: после серии сбоев живые запросы на паузе."""
    global _fails, _blocked_until
    if is_blocked():
        return None
    html = None
    try:  # одна попытка без пауз: pc.fetch на 403/429 спит 10+ секунд, а здесь ждёт пользователь
        r = pc.session.get(url, timeout=LIVE_TIMEOUT)
        if r.status_code == 200:
            r.encoding = "utf-8"
            html = r.text
    except Exception as e:
        print(f"  ! live fetch {url}: {e}")
    if html:
        _fails = 0
        return html
    _fails += 1
    if _fails >= BLOCK_AFTER:
        _blocked_until = time.time() + BLOCK_SECONDS
        _fails = 0
    return None


def parse_search(html: str) -> list[dict]:
    """Поиск на сайте отдаёт товары; сворачиваем их в производителей (по одному на компанию)."""
    import re
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    firms: dict[str, dict] = {}
    for card in soup.select("div.card_item.product"):
        comp = card.select_one("div.ii_company")
        a = comp.select_one("a[href]") if comp else None
        m = re.search(r"/producers/(\d+)/([\w-]+)", a["href"]) if a else None
        link = card.select_one("div.text a.link")
        if not m or not link:
            continue
        fid = m.group(2)  # slug компании: id в ссылке поиска — это id товара, не компании
        product = pc.clean(link.get_text())
        if fid in firms:
            firms[fid]["description"] += f" · {product}"
            continue
        city = card.select_one("div.ii_city")
        city_a = city.select_one("a[href]") if city else None
        rm = re.search(r"/r-([a-z0-9-]+-\d+)/", city_a["href"]) if city_a else None
        desc = card.select_one("div.item_descriptor")
        name = re.sub(r"^Производитель:\s*", "", comp.get("title", "")).strip() or product
        firms[fid] = {
            "id": int(m.group(1)), "slug": fid, "name": name, "url": pc.BASE + a["href"],
            "product_url": pc.BASE + link["href"],
            "region_slug": rm.group(1) if rm else None,
            "subcategory": product,
            "city": pc.clean(city.get_text()) if city else None,
            "description": f"{product}. " + (pc.clean(desc.get_text()) if desc else ""),
            "products_count": None, "categories": [],
        }
    return list(firms.values())


def search_live(con, query: str, pages: int = 2, city: str | None = None) -> int:
    """Ищет на сайте, кладёт новых производителей в БД. Возвращает число новых.
    Если задан город, сначала ищем «товар + город» (сайт ищет по И, выдача точнее), потом просто товар."""
    queries = ([(f"{query} {city}", 1)] if city else []) + [(query, pages)]
    new = 0
    for q, n_pages in queries:
        for page in range(1, n_pages + 1):
            url = f"{pc.BASE}/search/?q={requests_quote(q)}" + (f"&page={page}" if page > 1 else "")
            html = _fetch(url)
            items = parse_search(html) if html else []
            if not items:
                break
            for it in items:
                new += db.upsert(con, it)
    con.commit()
    return new


def requests_quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)


def _resolve_producer_url(item: dict) -> str | None:
    """У найденных поиском компаний в БД лежит ссылка на товар; настоящая страница компании — на нём."""
    import re
    html = _fetch(item["url"])
    if not html:
        return None
    m = re.search(rf'href="(/producers/\d+/{re.escape(item["slug"])})"', html)
    return pc.BASE + m.group(1) if m else None


def _fetch_one(item: dict):
    try:
        url = item["url"]
        if "/products/" in url:
            url = _resolve_producer_url(item)
            if not url:
                return item["id"], None, None
        html = _fetch(url)
        return item["id"], url, (pc.parse_details(html) if html else None)
    except Exception:
        return item["id"], None, None


def enrich_contacts(con, items: list[dict], workers: int = 3) -> int:
    """Подтягивает контакты тем, у кого их нет или они устарели. Возвращает число обновлённых."""
    now = time.time()
    todo = [i for i in items if not i.get("contacts_at") or now - i["contacts_at"] > CONTACTS_TTL]
    if not todo:
        return 0
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=min(len(todo), workers)) as ex:
            for sid, url, det in ex.map(_fetch_one, todo):
                if det is not None:
                    try:
                        if url:
                            db.set_url(con, sid, url)
                        db.save_contacts(con, sid, det["phones"], det["emails"])
                        done += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error in enrich_contacts executor: {e}")
    return done


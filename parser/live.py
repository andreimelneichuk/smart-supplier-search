"""Подгрузка данных с productcenter.ru по запросу: поиск на сайте и контакты производителей."""
import time
from concurrent.futures import ThreadPoolExecutor

import db
import productcenter_parser as pc
import ranking

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
            firms[fid]["found_products"].append(product)
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
            "products_count": None, "categories": [], "found_products": [product],
        }
    # по некоторым запросам сайт отдаёт не товары, а сразу карточки компаний
    for card in soup.select("div.card_item.firm"):
        link = card.select_one("div.text a.link[href]")
        m = re.search(r"/producers/(\d+)/([\w-]+)", link["href"]) if link else None
        if not m or m.group(2) in firms:
            continue
        city = card.select_one("div.ii_city")
        city_a = city.select_one("a[href]") if city else None
        rm = re.search(r"/r-([a-z0-9-]+-\d+)/", city_a["href"]) if city_a else None
        desc = card.select_one("div.item_descriptor")
        goods = card.select_one("div.ii_goods")
        firms[m.group(2)] = {
            "id": int(m.group(1)), "slug": m.group(2), "name": pc.clean(link.get_text()),
            "url": pc.BASE + link["href"],
            "region_slug": rm.group(1) if rm else None,
            "subcategory": None,
            "city": pc.clean(city.get_text()) if city else None,
            "description": pc.clean(desc.get_text()) if desc else "",
            "products_count": int(goods.get_text(strip=True)) if goods and goods.get_text(strip=True).isdigit() else None,
            "categories": [], "found_products": [], "from_search": True,
        }
    return list(firms.values())


def search_live(con, query: str, pages: int = 2, city: str | None = None, found: set | None = None) -> int:
    """Ищет на сайте, кладёт новых производителей в БД. Возвращает число новых.
    Если задан город, сначала ищем «товар + город» (сайт ищет по И, выдача точнее), потом просто товар.
    Если по всей фразе нашлось мало компаний, добавляем поиск по главному слову («креветки замороженные» → «креветки»).
    found — сюда складываются id всех компаний, которых сайт выдал по запросу (и новых, и уже известных)."""
    # «товар + город» сайт ищет по всем словам и отдаёт любые местные карточки со словом («чипсы из индейки»),
    # поэтому попаданием по запросу считаем только поиск по самой фразе
    queries = ([(f"{query} {city}", 1, True)] if city else []) + [(query, pages, False)]
    head = query.split()[0] if query.split() else ""
    if head and head != query:
        queries.append((head, pages, True))
    new, slugs = 0, set()
    for q, n_pages, broad in queries:
        if broad and q == head and len(slugs) >= 10:
            break  # по всей фразе нашлось достаточно
        for page in range(1, n_pages + 1):
            url = f"{pc.BASE}/search/?q={requests_quote(q)}" + (f"&page={page}" if page > 1 else "")
            html = _fetch(url)
            items = parse_search(html) if html else []
            if not items:
                break
            for it in items:
                if ranking.not_food(it.get("subcategory")):
                    continue  # сайт ищет по словам: на «айфон» отдаёт чехлы, на «мёд» — мыло с мёдом
                new += db.upsert(con, it)
                if not broad:  # по главному слову или с городом выдача слишком широкая — не считаем это попаданием
                    slugs.add(it["slug"])
    con.commit()
    if found is not None and slugs:
        marks = ",".join("?" * len(slugs))
        found.update(r[0] for r in con.execute(f"SELECT id FROM suppliers WHERE slug IN ({marks})", list(slugs)))
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



def _parse_product_cards(html: str) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for card in soup.select(".card_item.product"):
        link = card.select_one("a.link[href*='/products/']")
        if not link:
            continue
        price = card.select_one(".item_price b")
        img = card.select_one(".image img")
        src = img.get("src", "") if img else ""
        out.append({
            "name": link.get_text(" ", strip=True),
            "url": pc.BASE + link["href"] if link["href"].startswith("/") else link["href"],
            "price": f"от {price.get_text(strip=True)} ₽" if price else "",
            "image": pc.BASE + src if src.startswith("/") else src,
        })
    return out


def fetch_products(item: dict, limit: int = 60) -> list[dict]:
    """Товары компании со страницы производителя (в БД хранится только их число)."""
    import re
    url = item["url"]
    if "/products/" in url:
        url = _resolve_producer_url(item)
        if not url:
            return []
    html = _fetch(url)
    if not html:
        return []
    # на странице компании — только первые карточки; полный список — по ссылке «Смотреть всё»
    m = re.search(r'href="(/products/producer-\d+-[^"]+)"', html)
    if m:
        full = _fetch(pc.BASE + m.group(1))
        if full:
            html = full
    seen, items = set(), []
    for p in _parse_product_cards(html):
        if p["url"] not in seen:
            seen.add(p["url"])
            items.append(p)
    return items[:limit]

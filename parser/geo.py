"""Координаты городов и расстояния.

Источник 1: data/city.csv (открытая таблица городов РФ, hflabs/city) — работает офлайн.
Источник 2: геокодер OpenStreetMap Nominatim для того, чего нет в таблице (мелкие населённые
пункты, Москва, Санкт-Петербург). Результаты кэшируются в таблице geo_cache, лимит сервиса —
1 запрос в секунду.

    python geo.py --warm     # геокодировать все города из базы, которых нет в таблице
"""
import csv
import math
import sqlite3
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
CSV_PATH = HERE / "data" / "city.csv"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "supplier-search-mvp/0.1 (educational project)"

# города федерального значения в таблице идут как регионы, добавляем вручную
MANUAL = {
    "москва": (55.7558, 37.6173),
    "санкт-петербург": (59.9343, 30.3351),
    "севастополь": (44.6167, 33.5254),
}

_static: dict[str, list[tuple[int, float, float, str]]] | None = None
_regions: list[tuple[str, str]] = []  # (регион в нижнем регистре, полное название «Московская область»)
_last_request = 0.0


def norm(name: str) -> str:
    return (name or "").strip().lower().replace("ё", "е")


def _load_static():
    """имя города -> список кандидатов (население, lat, lon, регион) — тёзки в разных регионах."""
    global _static
    if _static is not None:
        return _static
    st: dict[str, list] = {}
    full: dict[str, str] = {}
    if CSV_PATH.exists():
        for r in csv.DictReader(CSV_PATH.open(encoding="utf-8")):
            if r["region"] and norm(r["region"]) not in full:
                t, n = r["region_type"], r["region"]
                full[norm(n)] = (f"Республика {n}" if t == "Респ" else f"{n} область" if t == "обл"
                                 else f"{n} край" if t == "край" else (n if "округ" in n.lower() else f"{n} автономный округ") if "АО" in t
                                 else n)
            name = r["city"] or r["settlement"]
            if name and r["geo_lat"]:
                st.setdefault(norm(name), []).append(
                    (int(r["population"] or 0), float(r["geo_lat"]), float(r["geo_lon"]), norm(r["region"])))
    for k, (lat, lon) in MANUAL.items():
        st[norm(k)] = [(10**8, lat, lon, norm(k))]
    _regions.extend(full.items())
    _static = st
    return st


def _region_matches(csv_region: str, site_region: str) -> bool:
    """«Пермский» (таблица) vs «Пермского края» (сайт, родительный падеж): сравниваем основу первого слова
    без окончания, чтобы «Московская» не совпала с «Москва»."""
    if not csv_region:
        return False
    return _stem(csv_region) in norm(site_region)


def _stem(region: str) -> str:
    word = region.split()[0]
    return word if len(word) <= 5 else word[:-2]


def region_nominative(site_region: str | None) -> str | None:
    """«Московской области» -> «Московская область» (геокодер плохо понимает родительный падеж)."""
    if not site_region:
        return None
    _load_static()
    hits = [(len(_stem(reg)), full_name) for reg, full_name in _regions if _region_matches(reg, site_region)]
    return max(hits)[1] if hits else site_region  # самая длинная основа: «московск» бьёт «моск»


def _ensure_cache(con: sqlite3.Connection):
    con.execute("CREATE TABLE IF NOT EXISTS geo_cache (city TEXT PRIMARY KEY, lat REAL, lon REAL)")


def _cache_key(city: str, region: str | None) -> str:
    return norm(city) + ("|" + norm(region)[:25] if region else "")


def cached_coords(con: sqlite3.Connection, city: str, region: str | None = None):
    """Только офлайн: таблица + кэш. None — координат пока нет.
    Если регион известен, а в таблице тёзка из другого региона, таблице не верим."""
    if not city:
        return None
    cands = _load_static().get(norm(city))
    if cands:
        if region:
            for pop, lat, lon, reg in sorted(cands, reverse=True):
                if _region_matches(reg, region) or reg == norm(city):  # Москва/СПб — регион = город
                    return (lat, lon)
        else:
            _, lat, lon, _ = max(cands)
            return (lat, lon)
    _ensure_cache(con)
    row = con.execute("SELECT lat, lon FROM geo_cache WHERE city=?", (_cache_key(city, region),)).fetchone()
    if row:
        return None if row[0] is None else (row[0], row[1])
    return None


def geocode(con: sqlite3.Connection, city: str, region: str | None = None):
    """Сначала офлайн; если нет — один запрос в Nominatim «город, регион, Россия» (пауза 1 с), результат в кэш."""
    global _last_request
    hit = cached_coords(con, city, region)
    if hit:
        return hit
    key = _cache_key(city, region)
    _ensure_cache(con)
    if con.execute("SELECT 1 FROM geo_cache WHERE city=?", (key,)).fetchone():
        return None  # уже пробовали и не нашли
    time.sleep(max(0.0, 1.1 - (time.time() - _last_request)))
    _last_request = time.time()
    q = ", ".join(x for x in (city, region_nominative(region), "Россия") if x)
    try:
        r = requests.get(NOMINATIM, params={"q": q, "format": "json", "limit": 1, "countrycodes": "ru"},
                         headers={"User-Agent": UA}, timeout=10)
        if r.status_code != 200:
            return None  # сбой сервиса — в кэш не пишем, попробуем позже
        data = r.json()
    except (requests.RequestException, ValueError):
        return None  # сбой сети — в кэш не пишем, попробуем позже
    lat, lon = (float(data[0]["lat"]), float(data[0]["lon"])) if data else (None, None)
    con.execute("INSERT OR REPLACE INTO geo_cache (city, lat, lon) VALUES (?,?,?)", (key, lat, lon))
    con.commit()
    return None if lat is None else (lat, lon)


def distance_km(a, b) -> float:
    """Расстояние по дуге большого круга (формула гаверсинуса)."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def warm(con: sqlite3.Connection):
    pairs = con.execute("SELECT DISTINCT city, region FROM suppliers WHERE city IS NOT NULL").fetchall()
    todo = [(c, r) for c, r in pairs if cached_coords(con, c, r) is None
            and not con.execute("SELECT 1 FROM geo_cache WHERE city=?", (_cache_key(c, r),)).fetchone()]
    print(f"Пар город+регион: {len(pairs)}, надо геокодировать: {len(todo)}")
    ok = 0
    for n, (c, r) in enumerate(todo, 1):
        res = geocode(con, c, r)
        ok += res is not None
        print(f"  {n}/{len(todo)} {c} [{r}]: {'%.2f, %.2f' % res if res else 'не найден'}")
    print(f"Найдено: {ok} из {len(todo)}")


if __name__ == "__main__":
    import db
    warm(db.connect())

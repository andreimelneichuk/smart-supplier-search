"""Разбор запроса и ранжирование поставщиков."""
import math
import re

import geo
import search

STOPWORDS = {
    "оптом", "опт", "купить", "поставщик", "поставщики", "поставщиков", "производитель",
    "производители", "производителей", "компания", "компании", "от", "в", "на", "для", "и",
    "по", "из", "с", "нужен", "нужны", "ищу", "найти", "найди", "дешево", "недорого",
}
# веса полей при поиске слова
W_NAME, W_SUB, W_CAT, W_DESC = 1.0, 0.9, 0.6, 0.5
# веса итоговой оценки
WEIGHTS = {"relevance": 0.50, "geo": 0.20, "completeness": 0.15, "size": 0.15}
MIN_RELEVANCE = 0.30
GEO_SCALE_KM = 400  # гео-оценка = exp(-расстояние / 400 км): 0 км → 1.0, 300 км → 0.47, 1100 км → 0.06


def stem(w: str) -> str:
    """Грубый «стемминг» для русского: режем окончание, сравниваем по основе."""
    w = w.lower().replace("ё", "е")
    return w[:5] if len(w) > 5 else w[: max(3, len(w) - 1)]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


CITY_ALIASES = {
    "питер": "Санкт-Петербург",
    "питере": "Санкт-Петербург",
    "питера": "Санкт-Петербург",
    "петербург": "Санкт-Петербург",
    "петербурге": "Санкт-Петербург",
    "петербурга": "Санкт-Петербург",
    "спб": "Санкт-Петербург",
    "питерский": "Санкт-Петербург",
    "мск": "Москва",
    "москва": "Москва",
    "москве": "Москва",
    "москвы": "Москва",
    "московский": "Москва",
    "екб": "Екатеринбург",
    "екатеринбург": "Екатеринбург",
    "екатеринбурге": "Екатеринбург",
    "нск": "Новосибирск",
    "новосибирск": "Новосибирск",
    "новосибирске": "Новосибирск",
    "нн": "Нижний Новгород",
    "нижний": "Нижний Новгород",
    "нижнем": "Нижний Новгород",
    "казань": "Казань",
    "казани": "Казань",
    "ростов": "Ростов-на-Дону",
    "ростове": "Ростов-на-Дону",
    "самара": "Самара",
    "самаре": "Самара",
    "воронеж": "Воронеж",
    "воронеже": "Воронеж",
    "краснодар": "Краснодар",
    "краснодаре": "Краснодар",
    "уфа": "Уфа",
    "уфе": "Уфа",
    "сочи": "Сочи",
    "барнаул": "Барнаул",
    "барнауле": "Барнаул",
}


def parse_query(q: str, known_cities: list[str]) -> dict:
    """-> {'terms': [...], 'city': 'Москва'|None, 'raw': q}. Город определяем по алиасам или основе слова."""
    city_by_stem = {stem(t): c for c in known_cities for t in tokens(c)}
    terms, city = [], None
    for t in tokens(q):
        if t in STOPWORDS:
            continue
        # 1. Проверяем разговорные и падежные формы (Питер, Москве, СПб)
        if t in CITY_ALIASES and city is None:
            city = CITY_ALIASES[t]
            continue
        # 2. Проверяем совпадение по корню города из базы
        if stem(t) in city_by_stem and city is None:
            city = city_by_stem[stem(t)]
            continue
        terms.append(t)
    return {"terms": terms, "city": city, "raw": q}


def relevance_stem(s: dict, terms: list[str], cat_names: dict) -> float:
    """Старая версия (стемы без учёта роли слова) — осталась как baseline для eval_search."""
    if not terms:
        return 1.0
    name = [stem(t) for t in tokens(s["name"])]
    sub = [stem(t) for t in tokens(s.get("subcategory"))]
    cat = [stem(t) for c in s.get("categories", []) for t in tokens(cat_names.get(c, ""))]
    desc = [stem(t) for t in tokens(s.get("description"))]
    total = 0.0
    for t in terms:
        st = stem(t)
        total += max(
            W_NAME if st in name else 0,
            W_SUB if st in sub else 0,
            W_CAT if st in cat else 0,
            W_DESC if st in desc else 0,
        )
    return total / len(terms)


def relevance(s: dict, terms: list[str], cat_names: dict) -> float:
    return search.lexical_score(s, search.query_lemmas(terms), cat_names)


def distance_to(s: dict, city: str | None, user_xy, coords) -> float | None:
    """Расстояние от поставщика до города покупателя, км (None — город не задан или координат нет)."""
    if not city:
        return None
    if s.get("city") == city:
        return 0.0
    if user_xy and coords and (s_xy := coords(s.get("city"), s.get("region"))):
        return geo.distance_km(user_xy, s_xy)
    return None


def rank(suppliers: list[dict], parsed: dict, cat_names: dict, coords=None, model: str = "bm25_cross") -> list[dict]:
    """coords(city, region) -> (lat, lon) | None — офлайн-функция координат (geo.cached_coords).
    model: 'bm25_cross' | 'bm25' | 'lexical' | 'llm'
    """
    import numpy as np
    user_xy = coords(parsed["city"], parsed.get("region")) if (coords and parsed["city"]) else None
    max_goods = max((s.get("products_count") or 0 for s in suppliers), default=0) or 1
    terms = parsed.get("terms", [])
    
    # 1. Считаем релевантность по выбранной модели
    if not terms:
        rel_scores = np.ones(len(suppliers), dtype=np.float32)
        min_gate = 0.0
    elif model == "bm25_cross":
        bm25_sc = search.bm25_scores(suppliers, terms, cat_names)
        max_b = float(np.max(bm25_sc)) if len(bm25_sc) > 0 else 0.0
        norm_b = (bm25_sc / max_b) if max_b > 0 else bm25_sc
        cand_indices = [i for i in np.argsort(-norm_b, kind="stable") if norm_b[i] > 0.0][:50]
        rel_scores = np.array(norm_b, dtype=np.float32)
        if cand_indices:
            import cross_encoder
            # в трансформер — только товарная часть запроса: город и «оптом» лишь размывают оценку
            scored = search.cross_encoder_rerank(" ".join(terms), suppliers, cand_indices)
            for idx, raw in scored:
                prob = cross_encoder.sigmoid(raw)
                rel_scores[idx] = float(0.3 * norm_b[idx] + 0.7 * prob)
        min_gate = 0.20
    elif model == "bm25":
        bm25_sc = search.bm25_scores(suppliers, terms, cat_names)
        max_b = float(np.max(bm25_sc)) if len(bm25_sc) > 0 else 0.0
        rel_scores = (bm25_sc / max_b) if max_b > 0 else bm25_sc
        min_gate = 0.15
    elif model == "llm":
        bm25_sc = search.bm25_scores(suppliers, terms, cat_names)
        cand_indices = [i for i in np.argsort(-bm25_sc, kind="stable") if bm25_sc[i] > 0.0][:25]
        rel_scores = np.zeros(len(suppliers), dtype=np.float32)
        if cand_indices:
            import llm_rerank
            grades = llm_rerank.grade(parsed["raw"], [suppliers[i] for i in cand_indices])
            for idx, g in zip(cand_indices, grades):
                if g is not None:
                    rel_scores[idx] = float(g / 3.0)
        min_gate = 0.20
    else:  # 'lexical'
        rel_scores = search.lexical_scores(suppliers, terms, cat_names)
        min_gate = MIN_RELEVANCE

    out = []
    for i, s in enumerate(suppliers):
        # Отсекаем непищевую продукцию (мыло, упаковка, косметика, оборудование), если она размечена LLM
        if s.get("is_food") == 0:
            continue
        rel = float(rel_scores[i])
        if rel < min_gate:
            continue
        dist = distance_to(s, parsed["city"], user_xy, coords)
        geo_score = 0.0 if dist is None else math.exp(-dist / GEO_SCALE_KM)
        comp = (0.5 if s.get("phones") else 0) + (0.5 if s.get("emails") else 0)
        size = math.log1p(s.get("products_count") or 0) / math.log1p(max_goods)
        parts = {"relevance": rel, "geo": geo_score, "completeness": comp, "size": size}
        w = dict(WEIGHTS)
        if not parsed["city"]:
            w["relevance"] += w.pop("geo")
            parts.pop("geo")
        score = sum(w[k] * v for k, v in parts.items())
        out.append({
            **s,
            "distance_km": None if dist is None else round(dist),
            "score": round(score, 4),
            "parts": {k: round(v, 2) for k, v in parts.items()}
        })
    return sorted(out, key=lambda r: r["score"], reverse=True)


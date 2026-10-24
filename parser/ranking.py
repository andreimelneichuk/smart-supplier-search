"""Разбор запроса и ранжирование поставщиков."""
import json
import math
import re
from pathlib import Path

import bm25
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
WEIGHTS = {"relevance": 0.40, "geo": 0.35, "completeness": 0.15, "size": 0.10}  # подобраны массовым тестом: запросы × города
MIN_RELEVANCE = 0.30
GEO_SCALE_KM = 400  # гео-оценка = exp(-(расстояние - до ближайшего) / 400 км): ближайший → 1.0, +300 км → 0.47
GEO_REL_GATE = 0.6  # с какой релевантности близость учитывается в полную силу
NON_FOOD_FACTOR = 0.3  # непищевые карточки без разметки is_food: оборудование, тара, инвентарь
NON_FOOD_HEADS = {"оборудование", "грунт", "удобрение", "нож", "лоток", "ящик", "камера", "станок", "машина",
                  "аппарат", "линия", "холодильник", "инвентарь", "посуда", "пленка", "пакет", "коробка",
                  "этикетка", "семя", "рассада", "саженец", "техника", "установка", "печь"}
DERIVED_FACTOR = 0.6  # те же «корм, приправа, упаковка, блюдо», что и в рубрике llm_rerank
DERIVED_HEADS = {"снек", "снэк", "чипсы", "корм", "консервы", "паштет", "приправа", "специя", "соус",
                 "добавка", "ароматизатор", "упаковка", "тара"}
SECONDARY_FACTOR = 0.7  # искомый товар не в специализации/названии компании, а только в описании
PARENT_REL = 0.45       # специализация — родительская категория из справочника (птица для индейки)
APPOSITION_REL = 0.35   # «сыр моцарелла» → просто сыроварня: слабее, специалисты по моцарелле должны быть выше
# родительские категории: из ОКПД2 (build_okpd_parents.py — виды рыбы, птицы, мяса) и вручную поверх
PARENT_PRODUCT = {**json.loads((Path(__file__).parent / "data" / "okpd_parents.json").read_text(encoding="utf-8")),
    "индейка": "птица", "курица": "птица", "цыпленок": "птица", "бройлер": "птица", "утка": "птица",
    "гусь": "птица", "перепел": "птица", "говядина": "мясо", "свинина": "мясо", "баранина": "мясо",
    "телятина": "мясо", "конина": "мясо", "крольчатина": "мясо",
}
HEAD_ONLY_FACTOR = 0.5  # запрос «товар + признак», а у поставщика совпал только товар
MODIFIER_POS = {"ADJF", "ADJS", "PRTF", "PRTS"}
SITE_REL_BONUS = 0.15  # компанию выдал поиск сайта по всей фразе запроса — небольшая прибавка к релевантности


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


def non_food_clash(s: dict, query_lemmas: set) -> bool:
    """Главное слово специализации — непищевой товар (оборудование, тара, грунт), которого нет в запросе."""
    head = next((l for l, pos, _ in search.content_lemmas(s.get("subcategory")) if pos == "NOUN"), None)
    return head in NON_FOOD_HEADS and head not in query_lemmas


def derived_clash(s: dict, query_lemmas: set) -> bool:
    """Специализация компании — производный продукт, которого нет в запросе («Мясные снеки» на «мясо индейки»)."""
    sub = set(bm25.tokenize_lemmas(f"{s.get('subcategory') or ''} {s.get('name') or ''}"))  # «Производитель корма…»
    return bool(sub & DERIVED_HEADS) and not (sub & DERIVED_HEADS & query_lemmas)


def is_main_product(s: dict, lemma: str, with_name: bool = True) -> bool:
    """Товар — в специализации, названии компании или ключевых словах каталога, а не только в описании."""
    fields = (s.get("subcategory"), s.get("name") if with_name else None, s.get("search_keywords"),
              " ".join(s.get("primary_products") or []))
    text = " ".join(str(x) for x in fields if x)
    return any(t == lemma or bm25.is_compound_of(t, lemma) for t in bm25.tokenize_lemmas(text))


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

    # «хлеб ржаной»: пекарня, у которой в карточке нет слова «ржаной», всё равно подходит — реранкер же ставит
    # таким почти ноль и они отсекаются порогом. Если остальные слова — только признаки (прилагательные),
    # даём производителям главного товара не меньше HEAD_ONLY_FACTOR от их совпадения по главному слову.
    # «мясо индейки», «сыр моцарелла» не трогаем: там второе слово и есть товар.
    head_floor = product = parent = None
    parent_rel = PARENT_REL
    if terms and model != "llm":
        lem = search.content_lemmas(" ".join(terms))
        nouns = [(l, role) for l, pos, role in lem if pos == "NOUN"]
        if len(terms) > 1 and len(nouns) == 1 and all(pos in MODIFIER_POS for l, pos, _ in lem if l != nouns[0][0]):
            head_floor = nouns[0][0]
        # искомый товар — последнее существительное: «мясо ИНДЕЙКИ», «сыр МОЦАРЕЛЛА», «ХЛЕБ ржаной»
        product = nouns[-1][0] if nouns else None
        # родительская категория: «индейка» → «птица»; «сыр моцарелла» → «сыр» (приложение, не родительный падеж)
        parent = PARENT_PRODUCT.get(product) if product else None
        parent_rel = PARENT_REL
        if not parent and len(nouns) > 1 and nouns[-1][1] != "gen":
            parent, parent_rel = nouns[0][0], APPOSITION_REL

    query_lemmas = set(bm25.tokenize_lemmas(" ".join(terms)))
    site_ids = parsed.get("site_ids") or set()
    kept = []
    for i, s in enumerate(suppliers):
        # Отсекаем непищевую продукцию (мыло, упаковка, косметика, оборудование), если она размечена LLM
        if s.get("is_food") == 0:
            continue
        rel = float(rel_scores[i])
        if head_floor and rel < HEAD_ONLY_FACTOR:
            rel = max(rel, HEAD_ONLY_FACTOR * search.lexical_score(s, [head_floor], cat_names))
        if non_food_clash(s, query_lemmas):
            # «Овощные грунты», «Ножи для резки овощей», «Лотки для овощей» по запросу «овощи замороженные»
            rel *= NON_FOOD_FACTOR
        derived = derived_clash(s, query_lemmas)
        if derived:
            # специализация — производный продукт (снеки, корм, консервы, соус), а искали само сырьё:
            # «чипсы из мяса индейки» — не поставщик мяса индейки
            rel *= DERIVED_FACTOR
        if product and not is_main_product(s, product):
            # товар есть только в описании, а основная специализация другая: консервный завод, у которого
            # среди прочего есть хлеб, не должен стоять наравне с хлебозаводами
            rel *= SECONDARY_FACTOR
            # по специализации, а не по названию: «Птицефабрика» с товаром «Яйцо куриное» индейку не продаёт
            if parent and is_main_product(s, parent, with_name=False):
                rel = max(rel, parent_rel)  # птицефабрика по запросу «мясо индейки»
        if s["id"] in site_ids and rel > 0 and not derived:  # сайт тоже ищет по словам и путает снеки с мясом
            rel = min(1.0, rel + SITE_REL_BONUS)
        if rel < min_gate:
            continue
        kept.append((s, rel, distance_to(s, parsed["city"], user_xy, coords)))
    # расстояние считаем от ближайшего подходящего поставщика: в Приморье все дальше 2000 км,
    # и абсолютная шкала давала всем ноль — ближайшие не отличались от самых далёких
    nearest = min((d for _, _, d in kept if d is not None), default=0.0)

    out = []
    for s, rel, dist in kept:
        geo_score = 0.0 if dist is None else math.exp(-(dist - nearest) / GEO_SCALE_KM)
        # близость помогает только подходящим по товару: иначе соседний завод с чужим товаром обгоняет нужный
        geo_score *= min(1.0, rel / GEO_REL_GATE)
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


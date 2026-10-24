"""Поиск релевантности: лемматизация, веса полей, позиция слова в названии (этап 1)
и гибрид с эмбеддингами через RRF (этап 2).

    lexical_scores(docs, terms)  -> [0..1] на документ   (этап 1)
    dense_scores(docs, query)    -> [0..1] на документ   (косинус, нормированный)
    hybrid_relevance(docs, terms, query) -> [0..1]       (этап 2: RRF по обоим рейтингам)
"""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymorphy3

from bm25 import is_compound_of

HERE = Path(__file__).parent
EMB_CACHE = HERE / "data" / "embeddings.npz"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_morph = pymorphy3.MorphAnalyzer()
SKIP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "PRED"}
STOP_LEMMAS = {"производитель", "производство", "компания", "оптом", "опт", "продукция", "продукт",
               "поставщик", "завод", "фабрика", "ооо", "ип", "ао", "тм"}

# предлоги, после которых слово — не сам продукт: «приправа ДЛЯ курицы», «пельмени С курицей»
ACCESSORY_PREPS = {"для", "с", "со", "на", "под", "к", "в", "по"}
# «головы» названия, которые не продукт питания: «Аромат курицы», «Комбикорм курица»
NON_FOOD_HEADS = {"аромат", "комбикорм", "корм", "оболочка", "приправа", "специя", "соус", "упаковка",
                  "решетка", "ароматизатор", "вкус", "добавка"}
GENITIVE_FACTOR, ACCESSORY_FACTOR = 0.9, 0.3
# слово есть только в категории/описании компании, а в названии товара его нет: «Чай» у мёдной компании
PARTIAL_FACTOR, SAME_FIELD_BONUS = 0.6, 0.1  # многословный запрос: покрыта только часть слов / все слова в подкатегории
NAME_ONLY_FACTOR = 0.6  # слово только в названии компании, а товар (подкатегория) другой: «Производитель мёда» -> «Чай»
NO_PRODUCT_MATCH_FACTOR = 1.0  # 0.5 проверен: P@10 0.79 -> 0.75, а «Чай» у мёдной компании не ушёл (слово в названии компании)

# веса полей и позиции совпадения
W_SUB, W_NAME, W_CAT, W_DESC = 1.0, 0.85, 0.5, 0.35
MODIFIER_FACTOR = 0.55  # слово найдено не в «главной» позиции названия: «Яйцо КУРИНОЕ», «Соус из КУРИЦЫ»


COMPOUND_FACTOR = 0.8  # «хлебокомбинат» для «хлеб»
@lru_cache(maxsize=50000)
def _parse(word: str):
    p = _morph.parse(word)[0]
    return p.normal_form.replace("ё", "е"), p.tag.POS, p.tag.case


def content_lemmas(text: str | None) -> list[tuple[str, str, str]]:
    """[(лемма, часть речи, роль)] без предлогов, союзов и служебных слов, порядок сохранён.
    Роль: "gen" — родительный падеж без «плохого» предлога («мясо КУРИЦЫ»), "acc" — после «для/с/на…»,
    иначе ""."""
    out = []
    prep = None
    for w in re.findall(r"[a-zа-яё0-9]+", (text or "").lower()):
        lemma, pos, case = _parse(w)
        if pos == "PREP":
            prep = lemma
            continue
        if pos in SKIP_POS or lemma in STOP_LEMMAS or (len(lemma) < 2 and not lemma.isdigit()):
            continue
        role = "acc" if prep in ACCESSORY_PREPS else ("gen" if case == "gent" and pos == "NOUN" else "")
        out.append((lemma, pos or "", role))
        prep = None
    return out


def query_lemmas(terms: list[str]) -> list[str]:
    return [l for t in terms for l, *_ in content_lemmas(t)]


def _field_match(lemma: str, tokens: list[tuple[str, str, str]], head_weighted: bool) -> float:
    """1.0, если лемма — «голова» названия (первое существительное). Иначе по роли слова:
    «мясо КУРИЦЫ» (родительный) — почти полное совпадение, «приправа ДЛЯ курицы» — слабое,
    прочее — MODIFIER_FACTOR."""
    hit = next(((l, pos, role) for l, pos, role in tokens if l == lemma), None)
    if hit is None:
        comp = next(((l, pos, role) for l, pos, role in tokens if is_compound_of(l, lemma)), None)
        if comp is None:
            return 0.0
        # «хлебокомбинат», «хлебобулочный» для «хлеб»: совпадение есть, но слабее точного
        return COMPOUND_FACTOR * _field_match(comp[0], tokens, head_weighted)
    if not head_weighted:
        return 1.0
    head = next((l for l, pos, _ in tokens if pos == "NOUN"), tokens[0][0] if tokens else None)
    if head == lemma:
        return 1.0
    if hit[2] == "acc":
        return ACCESSORY_FACTOR
    if head in NON_FOOD_HEADS:
        return ACCESSORY_FACTOR
    if hit[2] == "gen":
        return GENITIVE_FACTOR
    return MODIFIER_FACTOR


@lru_cache(maxsize=20000)
def _doc_tokens(sub, name, cats, desc):
    return (content_lemmas(sub), content_lemmas(name), content_lemmas(cats), content_lemmas(desc))


def doc_fields(d: dict, cat_names: dict) -> tuple:
    cats = " ".join(cat_names.get(c, "") for c in d.get("categories", []))
    return _doc_tokens(d.get("subcategory") or "", d.get("name") or "", cats, d.get("description") or "")


def lexical_score(d: dict, lemmas: list[str], cat_names: dict) -> float:
    if not lemmas:
        return 1.0
    sub, name, cats, desc = doc_fields(d, cat_names)
    total, matched = 0.0, 0
    for lem in lemmas:
        in_sub = W_SUB * _field_match(lem, sub, True)
        in_name = W_NAME * _field_match(lem, name, True)
        if sub and not in_sub:
            in_name *= NAME_ONLY_FACTOR
        product = max(in_sub, in_name)
        company = max(W_CAT * _field_match(lem, cats, False), W_DESC * _field_match(lem, desc, False))
        best = product if product else company * NO_PRODUCT_MATCH_FACTOR
        total += best
        matched += best > 0
    score = total / len(lemmas)
    if len(lemmas) > 1:
        if matched < len(lemmas):
            score *= PARTIAL_FACTOR          # «сыр» без «моцарелла»: только часть запроса
        elif all(any(l == lem for l, *_ in sub) for lem in lemmas):
            score = min(1.0, score + SAME_FIELD_BONUS)  # все слова в названии товара
    return score


def lexical_scores(docs: list[dict], terms: list[str], cat_names: dict) -> np.ndarray:
    lem = query_lemmas(terms)
    return np.array([lexical_score(d, lem, cat_names) for d in docs])


# ---------- эмбеддинги ----------
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL)
    return _model


def doc_text(d: dict) -> str:
    return f"{d.get('subcategory') or ''}. {d.get('name') or ''}. {(d.get('description') or '')[:220]}"


def _embed(texts: list[str]) -> np.ndarray:
    v = np.array(list(_get_model().embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def doc_matrix(docs: list[dict]) -> np.ndarray:
    """Векторы документов с кэшем на диске: пересчитываем только новые или изменившиеся тексты."""
    keys = [hashlib.md5(doc_text(d).encode()).hexdigest() for d in docs]
    cache: dict[str, np.ndarray] = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        new = _embed([doc_text(docs[i]) for i in missing])
        for i, v in zip(missing, new):
            cache[keys[i]] = v
        EMB_CACHE.parent.mkdir(exist_ok=True)
        np.savez(EMB_CACHE, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys])


def dense_cosine(docs: list[dict], query: str) -> np.ndarray:
    return doc_matrix(docs) @ _embed([query])[0]


def rrf(*scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: сумма 1/(k+ранг) по спискам; ранги считаются по убыванию оценки."""
    out = np.zeros(len(scores[0]))
    for s in scores:
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(len(s), dtype=int)
        ranks[order] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out


_bm25_index = None
_bm25_docs_key = None


def get_bm25(docs: list[dict], cat_names: dict):
    global _bm25_index, _bm25_docs_key
    # ключ по содержимому, а не по числу карточек: поиск на сайте дописывает товары в описания уже известных компаний
    key = hash(tuple((d["id"], d.get("subcategory"), d.get("description"), d.get("search_keywords")) for d in docs))
    if _bm25_index is None or _bm25_docs_key != key:
        import bm25
        _bm25_index = bm25.BM25Index(docs, cat_names)
        _bm25_docs_key = key
    return _bm25_index


def bm25_scores(docs: list[dict], terms: list[str], cat_names: dict) -> np.ndarray:
    idx = get_bm25(docs, cat_names)
    lemmas = query_lemmas(terms)
    return idx.score(lemmas)


def cross_encoder_rerank(query: str, docs: list[dict], candidate_indices: list[int]):
    import cross_encoder
    return cross_encoder.rerank_candidates(query, docs, candidate_indices)


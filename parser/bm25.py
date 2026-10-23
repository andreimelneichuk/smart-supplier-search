"""Канонический multi-field BM25 (Okapi BM25 с весами полей и лемматизацией).

Формула:
    IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
    TF_norm(q, d) = sum_f ( w_f * tf(q, d_f) / (1 - b_f + b_f * (len(d_f) / avgdl_f)) )
    BM25(q, d) = sum_q ( IDF(q) * TF_norm(q, d) * (k1 + 1) / (TF_norm(q, d) + k1) )
"""
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
STOP_LEMMAS = {
    "производитель", "производство", "компания", "оптом", "опт", "продукция", "продукт",
    "поставщик", "завод", "фабрика", "ооо", "ип", "ао", "тм", "купить", "цена", "прайс"
}
SKIP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "PRED"}

# Веса полей: нормализованные LLM-ключевые слова важнее всего, затем подкатегория товара, название компании и т.д.
FIELD_WEIGHTS = {
    "keywords": 4.0,
    "subcategory": 3.0,
    "name": 1.5,
    "categories": 1.0,
    "description": 0.8,
}
K1 = 1.5
B = 0.75


@lru_cache(maxsize=50000)
def _lemmatize(word: str) -> str | None:
    p = _morph.parse(word)[0]
    if p.tag.POS in SKIP_POS or len(word) < 2:
        return None
    lemma = p.normal_form.replace("ё", "е")
    return None if lemma in STOP_LEMMAS else lemma


def tokenize_lemmas(text: str | None) -> list[str]:
    if not text:
        return []
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    res = []
    for w in words:
        lem = _lemmatize(w)
        if lem:
            res.append(lem)
    return res


class BM25Index:
    """Индекс BM25 для коллекции документов поставщиков."""

    def __init__(self, docs: list[dict], cat_names: dict | None = None):
        self.docs = docs
        self.cat_names = cat_names or {}
        self.N = len(docs)

        # Извлекаем леммы по каждому полю для каждого документа
        self.doc_field_tokens: list[dict[str, list[str]]] = []
        self.field_lengths: dict[str, list[int]] = {f: [] for f in FIELD_WEIGHTS}
        self.doc_freqs: Counter = Counter()

        for d in docs:
            cats_text = " ".join(self.cat_names.get(c, "") for c in d.get("categories", []))
            fields = {
                "keywords": tokenize_lemmas(d.get("search_keywords")),
                "subcategory": tokenize_lemmas(d.get("subcategory")),
                "name": tokenize_lemmas(d.get("name")),
                "categories": tokenize_lemmas(cats_text),
                "description": tokenize_lemmas(d.get("description")),
            }
            self.doc_field_tokens.append(fields)

            all_doc_lemmas = set()
            for f, toks in fields.items():
                self.field_lengths[f].append(len(toks))
                all_doc_lemmas.update(toks)

            for lem in all_doc_lemmas:
                self.doc_freqs[lem] += 1

        # Средняя длина каждого поля
        self.avg_field_len = {
            f: (sum(lens) / self.N if self.N > 0 else 1.0) or 1.0
            for f, lens in self.field_lengths.items()
        }

    def idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query_lemmas: list[str]) -> np.ndarray:
        """Возвращает массив оценок BM25 для всех документов."""
        scores = np.zeros(self.N, dtype=np.float32)
        if not query_lemmas:
            return scores

        query_counts = Counter(query_lemmas)
        for term, q_cnt in query_counts.items():
            idf = self.idf(term)
            if idf <= 0:
                continue

            for i, fields in enumerate(self.doc_field_tokens):
                # Взвешенный нормализованный TF по полям
                norm_tf = 0.0
                for f, weight in FIELD_WEIGHTS.items():
                    toks = fields[f]
                    if not toks:
                        continue
                    tf = toks.count(term)
                    if tf > 0:
                        len_d = self.field_lengths[f][i]
                        avg_len = self.avg_field_len[f]
                        norm_tf += weight * (tf / (1.0 - B + B * (len_d / avg_len)))

                if norm_tf > 0:
                    term_score = idf * (norm_tf * (K1 + 1.0)) / (norm_tf + K1)
                    scores[i] += term_score

        return scores

"""Локальный Cross-Encoder реранкер на базе BAAI/bge-reranker-v2-m3-int8.

Работает локально через ONNX, без внешних API, без интернета и с нулевой стоимостью.
"""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np

MODEL_NAME = "BAAI/bge-reranker-v2-m3-int8"
_model = None


def get_reranker():
    global _model
    if _model is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _model = TextCrossEncoder(MODEL_NAME)
    return _model


def format_doc_for_rerank(d: dict) -> str:
    """Форматирует карточку с явными ролями полей для self-attention трансформера."""
    sub = d.get("subcategory") or "—"
    name = d.get("name") or ""
    desc = (d.get("description") or "").replace("\n", " ")[:300]
    return f"Товар: {sub}. Компания: {name}. Описание: {desc}"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


# (запрос, текст карточки) -> сырая оценка. Streamlit перезапускает скрипт на каждый клик,
# а пересчёт 50 пар занимает секунды — без кэша любое действие в интерфейсе «висит».
_score_cache: dict[tuple[str, str], float] = {}
_CACHE_MAX = 50000


def rerank_candidates(query: str, docs: list[dict], candidate_indices: list[int]) -> list[tuple[int, float]]:
    """Переранжирует список кандидатов по индексам.

    -> [(index, score), ...] отсортированный по убыванию оценки cross-encoder.
    """
    if not candidate_indices:
        return []

    texts = [format_doc_for_rerank(docs[i]) for i in candidate_indices]
    todo = [t for t in dict.fromkeys(texts) if (query, t) not in _score_cache]
    if todo:
        if len(_score_cache) > _CACHE_MAX:
            _score_cache.clear()
        for t, s in zip(todo, get_reranker().rerank(query, todo)):
            _score_cache[(query, t)] = float(s)

    scored = [(idx, _score_cache[(query, texts[pos])]) for pos, idx in enumerate(candidate_indices)]
    scored.sort(key=lambda x: -x[1])
    return scored


def rerank_scores(query: str, docs: list[dict], base_scores: np.ndarray, top_n: int = 30, weight: float = 0.7) -> np.ndarray:
    """Смешивает базовые оценки с оценками cross-encoder для топ-N кандидатов."""
    if len(docs) == 0:
        return np.zeros(0)
    
    order = np.argsort(-base_scores, kind="stable")[:top_n]
    if len(order) == 0:
        return np.array(base_scores)
    
    scored = rerank_candidates(query, docs, list(order))
    out = np.array(base_scores, copy=True)
    
    for idx, raw_score in scored:
        prob = sigmoid(raw_score)
        out[idx] = (1.0 - weight) * base_scores[idx] + weight * prob
        
    return out

"""Проверка на отложенных запросах: их не использовали ни для настройки весов, ни для правки синонимов.

    python eval_holdout.py

Метки: eval_holdout.json = {"запрос": [slug релевантных компаний, ...]}.
Разметка сделана по подкатегории и основным товарам карточки; пул кандидатов — подстрокой по тем же полям,
поэтому компании, которые не содержат слово запроса ни в одном поле, в разметку не попали.
"""
import json
from pathlib import Path

import numpy as np

import db
import eval_search

HERE = Path(__file__).parent
LABELS = HERE / "eval_holdout.json"
K = eval_search.K


def main():
    docs = db.all_suppliers(db.connect())
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = {}
    for q, rel_slugs in labels.items():
        rel = set(rel_slugs)
        for name, idx in eval_search.variants(docs, q).items():
            top = [1 if docs[i]["slug"] in rel else 0 for i in idx[:K]]
            rows.setdefault(name, []).append(
                (sum(top) / K, eval_search.ndcg(top + [0] * (K - len(top))), len(rel)))
    print(f"отложенный набор: {len(labels)} запросов, P@{K} и nDCG@{K}")
    print(f"{'вариант':14s} P@{K}   nDCG@{K}")
    for name, r in rows.items():
        print(f"{name:14s} {np.mean([x[0] for x in r]):.3f}  {np.mean([x[1] for x in r]):.3f}")
    print("\nпо запросам (P@10 / nDCG@10 для сервиса, в скобках — размер разметки):")
    for j, q in enumerate(labels):
        s = rows["service"][j] if "service" in rows else rows[next(iter(rows))][j]
        print(f"  {q:18s} {s[0]:.2f} / {s[1]:.2f}  ({s[2]})")


if __name__ == "__main__":
    main()

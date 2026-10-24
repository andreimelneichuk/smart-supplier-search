"""Проверка на отложенных запросах: их не использовали ни для настройки весов, ни для правки синонимов.

    python eval_holdout.py          # простые однословные запросы
    python eval_holdout.py narrow   # узкие запросы «товар + признак», градуированная разметка

Метки: eval_holdout.json = {"запрос": [slug релевантных компаний, ...]}.
Разметка сделана по подкатегории и основным товарам карточки; пул кандидатов — подстрокой по тем же полям,
поэтому компании, которые не содержат слово запроса ни в одном поле, в разметку не попали.

Узкий набор: eval_holdout_narrow.json = {"запрос": {"2": [slug], "1": [slug]}}.
2 — признак назван в карточке («сулугуни», «нерафинированное»), проверено вручную;
1 — компания делает главный товар (сыр, масло), отобрано правилом по подкатегории, товарам и названию.
P@10 считаем по оценке ≥1 и отдельно по оценке 2, nDCG — градуированный.
"""
import sys
import json
from pathlib import Path

import numpy as np

import db
import eval_search

HERE = Path(__file__).parent
LABELS = HERE / "eval_holdout.json"
NARROW = HERE / "eval_holdout_narrow.json"
K = eval_search.K


def narrow():
    docs = db.all_suppliers(db.connect())
    labels = json.loads(NARROW.read_text(encoding="utf-8"))
    rows = {}
    for q, grades in labels.items():
        gain = {slug: int(g) for g, slugs in grades.items() for slug in slugs}
        for name, idx in eval_search.variants(docs, q).items():
            top = [gain.get(docs[i]["slug"], 0) for i in idx[:K]]
            pad = top + [0] * (K - len(top))
            rows.setdefault(name, []).append((sum(g >= 1 for g in top) / K, sum(g == 2 for g in top),
                                              eval_search.ndcg(pad, list(gain.values())), len(grades["2"])))
    print(f"узкий отложенный набор: {len(labels)} запросов")
    print(f"{'вариант':14s} P@{K}(≥1)  точных в топ-{K}  nDCG@{K}")
    for name, r in rows.items():
        print(f"{name:14s} {np.mean([x[0] for x in r]):.3f}     {sum(x[1] for x in r):3d} из {sum(min(x[3], K) for x in r)}"
              f"      {np.mean([x[2] for x in r]):.3f}")
    print("\nсервис по запросам: P@10(≥1) / точных в топе из размеченных / nDCG")
    for j, q in enumerate(labels):
        s = rows["service"][j]
        print(f"  {q:38s} {s[0]:.2f} / {s[1]} из {s[3]} / {s[2]:.2f}")


def main():
    docs = db.all_suppliers(db.connect())
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = {}
    for q, rel_slugs in labels.items():
        rel = set(rel_slugs)
        for name, idx in eval_search.variants(docs, q).items():
            top = [1 if docs[i]["slug"] in rel else 0 for i in idx[:K]]
            rows.setdefault(name, []).append(
                (sum(top) / K, eval_search.ndcg(top + [0] * (K - len(top)), [1] * len(rel)), len(rel)))
    print(f"отложенный набор: {len(labels)} запросов, P@{K} и nDCG@{K}")
    print(f"{'вариант':14s} P@{K}   nDCG@{K}")
    for name, r in rows.items():
        print(f"{name:14s} {np.mean([x[0] for x in r]):.3f}  {np.mean([x[1] for x in r]):.3f}")
    print("\nпо запросам (P@10 / nDCG@10 для сервиса, в скобках — размер разметки):")
    for j, q in enumerate(labels):
        s = rows["service"][j] if "service" in rows else rows[next(iter(rows))][j]
        print(f"  {q:18s} {s[0]:.2f} / {s[1]:.2f}  ({s[2]})")


if __name__ == "__main__":
    narrow() if sys.argv[1:] == ["narrow"] else main()

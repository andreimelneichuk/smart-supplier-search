"""Согласие двух разметчиков на основном наборе (eval_labels.json).

    python eval_agreement.py export   # agreement_sheet.csv: пул кандидатов 8 запросов, колонка «релевантно» пустая
    python eval_agreement.py score    # после того как второй человек заполнил колонку 1/0

Второй разметчик не видит первую разметку: в листе только карточки. Правило то же, что у первого:
1 — компания производит или поставляет искомый товар, 0 — нет (оборудование, корм, упаковка, другой товар).
Считаем долю совпадений и каппу Коэна (поправка на случайное совпадение).
"""
import csv
import json
import sys
from pathlib import Path

import db
import eval_search

HERE = Path(__file__).parent
SHEET = HERE / "agreement_sheet.csv"
QUERIES = ["сыр моцарелла", "колбаса копчёная", "креветки замороженные", "мёд липовый",
           "орехи грецкие", "тушенка", "яйцо перепелиное", "рыба соленая"]
FIELDS = ["запрос", "slug", "название", "подкатегория", "описание", "релевантно"]


def export():
    docs = db.all_suppliers(db.connect())
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for q in QUERIES:
            pool = []
            for idx in eval_search.variants(docs, q).values():
                pool += [i for i in idx[:eval_search.K] if i not in pool]
            for i in sorted(pool, key=lambda i: docs[i]["slug"]):  # без порядка выдачи — не подсказываем
                d = docs[i]
                w.writerow({"запрос": q, "slug": d["slug"], "название": d["name"], "подкатегория": d.get("subcategory"),
                            "описание": (d.get("description") or "")[:300], "релевантно": ""})
    print(f"лист для второго разметчика: {SHEET}")


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def score():
    labels = json.loads((HERE / "eval_labels.json").read_text(encoding="utf-8"))
    rows = [r for r in csv.DictReader(SHEET.open(encoding="utf-8")) if r["релевантно"].strip() in ("0", "1")]
    if not rows:
        sys.exit("в листе нет заполненных оценок")
    first = [int(r["slug"] in labels.get(r["запрос"], [])) for r in rows]
    second = [int(r["релевантно"]) for r in rows]
    print(f"оценено пар: {len(rows)}, совпадение {sum(x == y for x, y in zip(first, second)) / len(rows):.0%}, "
          f"каппа Коэна {kappa(first, second):.2f}")
    for q in dict.fromkeys(r["запрос"] for r in rows):
        ix = [i for i, r in enumerate(rows) if r["запрос"] == q]
        diff = [rows[i]["slug"] for i in ix if first[i] != second[i]]
        print(f"  {q:24s} {len(ix):3d} пар, расхождений {len(diff)}" + (f": {', '.join(diff[:5])}" if diff else ""))


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()

"""Какая доля карточек базы содержит каждое поле и насколько свежа база.

    python dbstats.py           # таблица в markdown для README
"""
import re
from datetime import datetime

import db

# цена и минимальный заказ в карточках обычно пишутся словами или с валютой: проверяем текст описания
PRICE_RE = re.compile(r"₽|руб|цен[аы]|стоимост", re.IGNORECASE)
MIN_ORDER_RE = re.compile(r"минимальн\w* (заказ|партия|объ[её]м|отгрузк)|мин\.? ?заказ|от \d+ ?(кг|шт|тонн|т\b)", re.IGNORECASE)


def _has_text(v) -> bool:
    return bool(v) and str(v).strip() not in ("[]", "{}", "null", "None")


def coverage(con):
    rows = con.execute(
        "SELECT s.id, s.phones, s.emails, s.city, s.description, s.products_count, s.categories, "
        "s.contacts_at, s.added_at, e.primary_products "
        "FROM suppliers s LEFT JOIN enriched_catalog e ON e.supplier_id = s.id").fetchall()
    n = len(rows) or 1  # пустая база: все доли 0, без деления на ноль
    fields = {
        "город": lambda r: _has_text(r["city"]),
        "телефон": lambda r: _has_text(r["phones"]),
        "e-mail": lambda r: _has_text(r["emails"]),
        "описание (от 100 знаков)": lambda r: len(r["description"] or "") >= 100,
        "ассортимент (число товаров)": lambda r: (r["products_count"] or 0) > 0,
        "категория": lambda r: _has_text(r["categories"]),
        "основные товары (LLM-разметка)": lambda r: _has_text(r["primary_products"]),
        "контакты проверены с сайта": lambda r: r["contacts_at"] is not None,
        "упомянута цена (текст карточки)": lambda r: bool(PRICE_RE.search(r["description"] or "")),
        "упомянут минимальный заказ (текст)": lambda r: bool(MIN_ORDER_RE.search(r["description"] or "")),
    }
    result = {name: sum(1 for r in rows if check(r)) / n for name, check in fields.items()}
    added = [r["added_at"] for r in rows if r["added_at"]]
    freshness = {
        "всего": len(rows),
        "последнее добавление": datetime.fromtimestamp(max(added)).strftime("%d.%m.%Y") if added else None,
        "контакты обновлялись": sum(1 for r in rows if r["contacts_at"]),
    }
    return result, freshness


PASSPORT_FIELDS = {"min_order": "минимальный заказ", "price_range": "цена",
                   "certificates": "сертификаты", "delivery_terms": "доставка"}


def passport_fill(con):
    """Сколько открытых паспортов нашли условие в полном тексте страницы: (число паспортов, {поле: сколько нашли})."""
    from advisor import is_known
    rows = con.execute(f"SELECT {', '.join(PASSPORT_FIELDS)} FROM supplier_extra").fetchall()
    return len(rows), {label: sum(1 for r in rows if is_known(r[f])) for f, label in PASSPORT_FIELDS.items()}


def main():
    con = db.connect()
    result, freshness = coverage(con)
    print(f"| поле | доля карточек |\n|---|---|")
    for name, share in result.items():
        print(f"| {name} | {share:.0%} |")
    print(f"\nВсего карточек: {freshness['всего']}. Последнее добавление: {freshness['последнее добавление']}. "
          f"Контакты проверялись с сайта у {freshness['контакты обновлялись']}.")
    n, found = passport_fill(con)
    print(f"\nОткрыто паспортов: {n}. Условие найдено в полном тексте: "
          + ", ".join(f"{k} — {v}" for k, v in found.items()) + ".")


if __name__ == "__main__":
    main()

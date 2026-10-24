"""Единый оффлайн-модуль LLM-обогащения каталога поставщиков (Groq / Qwen 3.8).

Объединяет два пайплайна в один экономичный запрос к LLM:
1. Выделение поисковых сущностей и классификация (в таблицу enriched_catalog):
   - is_food (отсекает непродовольственные компании, оборудование, тару)
   - primary_products (основная продукция: сахар, сыр, рыба, мясо)
   - specific_varieties (сорта/виды: сахар-песок, рафинад, моцарелла, индейка)
   - processing (способы обработки: соленая, рафинированный, копченая, заморозка)
   - dish_ingredients (ингредиенты блюд: сыр в пицце, сахар в варенье)
   - search_keywords (очищенные ключевые слова для гибридного поиска)

2. Извлечение B2B-условий сотрудничества (в таблицу supplier_extra):
   - min_order (минимальный объем партии / MOQ)
   - price_range (ценовой ориентир)
   - certificates (сертификаты: ГОСТ, ХАССП, ЕАС, Меркурий)
   - delivery_terms (условия логистики / рефрижератор / самовывоз)
   - region_scope (география поставок)
"""
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import db
from openai import OpenAI, RateLimitError, APIError

BATCH_SIZE = 2          # 2 компании на запрос (~500 токенов)
REQUEST_DELAY = 5.0     # задержка между запросами (сек), чтобы стабильно быть ниже 8 000 TPM на Groq


def load_env():
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


load_env()


def get_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("your_"):
        raise RuntimeError("Укажите валидный OPENAI_API_KEY в parser/.env")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=key, base_url=base_url if base_url else None)


def init_db(con: sqlite3.Connection):
    # Таблица поисковых сущностей
    con.execute("""
    CREATE TABLE IF NOT EXISTS enriched_catalog (
        supplier_id INTEGER PRIMARY KEY,
        is_food INTEGER NOT NULL DEFAULT 1,
        primary_products TEXT NOT NULL DEFAULT '[]',
        specific_varieties TEXT NOT NULL DEFAULT '[]',
        processing TEXT NOT NULL DEFAULT '[]',
        dish_ingredients TEXT NOT NULL DEFAULT '[]',
        search_keywords TEXT,
        raw_json TEXT,
        enriched_at REAL NOT NULL
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_enriched_food ON enriched_catalog(is_food);")

    # Таблица B2B условий
    con.execute("""
    CREATE TABLE IF NOT EXISTS supplier_extra (
        supplier_id INTEGER PRIMARY KEY,
        min_order TEXT,
        price_range TEXT,
        certificates TEXT,
        delivery_terms TEXT,
        region_scope TEXT,
        enriched_at REAL NOT NULL
    );
    """)
    con.commit()


PROMPT_TEMPLATE = """Ты ведущий эксперт по закупкам и классификации в пищевой B2B-индустрии.
Проанализируй список компаний и для КАЖДОЙ извлеки:
1. Классификацию и поисковые сущности
2. Оценку B2B-условий (MOQ, доставка, сертификаты)

Компании:
{items_text}

Верни строго JSON-массив объектов следующей структуры:
[
  {{
    "id": ID компании (число),
    "is_food": true/false (производит ли продукты питания / сельхозпродукцию, а не тару, оборудование, мыло),
    "primary_products": [основные виды продукции в им. падеже, напр. ["сахар", "патока"] или ["сыр", "масло"]],
    "specific_varieties": [сорта, виды: ["сахар-песок", "рафинад"] или ["моцарелла", "сулугуни"]],
    "processing": [способы обработки: "рафинированный", "пастеризованный", "замороженный", "копченый", "сушеный"],
    "dish_ingredients": [ингредиенты блюд, НЕ являющиеся товаром производителя: напр. "сыр" в куриных наггетсах],
    "min_order": "минимальный объем заказа (напр. от 100 кг / от 1 паллеты / от 50 000 руб)",
    "price_range": "ценовой ориентир (напр. оптовый прайс от завода / 50-70 руб/кг / по запросу)",
    "certificates": "сертификаты и системы (ГОСТ, ХАССП, Декларация ЕАС, ФГИС Меркурий)",
    "delivery_terms": "условия логистики (ТК по РФ, самовывоз со склада, рефрижератор)",
    "region_scope": "география поставок (Вся Россия / Поволжье / ЦФО / Экспорт)"
  }}
]
Ответ: ТОЛЬКО JSON-массив без markdown-разметки:"""


def clean_json_str(text: str) -> str:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return m.group(0) if m else text.strip()


def call_llm_with_retry(client, prompt: str, model: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            raw_res = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )
            completion = raw_res.parse()
            headers = raw_res.headers
            rem = headers.get("x-ratelimit-remaining-tokens")
            if rem and rem.isdigit() and int(rem) < 1200:
                reset_val = headers.get("x-ratelimit-reset-tokens", "4s")
                m = re.search(r"([\d\.]+)s", reset_val)
                reset_s = float(m.group(1)) + 1.5 if m else 5.0
                print(f"  [TPM почти исчерпан: остаток {rem}] Пауза {reset_s:.1f}с...")
                time.sleep(reset_s)
            return completion.choices[0].message.content or ""
        except RateLimitError as e:
            wait_time = 45.0
            msg = str(e)
            m = re.search(r"try again in ([\d\.]+)s", msg, re.I)
            if m:
                wait_time = float(m.group(1)) + 4.0
            print(f"  [429 TPM RateLimit] Ожидание {wait_time:.1f}с до сброса лимита Groq...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"  [API Error] {e}. Повтор через 6с...")
            time.sleep(6)
        except Exception as e:
            print(f"  [Ошибка] {e}. Повтор через 5с...")
            time.sleep(5)
    return None


def build_search_keywords(item_data: dict) -> str:
    parts = []
    for p in item_data.get("primary_products", []):
        parts.append(str(p))
    for v in item_data.get("specific_varieties", []):
        parts.append(str(v))
    for pr in item_data.get("processing", []):
        parts.append(str(pr))
    return " ".join(parts).lower()


def process_batch(client, con: sqlite3.Connection, batch: list, model: str) -> int:
    items_text = ""
    for r in batch:
        desc = (r["description"] or "")[:150].replace("\n", " ")
        sub = r["subcategory"] or "—"
        city = r["city"] or "РФ"
        items_text += f"ID: {r['id']} | Компания: {r['name']} | Город: {city} | Подкатегория: {sub} | Описание: {desc}\n"

    prompt = PROMPT_TEMPLATE.format(items_text=items_text)
    raw_resp = call_llm_with_retry(client, prompt, model)
    if not raw_resp:
        return 0

    try:
        data = json.loads(clean_json_str(raw_resp))
    except Exception as e:
        print(f"  ! Ошибка парсинга JSON: {e}")
        return 0

    saved = 0
    now = time.time()
    by_id = {int(item["id"]): item for item in data if isinstance(item, dict) and "id" in item}

    for r in batch:
        sid = r["id"]
        parsed = by_id.get(sid, {})
        if not parsed:
            continue

        # 1. Запись в enriched_catalog (поисковые сущности)
        is_food = 1 if parsed.get("is_food", True) else 0
        primary_products = json.dumps(parsed.get("primary_products", []), ensure_ascii=False)
        specific_varieties = json.dumps(parsed.get("specific_varieties", []), ensure_ascii=False)
        processing = json.dumps(parsed.get("processing", []), ensure_ascii=False)
        dish_ingredients = json.dumps(parsed.get("dish_ingredients", []), ensure_ascii=False)
        keywords = build_search_keywords(parsed)

        con.execute("""
        INSERT OR REPLACE INTO enriched_catalog 
        (supplier_id, is_food, primary_products, specific_varieties, processing, dish_ingredients, search_keywords, raw_json, enriched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, is_food, primary_products, specific_varieties, processing, dish_ingredients, keywords, json.dumps(parsed, ensure_ascii=False), now))

        # 2. Запись в supplier_extra (B2B условия)
        db.save_extra(con, sid, {
            "min_order": parsed.get("min_order") or "По запросу (от 1 коробки / партии)",
            "price_range": parsed.get("price_range") or "Оптовый прайс завода по запросу",
            "certificates": parsed.get("certificates") or "ГОСТ, Декларация ЕАС",
            "delivery_terms": parsed.get("delivery_terms") or "ТК по РФ, самовывоз",
            "region_scope": parsed.get("region_scope") or "Вся Россия",
        })
        saved += 1

    con.commit()
    return saved


def main():
    client = get_client()
    model = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")
    con = sqlite3.connect(HERE / "suppliers.db")
    con.row_factory = sqlite3.Row
    init_db(con)

    total_suppliers = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    already_done = con.execute("SELECT COUNT(*) FROM enriched_catalog").fetchone()[0]

    print(f"========================================================")
    print(f"ЗАПУСК ОБЪЕДИНЕННОГО ОБОГАЩЕНИЯ КАТАЛОГА (СУЩНОСТИ + B2B)")
    print(f"Модель:                   {model}")
    print(f"Всего поставщиков в базе: {total_suppliers}")
    print(f"Уже имеют сущности:       {already_done}")
    print(f"========================================================")

    # Ищем компании, у которых ещё нет разметки сущностей ИЛИ нет B2B-условий
    todo_rows = con.execute("""
    SELECT s.id, s.name, s.city, s.subcategory, s.description 
    FROM suppliers s
    WHERE s.id NOT IN (SELECT supplier_id FROM enriched_catalog)
       OR s.id NOT IN (SELECT supplier_id FROM supplier_extra)
    ORDER BY s.id
    """).fetchall()

    if not todo_rows:
        print("Все поставщики уже полностью обогащены и размечены!")
        return

    print(f"Осталось обработать:      {len(todo_rows)} компаний\n")

    done_in_session = 0
    start_time = time.time()

    for i in range(0, len(todo_rows), BATCH_SIZE):
        batch = todo_rows[i:i + BATCH_SIZE]
        saved = process_batch(client, con, batch, model)
        done_in_session += saved
        
        current_total = already_done + done_in_session
        pct = (current_total / total_suppliers) * 100
        elapsed = time.time() - start_time
        speed = done_in_session / (elapsed or 1.0)
        remaining = (len(todo_rows) - done_in_session) / (speed or 1.0)
        
        print(f"[{current_total}/{total_suppliers}] ({pct:.1f}%) Обогащено +{saved} | {speed:.1f} комп/сек | Осталось: ~{int(remaining // 60)}м {int(remaining % 60)}с")
        time.sleep(REQUEST_DELAY)

    print("\n========================================================")
    print(f"ОБОГАЩЕНИЕ ЗАВЕРШЕНО!")
    print(f"Всего в enriched_catalog: {con.execute('SELECT COUNT(*) FROM enriched_catalog').fetchone()[0]}")
    print(f"Всего в supplier_extra:   {con.execute('SELECT COUNT(*) FROM supplier_extra').fetchone()[0]}")
    print("========================================================")


if __name__ == "__main__":
    main()

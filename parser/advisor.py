"""AI-советник по закупкам, модуль LLM-обогащения и генератор детальных B2B-паспортов товаров.

1. B2B-Паспорт товара (Deep Dive Product Card):
   - Условия закупки (минимальная партия MOQ, квант, упаковка, условия оплаты)
   - Сертификация и стандарты (ГОСТ, ТУ, ХАССП, ЕАС, ФГИС «Меркурий», Честный Знак)
   - Логистика и хранение (температурный режим, сроки годности, самовывоз со склада, ТК)
   - Чек-лист переговоров: Green Flags, Red Flags и 3 вопроса менеджеру для первого созвона
2. AI-отбор целевых кандидатов из поисковой выдачи
3. Интеллектуальное сравнение и выработка стратегии закупки (Procurement Copilot)
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
import db

HERE = Path(__file__).parent


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


def get_client() -> OpenAI | None:
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("your_"):
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=key, base_url=base_url if base_url else None)


MODEL = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODELS = [MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b"]


def call_llm(client: OpenAI, prompt: str, max_tokens: int = 800) -> str:
    """Вызывает LLM с автоматическим переключением на резервные модели при RateLimit (429)."""
    for m in FALLBACK_MODELS:
        try:
            budget = max(max_tokens, 1600) if "gpt-oss" in m else max_tokens
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=budget,
            )
            content = res.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception as e:
            print(f"Model {m} failed ({e}). Trying fallback...")
            continue
    return ""


def parse_json_from_llm(raw: str) -> dict | None:
    """Надежный парсер JSON из ответа языковой модели."""
    if not raw:
        return None
    cleaned = raw.strip()
    # Убираем markdown fences ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # Убираем trailing commas перед закрывающими скобками
            fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except Exception:
                pass
    return None


def fetch_full_page_text(url: str) -> str:
    """Загружает страницу поставщика/товара и извлекает полный чистый текст."""
    try:
        import live
        from bs4 import BeautifulSoup
        html = live._fetch(url)
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.decompose()
        main = soup.select_one("div.content") or soup.select_one("div.firm_content") or soup.select_one("div.main") or soup.body
        if main:
            text = main.get_text(" ")
            return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"Error fetching page text: {e}")
    return ""


NOT_SPECIFIED = "не указано"
# чего нет в тексте карточки, того не знаем: придуманные «ГОСТ, от 20 кг, 350 ₽/кг» закупщик примет за факт
DEFAULT_EXTRA = {k: NOT_SPECIFIED for k in ("min_order", "price_range", "certificates", "delivery_terms", "region_scope")}


def is_known(v) -> bool:
    return bool(v) and str(v).strip().lower() not in (NOT_SPECIFIED, "—", "-", "нет данных", "по запросу")


STRICT_RULES = """Правила:
- Бери только то, что прямо написано в тексте. Ничего не оценивай и не додумывай по «типичным» значениям.
- Если параметра в тексте нет — пиши ровно «не указано».
- Не называй ГОСТ, ТУ, сертификаты, цены и объёмы, которых нет в тексте."""


def extract_extra_info(client: OpenAI | None, supplier: dict) -> dict:
    """Извлекает или оценивает B2B-условия поставщика через LLM."""
    if not client:
        return dict(DEFAULT_EXTRA)

    desc = (supplier.get("description") or "")[:1500].replace("\n", " ")
    sub = supplier.get("subcategory") or "Продукты питания"
    name = supplier.get("name") or "Производитель"
    city = supplier.get("city") or "РФ"

    prompt = f"""Ты ведущий эксперт по B2B-закупкам пищевой продукции.
Выпиши из карточки компании оптовые условия, которые в ней указаны.
{STRICT_RULES}

Компания: {name}
Товарная группа: {sub}
Город: {city}
Описание: {desc}

Верни строго JSON-объект:
{{
  "min_order": "минимальный заказ из текста или «не указано»",
  "price_range": "цена из текста или «не указано»",
  "certificates": "документы и сертификаты из текста или «не указано»",
  "delivery_terms": "условия доставки из текста или «не указано»",
  "region_scope": "география поставок из текста или «не указано»"
}}
Только валидный JSON, без markdown:"""

    try:
        data = parse_json_from_llm(call_llm(client, prompt, max_tokens=250))
        if isinstance(data, dict):
            # без min_order поставщик считается необогащённым и LLM вызывается на каждом клике
            return {**DEFAULT_EXTRA, **{k: str(v).strip() for k, v in data.items() if k in DEFAULT_EXTRA and v}}
    except Exception as e:
        print(f"Error enriching {name}: {e}")

    return dict(DEFAULT_EXTRA)


def enrich_suppliers_extra(con, suppliers_list: list[dict], max_workers: int = 4):
    """Обогащает поставщиков, у которых еще нет данных в supplier_extra."""
    client = get_client()
    todo = [s for s in suppliers_list if not s.get("min_order")]
    if not todo:
        return

    def _worker(s):
        try:
            return s, extract_extra_info(client, s)
        except Exception as e:
            print(f"Error enriching {s.get('name')}: {e}")
            return s, dict(DEFAULT_EXTRA)

    try:
        with ThreadPoolExecutor(max_workers=min(len(todo), max_workers)) as executor:
            results = list(executor.map(_worker, todo))

        for s, data in results:
            try:
                db.save_extra(con, s["id"], data)
                s.update(data)
            except Exception as e:
                print(f"Error saving extra for {s.get('name')}: {e}")
    except Exception as e:
        print(f"Error in enrich_suppliers_extra executor: {e}")


def _brand(s: dict) -> str:
    """«Группа компаний «Вамин Татарстан»» и «Производитель … «ВАМИН»» -> «вамин»."""
    name = s.get("name") or ""
    m = re.search(r"[«\"]([^»\"]+)", name)
    words = re.findall(r"[а-яёa-z0-9]+", (m.group(1) if m else name).lower())
    return words[0] if words else ""


def same_company(a: dict, b: dict) -> bool:
    """Две карточки одной компании: общий телефон или почта, либо один бренд в одном городе."""
    phones = lambda s: {re.sub(r"\D", "", p)[-10:] for p in s.get("phones") or []}
    if phones(a) & phones(b) or set(a.get("emails") or []) & set(b.get("emails") or []):
        return True
    return bool(_brand(a)) and _brand(a) == _brand(b) and a.get("city") == b.get("city")


def dedupe(suppliers: list[dict]) -> list[dict]:
    out = []
    for s in suppliers:
        if not any(same_company(s, o) for o in out):
            out.append(s)
    return out


def recommend_best_supplier(suppliers: list[dict], query: str, user_city: str | None) -> dict:
    """Генерирует интеллектуальную рекомендацию и скрипт первого контакта."""
    client = get_client()
    if not suppliers:
        return {}
    suppliers = dedupe(suppliers)  # иначе одна компания становится и основным, и запасным вариантом

    candidates_info = ""
    for i, s in enumerate(suppliers, 1):
        dist_str = f"{s.get('distance_km')} км до г. {user_city}" if s.get("distance_km") is not None else "расстояние не указано"
        phones_str = ", ".join(s.get("phones") or []) or "телефон не указан"
        emails_str = ", ".join(s.get("emails") or []) or "email не указан"
        candidates_info += f"""
{i}. Компания: «{s.get('name')}»
   Город: {s.get('city') or '—'} ({dist_str})
   Специализация: {s.get('subcategory') or '—'}
   Товаров в каталоге: {s.get('products_count') or '—'}
   Контакты: тел: {phones_str}, email: {emails_str}
   Мин. заказ: {s.get('min_order') or NOT_SPECIFIED}
   Цена: {s.get('price_range') or NOT_SPECIFIED}
   Сертификаты: {s.get('certificates') or NOT_SPECIFIED}
   Доставка: {s.get('delivery_terms') or NOT_SPECIFIED}
   Описание: {(s.get('description') or '')[:400]}
"""

    prompt = f"""Ты ведущий эксперт по B2B-закупкам продуктов питания (Procurement Specialist / Food Buyer).
Пользователь ищет: «{query or 'продукты питания оптом'}» в локации: «{user_city or 'любой регион РФ'}».

Для детального сравнения выбраны следующие поставщики:
{candidates_info}

Проанализируй кандидатов по ключевым критериям закупки:
- Релевантность и специализация (прямой производитель нужного товара vs перекупщик/размытый ассортимент);
- Логистика и расстояние (сроки, плечо доставки, риски для сохранности товара);
- Доступность каналов связи (есть ли прямой телефон, мобильный отдела продаж, или только общий email);
- Условия поставок (минимальная партия, риски первой сделки).

Помоги закупщику принять взвешенное решение и подготовь проект первого обращения.
Правила:
- Опирайся только на данные карточек выше. Не приписывай компании товары, документы, ГОСТы, цены и объёмы,
  которых в её карточке нет. Если нужный товар в карточке не назван прямо — так и скажи и сделай это риском.
- Покупатель находится в «{user_city or 'не указан'}»: про логистику и доставку говори относительно этого города.
- Альтернатива — другая компания, не та же под другим названием.
Верни ответ СТРОГО в формате JSON:
{{
  "best_choice": {{
    "name": "Точное название лучшей компании",
    "reason": "Подробное и убедительное обоснование, почему именно она №1 (плюсы логистики, специализации, контактов)"
  }},
  "alternative": {{
    "name": "Название компании №2",
    "reason": "В каком сценарии звонить именно ей (например: если у первой нет нужного объема, для сбивания цены и т.д.)"
  }},
  "risks": [
    "Конкретный риск или нюанс на переговорах 1",
    "Конкретный риск или нюанс на переговорах 2"
  ],
  "outreach_script": "Персонализированный драфт первого сообщения для отправки в WhatsApp / Telegram / Email (с вежливым обращением, указанием конкретного товара, вопросов о минимальной партии, оптовом прайсе, сертификатах/Меркурии)"
}}
Ответ ТОЛЬКО валидный JSON:"""

    try:
        raw = call_llm(client, prompt, max_tokens=900)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print(f"Error calling LLM advisor: {e}")

    top = suppliers[0]
    sec = suppliers[1] if len(suppliers) > 1 else None
    return {
        "best_choice": {
            "name": top["name"],
            "reason": f"Наилучший баланс релевантности и оценки ({top.get('score', 0):.2f}). Находится в г. {top.get('city') or '—'}.",
        },
        "alternative": {
            "name": sec["name"] if sec else "—",
            "reason": f"Резервный поставщик ({sec.get('city') or '—'}) на случай отсутствия объема у лидера." if sec else "—",
        },
        "risks": [
            "Перед оплатой первого счета обязательно запросите уставные документы и подтверждение в ФГИС «Меркурий».",
            "Уточните точную стоимость и температурный режим доставки до вашего склада.",
        ],
        "outreach_script": f"Здравствуйте! Заинтересовала ваша продукция ({query or top.get('subcategory')}). Планируем оптовую закупку. Пришлите, пожалуйста, актуальный оптовый прайс-лист, условия доставки и минимальный объем отгрузки.",
    }


def select_candidates_via_llm(candidates: list[dict], query: str, user_city: str | None) -> dict:
    """Анализирует поисковую выдачу через LLM, отсеивает ложные срабатывания
    и отбирает 2-5 лучших целевых поставщиков для отправки в сравнение.
    """
    client = get_client()
    if not candidates:
        return {"selected_ids": [], "summary": "Список выдачи пуст."}

    if not client or not query:
        top_ids = [c["id"] for c in dedupe(candidates)[:3]]
        return {"selected_ids": top_ids, "summary": "Отобраны топ-3 кандидата с наивысшим баллом ранжирования."}

    candidates_text = ""
    for c in candidates[:15]:
        cid = c["id"]
        name = c.get("name") or "—"
        sub = c.get("subcategory") or "—"
        city = c.get("city") or "РФ"
        desc = (c.get("description") or "")[:120].replace("\n", " ")
        dist = f"{c.get('distance_km')} км" if c.get("distance_km") is not None else "—"
        candidates_text += f"ID: {cid} | Компания: {name} | Город: {city} ({dist}) | Товар: {sub} | Описание: {desc}\n"

    prompt = f"""Ты ведущий эксперт по закупкам продуктов питания (B2B Food Buyer).
Покупатель ввел поисковый запрос: «{query}» (целевая локация: {user_city or 'любой регион РФ'}).

Вот топ кандидатов из поисковой системы (включая возможные ложные совпадения по словам):
{candidates_text}

Твоя задача:
1. Отсеять неподходящие карточки:
   - компании, которые продают готовые ресторанные блюда/пиццу вместо оптового сырья;
   - компании, продающие упаковку, добавки или корма;
   - компании, где совпало только второстепенное слово, но профиль совсем другой;
   - повторные карточки одной и той же компании (берёшь одну).
2. Отобрать от 2 до 5 САМЫХ ПОДХОДЯЩИХ поставщиков/производителей целевого товара для детального сравнения.
   Постарайся включить сильных кандидатов (например, локального производителя поближе к {user_city or 'клиенту'}, крупный завод или узкоспециализированную компанию).

Верни строго JSON:
{{
  "selected_ids": [числовой_id1, числовой_id2, ...],
  "summary": "1-2 предложения, по какому принципу отобраны компании; только факты из списка выше"
}}
Ответ ТОЛЬКО валидный JSON:"""

    try:
        raw = call_llm(client, prompt, max_tokens=400)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            valid_ids = {c["id"] for c in candidates}
            by_id = {c["id"]: c for c in candidates}
            raw_chosen = data.get("selected_ids", [])
            chosen = []
            for cid in raw_chosen:  # модель может вернуть id строкой: "123" -> 123
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    continue
                if cid in valid_ids and cid not in chosen and not any(same_company(by_id[cid], by_id[o]) for o in chosen):
                    chosen.append(cid)
            if not chosen:
                return {
                    "selected_ids": [c["id"] for c in dedupe(candidates)[:3]],
                    "summary": "AI не нашёл в топе выдачи точных совпадений — в сравнение добавлены 3 лучших по рейтингу.",
                }
            # пояснение собираем из карточек, а не из текста модели: она приписывает компаниям чужие товары
            summary = "Отобраны: " + "; ".join(
                f"{by_id[c].get('name') or '—'}" + (f" ({by_id[c]['city']})" if by_id[c].get("city") else "")
                for c in chosen) + "."
            return {"selected_ids": chosen, "summary": summary}
    except Exception as e:
        print(f"Error in select_candidates_via_llm: {e}")

    return {
        "selected_ids": [c["id"] for c in dedupe(candidates)[:3]],
        "summary": "Отобраны 3 лучших поставщика по общему баллу релевантности."
    }


def get_or_create_b2b_passport(con, supplier: dict, force_refresh: bool = False) -> dict:
    """Извлекает из кэша SQLite или генерирует через LLM полный B2B-паспорт товара и компании.
    
    Анализирует описание, товарные позиции и внешнюю страницу для извлечения:
    - Минимального заказа (MOQ), квантов, тары, условий оплаты
    - Стандартов (ГОСТ/ТУ), ХАССП, ЕАС, ФГИС «Меркурий»
    - Логистических требований и температурного режима
    - Экспертной оценки: Green Flags, Red Flags и 3 вопросов менеджеру на первый созвон
    """
    sid = supplier["id"]
    cached = db.get_passport(con, sid)
    if not force_refresh and cached and cached.get("passport"):
        return cached

    client = get_client()

    url = supplier.get("url") or ""
    full_page_text = fetch_full_page_text(url) if url else ""
    base_desc = supplier.get("description") or ""

    # Собираем контекст из всех имеющихся источников
    parts = []
    if full_page_text and len(full_page_text) > 150:
        parts.append(full_page_text[:1800])
    elif base_desc:
        parts.append(base_desc)

    # Добавляем обогащенные позиции ассортимента из базы
    if supplier.get("primary_products"):
        prods = supplier["primary_products"]
        parts.append("Основная продукция: " + (", ".join(prods) if isinstance(prods, list) else str(prods)))
    if supplier.get("specific_varieties"):
        vars_ = supplier["specific_varieties"]
        parts.append("Сорта и виды: " + (", ".join(vars_) if isinstance(vars_, list) else str(vars_)))
    if supplier.get("dish_ingredients"):
        dishes = supplier["dish_ingredients"]
        parts.append("Применение в HoReCa / производстве: " + (", ".join(dishes) if isinstance(dishes, list) else str(dishes)))

    context_text = "\n\n".join(parts).strip()
    if not context_text:
        context_text = f"Производитель пищевой продукции {supplier.get('name', '')}. Специализация: {supplier.get('subcategory', '')}."

    sub = supplier.get("subcategory") or "Продукты питания"
    name = supplier.get("name") or "Производитель"
    city = supplier.get("city") or "РФ"

    default_passport = {
        "commercial_terms": {
            "min_order": supplier.get("min_order") or NOT_SPECIFIED,
            "packaging": NOT_SPECIFIED,
            "price_estimate": supplier.get("price_range") or NOT_SPECIFIED,
            "payment_terms": NOT_SPECIFIED
        },
        "compliance_and_quality": {
            "standards": NOT_SPECIFIED,
            "certifications": supplier.get("certificates") or NOT_SPECIFIED,
            "shelf_life": NOT_SPECIFIED,
            "product_specs": f"Специализация на сайте: {sub}"
        },
        "logistics": {
            "delivery_type": supplier.get("delivery_terms") or NOT_SPECIFIED,
            "storage_temp": NOT_SPECIFIED
        },
        "buyer_verdict": {
            "green_flags": [f"Специализация на сайте каталога: {sub}"],
            "red_flags": ["Условия поставки в карточке не указаны — запросите прайс и документы до заказа"],
            "negotiation_questions": [
                "Каковы минимальный объем первой отгрузки и условия доставки?",
                "Предоставляете ли ветеринарные свидетельства во ФГИС «Меркурий»?",
                "Возможна ли отправка образцов продукции перед основным договором?"
            ]
        }
    }

    if not client:
        db.save_passport(con, sid, default_passport, context_text)
        return {"passport": default_passport, "full_text": context_text}

    prompt = f"""Ты главный эксперт по B2B-закупкам продуктов питания (Senior Food Procurement Auditor).
Проведи глубокий аудит карточки товара и производителя для профессионального байера (закупщика):
Компания: {name}
Товар / Категория: {sub}
Город: {city}
Полный текст описания и спецификации:
{context_text}

Извлеки и структурируй полную B2B-спецификацию. Поля:
- commercial_terms.min_order — минимальный заказ / партия;
- commercial_terms.packaging — фасовка и тара;
- commercial_terms.price_estimate — ориентир цены или условия прайса;
- commercial_terms.payment_terms — условия оплаты;
- compliance_and_quality.standards — ГОСТ / ТУ / ТР ТС;
- compliance_and_quality.certifications — ХАССП, декларации ЕАС, ФГИС «Меркурий», «Честный Знак»;
- compliance_and_quality.shelf_life — срок годности и условия хранения;
- compliance_and_quality.product_specs — сорт, жирность, состав, технические свойства;
- logistics.delivery_type — условия и география доставки;
- logistics.storage_temp — температурный режим перевозки;
- buyer_verdict.green_flags — 2-3 сильные стороны именно этого поставщика;
- buyer_verdict.red_flags — 1-3 риска, которые нужно проверить закупщику;
- buyer_verdict.negotiation_questions — 3 конкретных вопроса менеджеру на первый созвон.

Каждое значение — строка на русском (в buyer_verdict — списки строк) с фактом из текста для этой
компании, а не название поля и не пример формата.
{STRICT_RULES}
- green_flags — только подтверждённые текстом факты; red_flags — в том числе чего не хватает в карточке
  (нет цен, нет документов); вопросы — о том, что в карточке не указано.

Верни ТОЛЬКО валидный JSON такой структуры:
{{
  "commercial_terms": {{"min_order": "", "packaging": "", "price_estimate": "", "payment_terms": ""}},
  "compliance_and_quality": {{"standards": "", "certifications": "", "shelf_life": "", "product_specs": ""}},
  "logistics": {{"delivery_type": "", "storage_temp": ""}},
  "buyer_verdict": {{"green_flags": [], "red_flags": [], "negotiation_questions": []}}
}}"""

    passport_data = None
    try:
        raw = call_llm(client, prompt, max_tokens=1000)
        passport_data = parse_json_from_llm(raw)
    except Exception as e:
        print(f"Error generating passport for {name}: {e}")

    if not passport_data or not isinstance(passport_data, dict):
        # LLM недоступна: не затираем прежний аудит шаблоном и не кэшируем шаблон — при следующем открытии повторим
        if cached and cached.get("passport"):
            return cached
        return {"passport": default_passport, "full_text": context_text}
    else:
        # Убедимся, что все ключи верхнего уровня присутствуют
        for section in ("commercial_terms", "compliance_and_quality", "logistics", "buyer_verdict"):
            if not isinstance(passport_data.get(section), dict):
                passport_data[section] = default_passport[section]
            # пустое поле дало бы «—» в паспорте и пустой min_order, из-за которого сравнение заново дёргает LLM
            for k, v in default_passport[section].items():
                val = passport_data[section].get(k)
                if not val or (isinstance(val, str) and not re.sub(r"[\W_]", "", val)):
                    passport_data[section][k] = v

    # Сохраняем в кэш БД
    db.save_passport(con, sid, passport_data, context_text)

    # Синхронизируем базовые экстра-поля для таблицы сравнения
    comm = passport_data.get("commercial_terms", {})
    compl = passport_data.get("compliance_and_quality", {})
    logi = passport_data.get("logistics", {})
    extra_sync = {
        "min_order": comm.get("min_order"),
        "price_range": comm.get("price_estimate"),
        "certificates": compl.get("certifications"),
        "delivery_terms": logi.get("delivery_type"),
        "region_scope": supplier.get("region_scope") or NOT_SPECIFIED,
    }
    db.save_extra(con, sid, extra_sync)

    return {"passport": passport_data, "full_text": context_text}

"""LLM-реранкер: оценивает пары «запрос — компания» по рубрике и переупорядочивает топ-N выдачи.

Бэкенд: OpenAI API (gpt-4o-mini по умолчанию, задаётся через OPENAI_API_KEY в .env).
Оценки кэшируются в SQLite (llm_cache.db), так что повторный запрос ничего не стоит.
"""
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent


def _load_env():
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


_load_env()

API_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BATCH = 25

RUBRIC = """Ты оцениваешь релевантность поставщиков пищевой продукции поисковому запросу покупателя.
Для каждой компании поставь оценку:
3 — компания делает именно то, что просят (все слова запроса выполнены: вид продукта и уточнение);
2 — делает нужный продукт, но уточнение (сорт, вид, обработка) не подтверждено или не совпадает;
1 — нужное слово встречается, но это другой продукт (корм, приправа «для курицы», упаковка, блюдо с ингредиентом, оборудование);
0 — не относится к запросу.
Судить нужно по названию товара (подкатегория), названию компании и описанию. Не додумывай то, чего в тексте нет.
Ответ — только JSON-массив чисел той же длины и порядка, что список компаний. Без пояснений."""


def _line(i, d):
    desc = (d.get("description") or "").replace("\n", " ")[:160]
    return f"{i + 1}. товар: {d.get('subcategory') or '—'} | компания: {d.get('name')} | {desc}"


def _prompt(query, docs):
    return f"{RUBRIC}\n\nЗапрос: «{query}»\n\nКомпании:\n" + "\n".join(_line(i, d) for i, d in enumerate(docs))


def _call(prompt: str) -> str:
    _load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key == "your_openai_api_key_here":
        raise RuntimeError("Укажите валидный OPENAI_API_KEY в файле parser/.env или в переменных окружения")
    base_url = os.environ.get("OPENAI_BASE_URL")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url if base_url else None)
    res = client.chat.completions.create(
        model=API_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content or ""


def _parse(text: str, n: int):
    m = re.search(r"\[[\d,\s]*\]", text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if len(arr) == n and all(isinstance(x, int) and 0 <= x <= 3 for x in arr) else None
    except Exception:
        return None


def _cache(path=None):
    p = HERE / "data" / "llm_cache.db" if path is None else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE IF NOT EXISTS g (k TEXT PRIMARY KEY, grade INTEGER)")
    return con


def _key(query, d):
    text = f"{API_MODEL}|{query}|{d.get('subcategory')}|{d.get('name')}|{(d.get('description') or '')[:160]}"
    return hashlib.md5(text.encode()).hexdigest()


def grade(query: str, docs: list[dict]) -> list[int | None]:
    """Оценки 0..3 для каждого документа (None — модель не вернула валидный ответ)."""
    cache = _cache()
    keys = [_key(query, d) for d in docs]
    got = dict(cache.execute(
        f"SELECT k, grade FROM g WHERE k IN ({','.join('?' * len(keys))})", keys)) if keys else {}
    todo = [i for i, k in enumerate(keys) if k not in got]
    for s in range(0, len(todo), BATCH):
        part = todo[s:s + BATCH]
        raw = _call(_prompt(query, [docs[i] for i in part]))
        arr = _parse(raw, len(part))
        if arr is None:
            continue
        for i, g in zip(part, arr):
            got[keys[i]] = g
            cache.execute("INSERT OR REPLACE INTO g VALUES (?, ?)", (keys[i], g))
        cache.commit()
    return [got.get(k) for k in keys]


def rerank(query: str, docs: list[dict], base_scores: list[float], top_n=30, weight=0.5):
    """Новые оценки: для топ-N смешиваем базовую оценку и LLM-оценку/3; остальные остаются как есть."""
    order = sorted(range(len(docs)), key=lambda i: -base_scores[i])[:top_n]
    grades = grade(query, [docs[i] for i in order])
    out = list(base_scores)
    for i, g in zip(order, grades):
        if g is not None:
            out[i] = (1 - weight) * base_scores[i] + weight * g / 3
    return out

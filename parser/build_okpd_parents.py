"""Справочник «товар → родительская категория» из ОКПД2 для ранжирования (data/okpd_parents.json).

    python build_okpd_parents.py

1. Обходит пищевые разделы ОКПД2 (01, 03, 10, 11) на classifikators.ru.
2. Для каждого существительного из названий кодов берёт слова из названий групп-предков
   («10.12.10.120 Мясо индеек» → «10.12.10 Мясо птицы охлажденное» → «птица»).
3. Оставляет родителя, только если он частый в специализациях нашей базы, а само слово в ней заметно реже.
4. Названия групп в ОКПД2 — длинные перечни, поэтому оставляем только надёжных родителей (рыба, птица, мясо)
   и отбрасываем технические слова. Массовый тест: подходящих в топ-10 7.23 → 7.88 против ручного справочника,
   а связи вроде «филе → рыба», «горох → рис», «творог → сыр» портили выдачу.
"""
import collections
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import bm25
import db
import search

OUT = Path(__file__).parent / "data" / "okpd_parents.json"
SECTIONS = ["01", "03", "10", "11"]
GENERIC = {"продукция", "продукт", "изделие", "группировка", "услуга", "культура", "часть", "вид", "прочее", "сырье",
           "свежий", "детский", "питание", "том", "число"}
RELIABLE_PARENTS = {"рыба", "птица", "мясо"}
NOT_KIND = {"филе", "фарш", "печень", "сушка", "копчение", "кровь", "субпродукт", "категория", "хозяйство", "отряд",
            "поголовье", "молодь", "малек", "годовик", "сеголетка", "перо", "ингредиент", "обвалка", "консервирование",
            "рыбоводство", "рыболовство", "жемчуг", "молока", "шкварка", "шпик", "лошак", "мул", "лось", "кит"}


def crawl() -> dict[str, str]:
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"

    def page(code):
        for _ in range(3):
            try:
                r = sess.get(f"https://classifikators.ru/okpd/{code}", timeout=20)
                if r.status_code == 200:
                    return code, r.text
            except requests.RequestException:
                pass
            time.sleep(2)
        return code, ""

    names, todo, seen = {}, list(SECTIONS), set()
    while todo:
        batch = [c for c in todo if c not in seen]
        todo = []
        seen.update(batch)
        with ThreadPoolExecutor(4) as ex:
            for code, html in ex.map(page, batch):
                soup = BeautifulSoup(html, "html.parser")
                m = re.search(r"код [\d.]+ — (.+)$", soup.title.get_text(strip=True) if soup.title else "")
                if m:
                    names[code] = m.group(1)
                for a in soup.select('a[href^="/okpd/"]'):
                    c = a["href"].rsplit("/", 1)[1]
                    if c.startswith(code) and len(c) > len(code) and c not in seen:
                        names.setdefault(c, a.get_text(" ", strip=True))
                        todo.append(c)
    return names


def build(names: dict[str, str]) -> dict[str, str]:
    nouns = {k: [l for l, pos, _ in search.content_lemmas(v) if pos == "NOUN" and l not in GENERIC]
             for k, v in names.items()}
    rows = db.all_suppliers(db.connect())
    df = collections.Counter(l for r in rows for l in set(bm25.tokenize_lemmas(r.get("subcategory"))))
    out = {}
    for lemma in {n for ns in nouns.values() for n in ns} - NOT_KIND:
        hits = [k for k, ns in nouns.items() if lemma in ns]
        cnt = collections.Counter()
        for k in hits:
            p = k.rsplit(".", 1)[0] if "." in k else None
            while p and p in nouns and not [n for n in nouns[p] if n != lemma]:
                p = p.rsplit(".", 1)[0] if "." in p else None
            if p in nouns:
                cnt.update(set(nouns[p]) - {lemma})
        scored = [(c / len(hits) * math.log1p(df[n]), n) for n, c in cnt.items()]
        if not scored:
            continue
        score, parent = max(scored)
        if score >= 2.0 and df[parent] >= 20 and df[lemma] * 2 < df[parent] and parent in RELIABLE_PARENTS:
            out[lemma] = parent
    return dict(sorted(out.items()))


if __name__ == "__main__":
    parents = build(crawl())
    OUT.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(parents)} связей → {OUT}")

"""Скрипт масштабного фонового расширения базы поставщиков suppliers.db.

1. Обходит страницы листинга ProductCenter:
   - Основная категория «Продукты питания» (до 85 страниц, ~4000+ компаний).
   - Специализированные категории (Сельское хозяйство, Ингредиенты, Бакалея, Мясо, Молочка, Рыба и др.).
   - Региональные листинги ключевых регионов (Пермский край, Урал, Поволжье).
2. Запускает точечный сбор по базовым сырьевым и оптовым категориям (сахар, мука, масло, крупы, мясо и т.д.).
3. Сохраняет всё напрямую в suppliers.db через db.upsert с дедупликацией по слагу.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import db
import live
import productcenter_parser as pc

PAGE_DELAY = 1.1         # пауза между запросами (сек)

# Лимиты страниц для разных категорий
CATEGORY_LIMITS = {
    "produkty-pitaniia-45": 85,          # генеральная категория продуктов питания
    "sielskoie-khoziaistvo-46": 20,      # агрохолдинги, фермы, зерно
    "bakalieia-208": 15,                 # бакалея (крупы, сахар, макароны)
    "molochnyie-produkty-316": 12,       # молочные заводы, сыроварни
    "miaso-203": 12,                     # мясокомбинаты, птицефабрики
    "konditierskiie-izdieliia-427": 16,  # кондитерские фабрики
    "napitki-201": 15,                   # напитки, соки, вода
    "ryba-morieprodukty-317": 12,        # рыба, морепродукты, икра
    "konsiervy-615": 10,                 # консервы мясные, рыбные, овощные
    "khliebobulochnyie-izdieliia-297": 10,# хлебозаводы, пекарни
    "ovoshchi-773": 10,                  # овощные базы, картофель
    "oriekhi-799": 8,                    # орехи, сухофрукты
    "frukty-204": 8,                     # фрукты, яблоки
    "pishchievyie-ingriediienty-426": 12,# пищевые ингредиенты, добавки
    "sniekovaia-produktsiia-451": 8,     # снеки, сухарики, чипсы
    "sousy-610": 8,                      # соусы, майонез, кетчуп
    "griby-788": 5,                      # грибы свежие, сушеные, заморозка
    "iagody-3951": 5,                    # ягоды свежие, заморозка
    "pchielovodstvo-412": 5,             # мёд, пчелопродукты
    "iaitso-516": 5,                     # яйцо куриное, перепелиное
    "dietskoie-pitaniie-207": 5,         # детское питание
}

# Регионы с акцентом на Урал, Поволжье и Черноземье для локального покрытия
TARGET_REGIONS = [
    ("piermskii-krai-200", "Пермский край"),
    ("svierdlovskaia-obl-209", "Свердловская область"),
    ("tatarstan-riesp-214", "Татарстан"),
    ("bashkortostan-riesp-157", "Башкортостан"),
    ("chieliapinskaia-obl-218", "Челябинская область"),
    ("udmurtskaia-riesp-216", "Удмуртия"),
    ("krasnodarskii-krai-176", "Краснодарский край"),
    ("voroniezhskaia-obl-164", "Воронежская область"),
    ("rostovskaia-obl-203", "Ростовская область"),
    ("altaiskii-krai-155", "Алтайский край"),
]

# Точечные запросы на оптовые и сырьевые позиции B2B
TARGET_QUERIES = [
    # Сахар и бакалея
    "сахар", "сахар песок", "сахарный завод", "сахар оптом",
    "мука", "мука пшеничная", "мукомольный комбинат", "мельница",
    "масло подсолнечное", "растительное масло", "масло сливочное",
    "крупа гречневая", "гречка оптом", "рис оптом", "макароны оптом",
    "соль пищевая", "соль оптом", "дрожжи",
    # Мясо и птица
    "говядина оптом", "свинина оптом", "мясо птицы", "курица оптом",
    "индейка оптом", "мясо индейки", "мясокомбинат", "птицефабрика",
    "колбаса оптом", "колбасные изделия", "полуфабрикаты мясные",
    # Молочка и сыр
    "молоко оптом", "молоко сырое", "молокозавод", "молочный комбинат",
    "сыр оптом", "сыроварня", "сыродельный комбинат", "моцарелла", "сулугуни",
    "творог оптом", "сметана оптом", "сливки оптом", "козье молоко",
    # Рыба
    "рыба оптом", "рыбокомбинат", "форель оптом", "семга оптом",
    "лосось оптом", "минтай оптом", "сельдь соленая", "пресервы рыбные", "икра оптом",
    # Овощи, фрукты, грибы
    "картофель оптом", "овощи оптом", "лук репчатый", "морковь оптом",
    "капуста оптом", "яблоки оптом", "грибы оптом", "ягоды замороженные",
    # Напитки, консервы, кондитерка
    "соки оптом", "вода минеральная", "напитки безалкогольные",
    "консервы мясные", "тушенка гост", "консервы рыбные",
    "кондитерская фабрика", "шоколад оптом", "печенье оптом",
    "мёд оптом", "липовый мед", "яйцо оптом", "яйцо перепелиное",
]


def expand_category_pages(con, slug: str, cat_name: str, max_pages: int) -> int:
    added = 0
    consecutive_empty = 0
    print(f"\n--- Раздел: {cat_name} ({slug}, до {max_pages} стр.) ---")
    
    for page in range(1, max_pages + 1):
        url = f"{pc.BASE}/producers/catalog-{slug}" + (f"/page-{page}" if page > 1 else "")
        html = pc.fetch(url, timeout=20)
        if not html:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print(f"  [{slug}] стр. {page}: 2 пустых ответа подряд, переход к следующей категории")
                break
            continue
            
        items = pc.parse_listing(html, slug)
        if not items:
            print(f"  [{slug}] стр. {page}: карточки не найдены, завершаем раздел")
            break
            
        page_added = 0
        for it in items:
            is_new = db.upsert(con, it)
            if is_new:
                page_added += 1
                
        con.commit()
        added += page_added
        cur_count = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
        print(f"  [{cat_name}] стр. {page}/{max_pages}: получено {len(items)}, новых +{page_added} (всего в базе: {cur_count})")
        time.sleep(PAGE_DELAY)
        
    return added


def expand_regions(con) -> int:
    print("\n=== Сбор производителей по приоритетным регионам ===")
    total_added = 0
    for reg_slug, reg_name in TARGET_REGIONS:
        print(f"\n--- Регион: {reg_name} ({reg_slug}) ---")
        for page in range(1, 10):
            url = f"{pc.BASE}/producers/r-{reg_slug}/catalog-produkty-pitaniia-45" + (f"/page-{page}" if page > 1 else "")
            html = pc.fetch(url, timeout=20)
            if not html:
                break
            items = pc.parse_listing(html, "produkty-pitaniia-45")
            if not items:
                break
            page_added = 0
            for it in items:
                it["region_slug"] = reg_slug
                if db.upsert(con, it):
                    page_added += 1
            con.commit()
            total_added += page_added
            cur_count = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
            print(f"  [{reg_name}] стр. {page}: найдено {len(items)}, новых +{page_added} (всего: {cur_count})")
            time.sleep(PAGE_DELAY)
            if len(items) < 40:
                break
    return total_added


def expand_targeted_queries(con) -> int:
    print("\n=== Точечный сбор по базовым оптовым товарам через поиск ===")
    total_new = 0
    for q in TARGET_QUERIES:
        try:
            added = live.search_live(con, q, pages=3)
            total_new += added
            cur_count = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
            print(f"  Поиск «{q}»: +{added} новых (всего в базе: {cur_count})")
            time.sleep(PAGE_DELAY)
        except Exception as e:
            print(f"  Ошибка при поиске «{q}»: {e}")
    return total_new


def main():
    con = db.connect()
    start_count = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    print(f"========================================================")
    print(f"СТАРТ МАСШТАБНОГО РАСШИРЕНИЯ КАТАЛОГА ПОСТАВЩИКОВ")
    print(f"Текущее количество в suppliers.db: {start_count}")
    print(f"========================================================")
    
    # 1. Сначала точечные поиски по сырью и дефицитам (сахар, мука, масло и т.д.)
    # Это сразу насытит базу конкретными производителями сахара, зерна, мяса!
    added_targets = expand_targeted_queries(con)
    
    # 2. Региональные выборки для плотности в целевых городах (Пермь, Урал, Поволжье)
    added_regions = expand_regions(con)
    
    # 3. Полный обход разделов каталога (начиная с генеральной категории «Продукты питания» на 85 страниц)
    added_cats = 0
    for slug, limit in CATEGORY_LIMITS.items():
        cat_name = pc.CATEGORIES.get(slug, slug)
        added_cats += expand_category_pages(con, slug, cat_name, limit)
        
    final_count = con.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    print(f"\n========================================================")
    print(f"РАСШИРЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print(f"Было в базе:         {start_count}")
    print(f"Добавлено по поиску: +{added_targets}")
    print(f"Добавлено по регионам: +{added_regions}")
    print(f"Добавлено по разделам: +{added_cats}")
    print(f"Всего новых:         +{final_count - start_count}")
    print(f"ИТОГО в suppliers.db: {final_count} поставщиков")
    print(f"========================================================")


if __name__ == "__main__":
    main()

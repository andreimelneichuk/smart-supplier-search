"""Streamlit-интерфейс каталога поставщиков. Запуск: streamlit run parser/app.py"""
import time
from datetime import datetime
import streamlit as st

import db
import live
import productcenter_parser as pc
import geo
import ranking
import advisor

PAGE_SIZE = 10
GOOD_SCORE = 0.8     # «хороший» результат
LIVE_THRESHOLD = 10  # если хороших результатов меньше — добираем с источника

st.set_page_config(page_title="Поиск поставщиков продуктов", page_icon=":material/storefront:", layout="wide")

STATUS_BADGES = {
    "Не связывались": ":gray[:material/radio_button_unchecked:] Не связывались",
    "Отправлен запрос КП": ":orange[:material/send:] Отправили запрос",
    "Ждем прайс": ":blue[:material/hourglass_top:] Ждём цены",
    "Договорились": ":green[:material/check_circle:] Договорились",
    "Отказ": ":red[:material/cancel:] Отказ",
}


@st.cache_resource
def get_con():
    con = db.connect()
    db.init_from_json_if_empty(con)
    return con


con = get_con()
st.session_state.setdefault("selected", set())
st.session_state.setdefault("live_done", set())
st.session_state.setdefault("contacts_tried", set())  # кому уже пытались подтянуть контакты в этой сессии
st.session_state.setdefault("q", "")

STATUS_OPTS = list(STATUS_BADGES)


def set_query(value: str):
    st.session_state["q"] = value


def reset_query():
    st.session_state["q"] = ""
    st.session_state.pop("selection_summary", None)


def toggle_compare(sid: int):
    if st.session_state.get(f"cmp{sid}"):
        st.session_state.selected.add(sid)
    else:
        st.session_state.selected.discard(sid)


def clear_compare():
    st.session_state.selected = set()
    st.session_state.pop("last_advice", None)
    st.session_state.pop("selection_summary", None)


def save_crm(sid: int, status_key: str, note_key: str, prev_note: str, org: str, author: str):
    """Колбэк «Сохранить»: пишет статус/заметку и очищает поле ввода (иначе повторный клик дублирует запись)."""
    note = (st.session_state.get(note_key) or "").strip()
    db.save_note(con, sid, st.session_state[status_key], note or prev_note, org_name=org, author=author)
    st.session_state[note_key] = ""
    # toast из колбэка в диалоге (fragment rerun) ломает страницу — показываем его при отрисовке
    st.session_state["crm_saved"] = True


def show_saved_toast():
    if st.session_state.pop("crm_saved", False):
        st.toast("Сохранено", icon=":material/check:")


def history_line(h: dict, fmt: str) -> str:
    t_str = datetime.fromtimestamp(h["created_at"]).strftime(fmt)
    line = f"*{t_str}* · **{h['author']}**: {STATUS_BADGES.get(h['status'], h['status'])}"
    return line + (f" — *{h['text']}*" if h.get("text") else "")


show_saved_toast()

# --- 1. Боковая панель: Организация, Пользователь и Фильтры ---
with st.sidebar:
    st.markdown("### :material/business: Ваша компания")
    st.caption("Заметки и статусы видны всем сотрудникам вашей компании.")
    
    org_options = [
        "ООО «ФудТрейд»",
        "Сеть пиццерий «Додо»",
        "Холдинг «ВкусВилл»",
        "HoReCa Рестораны",
        "+ Другая компания..."
    ]
    chosen_org = st.selectbox("Компания", org_options, index=0)
    if chosen_org == "+ Другая компания...":
        current_org = st.text_input("Название компании", value="Моя Компания").strip() or "Моя Компания"
    else:
        current_org = chosen_org

    user_options = [
        "Алексей (Старший закупщик)",
        "Мария (Категорийный менеджер)",
        "Иван (Снабжение HoReCa)",
        "+ Другой сотрудник..."
    ]
    chosen_user = st.selectbox("Сотрудник", user_options, index=0)
    if chosen_user == "+ Другой сотрудник...":
        current_author = st.text_input("Ваше имя", value="Менеджер по закупкам").strip() or "Менеджер"
    else:
        current_author = chosen_user

    st.divider()

    # выбор модели ранжирования убран из интерфейса: обычному пользователю он ничего не говорит
    selected_model = "bm25_cross"

    st.header(":material/filter_alt: Фильтры")
    # Получаем поставщиков для текущей организации
    suppliers = db.all_suppliers(con, org_name=current_org)
    cities = sorted({s["city"] for s in suppliers if s["city"]})
    
    # Город
    city_f = st.selectbox("Город", ["— любой —"] + cities, index=0, key="city_filter")
    
    # Категория
    cat_names_list = ["— любая —"] + list(pc.CATEGORIES.values())
    cat_f = st.selectbox("Категория", cat_names_list, index=0)
    
    only_contacts = st.checkbox("Только с телефоном или почтой", value=False)
    only_my_crm = st.checkbox("Только те, с кем уже работаем", value=False,
                              help="Поставщики, у которых уже есть статус или заметка")

    st.caption(f"В базе: **{len(suppliers)}** производителей")


# --- 2. Верхняя шапка и быстрый поиск ---
st.title(":material/storefront: Поиск поставщиков продуктов")
st.caption(f"Найдите производителей, сравните условия и решите, кому написать первым · **{current_org}**, {current_author}")

# Поле поиска
col_search, col_reset = st.columns([5, 1], vertical_alignment="bottom")
q = col_search.text_input("Что ищете?", key="q",
                          placeholder="например: сыр моцарелла оптом, индейка Казань, мёд липовый, грецкий орех")

col_reset.button("Сбросить", icon=":material/close:", use_container_width=True, help="Очистить поиск", on_click=reset_query)

# Быстрые чипы для популярных товаров
st.markdown("**Популярные запросы:**")
chips_col = st.columns(7)
quick_queries = [
    ("Моцарелла", "сыр моцарелла"),
    ("Индейка", "мясо индейки"),
    ("Рыба", "рыба соленая"),
    ("Мёд", "мёд липовый"),
    ("Хлеб", "хлеб ржаной"),
    ("Креветки", "креветки замороженные"),
    ("Орехи", "орехи грецкие"),
]
for col, (label, query_val) in zip(chips_col, quick_queries):
    col.button(label, use_container_width=True, help=f"Искать: {query_val}", on_click=set_query, args=(query_val,))

# Разбор запроса
parsed = ranking.parse_query(q, cities)
if city_f != "— любой —":
    parsed["city"] = city_f


def coords(city, region=None):
    return geo.cached_coords(con, city, region)




RANK_RETRY_SEC = 120  # упавшую модель не дёргаем на каждом клике: иначе каждый rerun ждёт API и снова падает


def run_ranking(rows=None):
    rows = rows if rows is not None else db.all_suppliers(con, org_name=current_org)
    failed = st.session_state.setdefault("rank_failed_at", {})
    # LLM без ключа / лимит API / cross-encoder не скачался — не роняем страницу, а спускаемся до BM25
    for model in dict.fromkeys([selected_model, "bm25_cross", "bm25"]):
        if model != "bm25" and time.time() - failed.get(model, 0) < RANK_RETRY_SEC:
            continue
        try:
            res = ranking.rank(rows, parsed, pc.CATEGORIES, coords, model=model)
        except Exception as e:
            if model == "bm25":
                raise
            print(f"Ranking model {model} failed: {e}")
            failed[model] = time.time()
            continue
        return res


def distance_km(s: dict):
    """Расстояние до города из запроса/фильтра для карточек вне текущей выдачи (сравнение, паспорт)."""
    d = ranking.distance_to(s, parsed["city"], coords(parsed["city"]) if parsed["city"] else None, coords)
    return None if d is None else round(d)


def geocode_missing(rows, limit=5):
    todo = []
    if parsed["city"] and coords(parsed["city"]) is None:
        todo.append((parsed["city"], None))
    todo += sorted({(r["city"], r.get("region")) for r in rows
                    if r["city"] and coords(r["city"], r.get("region")) is None}, key=str)
    for c, reg in todo[:limit]:
        geo.geocode(con, c, reg)
    return bool(todo)


results = run_ranking(suppliers)

# --- Добор с источника, если по запросу мало ---
query_key = " ".join(parsed["terms"]) + "|" + (parsed["city"] or "")
good = sum(1 for r in results if r["score"] > GOOD_SCORE)
if live.is_blocked():
    st.warning("Источник временно ограничил запросы — показываю только сохранённые данные. "
               "Живой поиск вернётся через несколько минут.")
elif parsed["terms"] and good < LIVE_THRESHOLD and query_key not in st.session_state.live_done:
    with st.spinner("Мало сильных совпадений — ищу на отраслевом источнике…"):
        added = live.search_live(con, " ".join(parsed["terms"]), city=parsed["city"])
    st.session_state.live_done.add(query_key)
    if added:
        st.toast(f"Добавлено с источника: {added}")
    results = run_ranking()

if parsed["city"] and results is not None:
    with st.spinner("Считаю расстояния…") if any(
            coords(r["city"], r.get("region")) is None for r in results[:30] if r["city"]) else st.empty():
        if geocode_missing(results[:30]):
            results = run_ranking()


# Фильтры
if only_contacts:
    results = [r for r in results if r["phones"] or r["emails"]]

if only_my_crm:
    results = [r for r in results if r.get("user_status") != "Не связывались" or r.get("user_note")]

if cat_f != "— любая —":
    cat_slug = next((k for k, v in pc.CATEGORIES.items() if v == cat_f), None)
    if cat_slug:
        results = [r for r in results if cat_slug in r.get("categories", []) or (r.get("subcategory") and cat_f.lower() in r["subcategory"].lower())]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_products(url: str, slug: str) -> list[dict]:
    """Ассортимент в БД не хранится (только число товаров) — берём со страницы производителя."""
    goods = live.fetch_products({"url": url, "slug": slug})
    if not goods:
        raise LookupError("no products")  # пустой результат не кэшируем: сайт мог временно не ответить
    return goods


# --- 2.5 Модальный диалог: Полный B2B-Паспорт товара и поставщика ---
@st.dialog("Карточка поставщика", width="large")
def show_b2b_passport_dialog(item_id: int):
    show_saved_toast()
    fresh_suppliers = {s["id"]: s for s in db.all_suppliers(con, org_name=current_org)}
    item = fresh_suppliers.get(item_id)
    if not item:
        st.error("Поставщик не найден в базе данных.")
        return

    top_col, btn_col = st.columns([3.8, 2.2])
    with top_col:
        st.markdown(f"### [{item['name']}]({item['url']})")
        meta_items = [f":material/location_on: **{item['city'] or 'Город не указан'}**"]
        item_dist = distance_km(item)
        if item_dist:
            meta_items.append(f"*(≈ {item_dist} км от г. {parsed['city']})*")
        meta_items.append(f"Товар: **{item['subcategory'] or '—'}**")
        if item.get("products_count"):
            meta_items.append(f"Ассортимент: **{item['products_count']} товаров**")
        st.markdown(" · ".join(meta_items))

    force_key = f"force_ai_{item['id']}"
    with btn_col:
        b_c1, b_c2 = st.columns(2)
        is_sel = item["id"] in st.session_state.selected
        if b_c1.button("Убрать из сравнения" if is_sel else "К сравнению",
                       icon=":material/remove:" if is_sel else ":material/add:",
                       key=f"dlg_cmp_{item['id']}", use_container_width=True):
            if is_sel:
                st.session_state.selected.discard(item["id"])
            else:
                st.session_state.selected.add(item["id"])
            st.rerun()

        if b_c2.button("Обновить анализ", icon=":material/refresh:", key=f"dlg_reai_{item['id']}",
                       use_container_width=True, help="Заново разобрать условия поставщика с помощью нейросети"):
            st.session_state[force_key] = True
            # перерисовываем только окно: полный rerun закрывает диалог, и аудит запускался при следующем открытии
            st.rerun(scope="fragment")

    st.divider()

    is_force = st.session_state.pop(force_key, False)
    with st.spinner("Разбираю условия поставщика…"):
        passport_data = advisor.get_or_create_b2b_passport(con, item, force_refresh=is_force)

    p = passport_data.get("passport", {})
    comm = p.get("commercial_terms", {})
    compl = p.get("compliance_and_quality", {})
    logi = p.get("logistics", {})
    verdict = p.get("buyer_verdict", {})
    full_text = passport_data.get("full_text") or item.get("description") or ""

    def fmt_val(v):
        if isinstance(v, dict):
            parts = []
            for k1, v1 in v.items():
                if isinstance(v1, dict):
                    inner = ", ".join(f"{ik}: {iv}" for ik, iv in v1.items())
                    parts.append(f"**{k1}**: {inner}")
                else:
                    parts.append(f"**{k1}**: {v1}")
            return "\n\n".join(parts)
        if isinstance(v, list):
            return "\n".join(f"- {x}" for x in v)
        return str(v or "—")

    t_comm, t_goods, t_qual, t_logi, t_ai, t_full, t_crm = st.tabs([
        ":material/payments: Условия и оплата",
        f":material/inventory_2: Ассортимент ({item['products_count'] or '—'})",
        ":material/verified: Качество и документы",
        ":material/local_shipping: Доставка",
        ":material/fact_check: Плюсы и риски",
        ":material/description: Описание и контакты",
        ":material/edit_note: Заметки",
    ])

    with t_comm:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Минимальный заказ")
            st.info(fmt_val(comm.get("min_order", "По запросу")))

            st.markdown("##### Цены")
            st.markdown(fmt_val(comm.get("price_estimate", "По запросу")))

        with c2:
            st.markdown("##### Упаковка")
            st.write(fmt_val(comm.get("packaging", "Стандартная оптовая тара")))

            st.markdown("##### Оплата")
            st.write(fmt_val(comm.get("payment_terms", "Безналичный расчет с НДС")))

    with t_qual:
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("##### Стандарты (ГОСТ, ТУ)")
            st.info(fmt_val(compl.get("standards", "ГОСТ / ТУ")))

            st.markdown("##### Сертификаты и документы")
            st.write(fmt_val(compl.get("certifications", "Декларация ЕАС, ХАССП, ФГИС «Меркурий»")))

        with q2:
            st.markdown("##### Срок годности")
            st.write(fmt_val(compl.get("shelf_life", "В соответствии с регламентом")))

            st.markdown("##### Характеристики товара")
            st.write(fmt_val(compl.get("product_specs", f"Категория: {item.get('subcategory', '—')}")))

    with t_goods:
        with st.spinner("Загружаю товары с сайта производителя…"):
            try:
                goods = load_products(item["url"], item.get("slug") or "")
            except Exception as e:
                print(f"Error loading products: {e}")
                goods = []
        if goods:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(goods)[["image", "name", "price", "url"]],
                hide_index=True, use_container_width=True, row_height=60,
                column_config={
                    "image": st.column_config.ImageColumn("Фото", width="small"),
                    "name": st.column_config.TextColumn("Товар", width="large"),
                    "price": st.column_config.TextColumn("Цена"),
                    "url": st.column_config.LinkColumn("Ссылка", display_text="Открыть"),
                },
            )
            if item.get("products_count") and len(goods) < item["products_count"]:
                st.caption(f"Показаны {len(goods)} из {item['products_count']} — остальные на [странице производителя]({item['url']}).")
        else:
            st.info("Не удалось загрузить список товаров. Он есть на [странице производителя]({}).".format(item["url"]),
                    icon=":material/info:")

    with t_logi:
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("##### Как и куда доставляют")
            st.info(fmt_val(logi.get("delivery_type", "ТК по РФ, самовывоз")))
        with l2:
            st.markdown("##### Температура при перевозке")
            st.write(fmt_val(logi.get("storage_temp", "Соответствует категории продукции")))

    with t_ai:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("##### :green[:material/thumb_up:] Плюсы")
            for gf in verdict.get("green_flags", []):
                st.markdown(f"- {gf}")

        with a2:
            st.markdown("##### :red[:material/warning:] Что проверить")
            for rf in verdict.get("red_flags", []):
                st.markdown(f"- {rf}")

        st.markdown("##### Вопросы для первого звонка")
        for q_idx, q_text in enumerate(verdict.get("negotiation_questions", []), 1):
            st.markdown(f"**{q_idx}.** {q_text}")

        st.markdown("##### Первое письмо — можно скопировать")
        first_contact = f"Здравствуйте! Заинтересовала продукция «{item.get('subcategory') or 'продукты питания'}» компании «{item['name']}». Прошу направить актуальный оптовый прайс-лист, условия минимальной партии и доставки до г. {parsed.get('city') or 'нашего склада'}."
        st.code(first_contact, language="text")

    with t_full:
        st.markdown("##### Полное описание")
        st.text_area("Полный текст", label_visibility="collapsed", value=max(full_text, item.get("description") or "", key=len), height=180, disabled=True, key=f"dlg_raw_{item['id']}")

        extra_info_rows = []
        if item.get("primary_products"):
            prods = item["primary_products"]
            extra_info_rows.append(f"**Основные товары:** {', '.join(prods) if isinstance(prods, list) else prods}")
        if item.get("specific_varieties"):
            vars_ = item["specific_varieties"]
            extra_info_rows.append(f"**Сорта и виды:** {', '.join(vars_) if isinstance(vars_, list) else vars_}")
        if item.get("dish_ingredients"):
            dishes = item["dish_ingredients"]
            extra_info_rows.append(f"**Где используют:** {', '.join(dishes) if isinstance(dishes, list) else dishes}")
        if item.get("processing"):
            procs = item["processing"]
            extra_info_rows.append(f"**Обработка:** {', '.join(procs) if isinstance(procs, list) else procs}")

        if extra_info_rows:
            st.markdown("##### Из каталога")
            for eir in extra_info_rows:
                st.markdown(f"- {eir}")

        st.markdown("##### Контакты")
        c_lines = []
        if item.get("phones"):
            c_lines.append(":material/call: " + " · ".join(f"[{p}](tel:{p.replace(' ', '').replace('-', '')})" for p in item["phones"]))
        if item.get("emails"):
            c_lines.append(":material/mail: " + " · ".join(f"[{e}](mailto:{e})" for e in item["emails"]))
        c_lines.append(f":material/open_in_new: [Страница производителя]({item['url']})")
        st.markdown("\n\n".join(c_lines))

    with t_crm:
        st.caption(f"Видно всем сотрудникам «{current_org}». Сохраняете от имени: **{current_author}**")

        crm_status = item.get("user_status", "Не связывались")
        crm_note = item.get("user_note", "")

        c_st, c_nt = st.columns([1, 2])
        idx_cur = STATUS_OPTS.index(crm_status) if crm_status in STATUS_OPTS else 0
        c_st.selectbox("Статус", STATUS_OPTS, index=idx_cur, key=f"dlg_crm_st_{item['id']}")
        c_nt.text_input("Заметка", placeholder="Например: обещали прислать цены до пятницы, менеджер Анна", key=f"dlg_crm_nt_{item['id']}")

        st.button("Сохранить", icon=":material/save:", key=f"dlg_crm_save_{item['id']}", type="primary", on_click=save_crm,
                  args=(item["id"], f"dlg_crm_st_{item['id']}", f"dlg_crm_nt_{item['id']}", crm_note,
                        current_org, current_author))

        hist = db.get_note_history(con, item["id"], org_name=current_org)
        if hist:
            st.markdown("---")
            st.markdown("**История**")
            for h in hist:
                st.caption(history_line(h, "%d.%m.%Y %H:%M"))


# --- 3. Вкладки: Результаты и Сравнение ---
cmp_title = f"Сравнение ({len(st.session_state.selected)})" if st.session_state.selected else "Сравнение"
tab_list, tab_cmp = st.tabs([":material/list: Найденные", f":material/compare_arrows: {cmp_title}"])

with tab_list:
    pages = max(1, -(-len(results) // PAGE_SIZE))
    c1, c_ai = st.columns([3, 3])
    c1.subheader(f"Найдено поставщиков: {len(results)}")
    
    # Кнопка умного отбора LLM
    if results and q:
        if c_ai.button("Подобрать лучших для сравнения", icon=":material/auto_awesome:", type="primary",
                       use_container_width=True,
                       help="Нейросеть просмотрит первые результаты, отсеет лишнее (готовые блюда, добавки) и добавит подходящих производителей в сравнение"):
            with st.spinner("Выбираю подходящих производителей…"):
                sel_res = advisor.select_candidates_via_llm(results[:15], q, parsed.get("city"))
                st.session_state.selected.update(sel_res.get("selected_ids", []))
                st.session_state["selection_summary"] = sel_res.get("summary", "")
                st.toast(f"Добавлено в сравнение: {len(sel_res.get('selected_ids', []))}", icon=":material/auto_awesome:")
                st.rerun()

    # ключ зависит от запроса и фильтров: новая выдача открывается с первой страницы
    page_key = f"page|{query_key}|{selected_model}|{cat_f}|{only_contacts}|{only_my_crm}"
    # переключатель страниц рисуется под списком, а номер нужен уже здесь — берём его из состояния виджета
    page = min(st.session_state.get(page_key, 1), pages)

    if "selection_summary" in st.session_state and st.session_state["selection_summary"]:
        c_inf, c_cl = st.columns([10, 1])
        c_inf.info(f"**Подбор:** {st.session_state['selection_summary']} Откройте вкладку «{cmp_title}».", icon=":material/auto_awesome:")
        if c_cl.button(":material/close:", key="close_summary", help="Скрыть подсказку"):
            st.session_state.pop("selection_summary", None)
            st.rerun()

    chunk = results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    # контакты подгружаем только тем, кто на экране
    # неудачные попытки не повторяем на каждом клике: иначе любое действие ждёт сетевые таймауты
    need = [r for r in chunk if not r.get("contacts_at") and r["id"] not in st.session_state.contacts_tried]
    st.session_state.contacts_tried.update(r["id"] for r in need)
    if need and not live.is_blocked():
        with st.spinner(f"Загружаю контакты ({len(need)})…"):
            try:
                live.enrich_contacts(con, need)
            except Exception as e:
                print(f"Error enriching contacts: {e}")
            by_id = {r["id"]: r for r in db.all_suppliers(con, org_name=current_org)}
            for r in chunk:
                if r["id"] in by_id:
                    fresh = by_id[r["id"]]
                    r.update(phones=fresh["phones"], emails=fresh["emails"], contacts_at=fresh["contacts_at"])

    if not results:
        st.info("По вашему запросу ничего не найдено. Попробуйте изменить формулировку или сбросить фильтры.")

    for r in chunk:
        with st.container(border=True):
            left, right = st.columns([5, 1.2])
            
            # Название и география
            left.markdown(f"### [{r['name']}]({r['url']})")
            meta = f":material/location_on: **{r['city'] or 'Город не указан'}**"
            if parsed["city"] and r.get("distance_km") is not None and r["distance_km"] > 0:
                meta += f" *(≈ {r['distance_km']} км от г. {parsed['city']})*"
            meta += f" · Товар: **{r['subcategory'] or '—'}**"
            if r["products_count"]:
                meta += f" · Ассортимент: **{r['products_count']} товаров**"
            left.markdown(meta)
            
            left.write(r["description"])

            # B2B-параметры (обогащенные данные)
            b2b_badges = []
            if r.get("min_order"):
                b2b_badges.append(f"Мин. заказ: **{r['min_order']}**")
            if r.get("price_range"):
                b2b_badges.append(f"Цена: **{r['price_range']}**")
            if r.get("certificates"):
                b2b_badges.append(f"Документы: **{r['certificates']}**")
            if r.get("delivery_terms"):
                b2b_badges.append(f"Доставка: **{r['delivery_terms']}**")
            if b2b_badges:
                left.info(" · ".join(b2b_badges), icon=":material/info:")

            # Контакты
            if r["phones"] or r["emails"]:
                contact_line = ""
                if r["phones"]:
                    contact_line += ":material/call: " + " · ".join(f"[{p}](tel:{p.replace(' ', '').replace('-', '')})" for p in r["phones"])
                if r["emails"]:
                    contact_line += (" · " if contact_line else "") + ":material/mail: " + " · ".join(f"[{e}](mailto:{e})" for e in r["emails"])
                left.markdown(contact_line)
            else:
                left.caption("Контакты на сайте не указаны")

            # Статус в CRM и заметки текущей организации
            crm_status = r.get("user_status", "Не связывались")
            crm_note = r.get("user_note", "")
            badge_icon = STATUS_BADGES.get(crm_status, crm_status)
            
            note_display = f"Статус: **{badge_icon}**"
            if crm_note:
                author_txt = f" *(автор: {r.get('user_note_author', 'Менеджер')})*"
                note_display += f" · *«{crm_note}»*{author_txt}"
            left.caption(note_display)

            # Правая колонка: Оценка и действия
            right.metric("Совпадение", f"{r['score']:.0%}", help="Насколько поставщик подходит под запрос, с учётом расстояния и контактов")

            if right.button("Подробнее", icon=":material/open_in_full:", key=f"passport_btn_{r['id']}",
                            use_container_width=True, help="Условия, ассортимент, контакты и заметки"):
                show_b2b_passport_dialog(r["id"])

            # Поповер заметок с историей
            with right.popover("Заметки", icon=":material/edit_note:"):
                st.caption(r['name'][:40])
                idx_cur = STATUS_OPTS.index(crm_status) if crm_status in STATUS_OPTS else 0
                st.selectbox("Статус", STATUS_OPTS, index=idx_cur, key=f"c_st_{r['id']}")
                st.text_input("Заметка", placeholder="Договорённость, имя менеджера, цены…", key=f"c_nt_{r['id']}")
                
                st.button("Сохранить", icon=":material/save:", key=f"c_btn_{r['id']}", on_click=save_crm,
                          args=(r["id"], f"c_st_{r['id']}", f"c_nt_{r['id']}", crm_note, current_org, current_author))

                # История изменений
                history = db.get_note_history(con, r["id"], org_name=current_org)
                if history:
                    st.markdown("---")
                    st.markdown("**История**")
                    for h in history[:3]:
                        st.caption(history_line(h, "%d.%m %H:%M"))

            # состояние галочки берём из списка сравнения: его меняют и AI-отбор, и «Очистить список»
            st.session_state[f"cmp{r['id']}"] = r["id"] in st.session_state.selected
            right.checkbox("Сравнить", key=f"cmp{r['id']}", on_change=toggle_compare, args=(r["id"],))

    if pages > 1:
        st.number_input(f"Страница (всего {pages})", 1, pages, 1, key=page_key, width=200)


# --- 4. Вкладка Сравнение (Decision Copilot) ---
with tab_cmp:
    all_by_id = {s["id"]: s for s in db.all_suppliers(con, org_name=current_org)}
    sel = [all_by_id[sid] for sid in st.session_state.selected if sid in all_by_id]
    
    if not sel:
        st.info("Отметьте «Сравнить» у 2–4 поставщиков на вкладке «Найденные» или нажмите «Подобрать лучших для сравнения».", icon=":material/lightbulb:")
    else:
        # Автоматическое B2B-обогащение через LLM
        missing_extra = [s for s in sel if not s.get("min_order")]
        if missing_extra:
            with st.spinner("Уточняю условия поставщиков: заказ, цены, доставка, документы…"):
                try:
                    advisor.enrich_suppliers_extra(con, missing_extra)
                except Exception as e:
                    print(f"Error enriching extra: {e}")
                all_by_id = {s["id"]: s for s in db.all_suppliers(con, org_name=current_org)}
                sel = [all_by_id[s["id"]] for s in sel if s["id"] in all_by_id]

        # в БД нет расстояния и оценки — они зависят от запроса; добираем их из выдачи или считаем
        ranked = {r["id"]: r for r in results}
        sel = [{**s, "distance_km": distance_km(s), "score": ranked.get(s["id"], {}).get("score", 0)} for s in sel]
        sel.sort(key=lambda s: -s["score"])

        # Верхняя плашка метрик
        c_cnt, c_clr = st.columns([4, 1])
        c_cnt.subheader(f"Сравнение поставщиков ({len(sel)})")
        c_clr.button("Очистить", icon=":material/delete:", use_container_width=True, on_click=clear_compare)

        import pandas as pd
        table_data = []
        for s in sel:
            dist_val = f"{s.get('distance_km')} км" if s.get("distance_km") is not None else "—"
            table_data.append({
                "Поставщик": s["name"],
                "Город": s["city"] or "—",
                "Расстояние": dist_val,
                "Товар": s["subcategory"] or "—",
                "Мин. заказ": s.get("min_order") or "По запросу",
                "Цена": s.get("price_range") or "Оптовый прайс по запросу",
                "Сертификаты": s.get("certificates") or "Декларация ЕАС, ГОСТ",
                "Доставка": s.get("delivery_terms") or "ТК по РФ, самовывоз",
                "Куда доставляют": s.get("region_scope") or "Вся Россия",
                "Товаров": str(s["products_count"] or "—"),
                "Телефоны": ", ".join(s["phones"]) or "—",
                "Email": ", ".join(s["emails"]) or "—",
                "Статус": s.get("user_status") or "Не связывались",
                "Заметка": s.get("user_note", ""),
                "Ссылка": s["url"],
            })
        st.dataframe(pd.DataFrame(table_data).set_index("Поставщик").T, use_container_width=True)

        st.markdown("---")
        st.subheader(":material/auto_awesome: Кого выбрать")
        st.caption("Нейросеть сравнит условия, расстояние и контакты, подскажет, с кого начать, и напишет первое сообщение.")

        col_btn, _ = st.columns([2, 3])
        run_ai = col_btn.button("Получить совет", icon=":material/auto_awesome:", type="primary", use_container_width=True)

        if run_ai:
            with st.spinner("Сравниваю поставщиков…"):
                advice = advisor.recommend_best_supplier(sel, q, parsed.get("city"))
                st.session_state["last_advice"] = advice
                st.session_state["last_advice_for"] = sorted(st.session_state.selected)

        if "last_advice" in st.session_state and st.session_state["last_advice"]:
            adv = st.session_state["last_advice"]
            if st.session_state.get("last_advice_for") != sorted(st.session_state.selected):
                st.warning("Список поставщиков изменился — нажмите «Получить совет» ещё раз.", icon=":material/sync_problem:")
            cb, ca = st.columns(2)
            with cb:
                st.success(f"**Лучший вариант: {adv.get('best_choice', {}).get('name', '—')}**", icon=":material/trophy:")
                st.write(adv.get("best_choice", {}).get("reason", ""))
            with ca:
                st.info(f"**Запасной вариант: {adv.get('alternative', {}).get('name', '—')}**", icon=":material/swap_horiz:")
                st.write(adv.get("alternative", {}).get("reason", ""))

            st.markdown("##### На что обратить внимание")
            for rk in adv.get("risks", []):
                st.markdown(f"- {rk}")

            st.markdown("##### Готовое сообщение поставщику")
            script_text = adv.get("outreach_script", "")
            # ключ от текста: иначе виджет держит первый вариант и новая рекомендация не отображается
            st.text_area("Можно скопировать и отправить в мессенджер или по почте:", value=script_text, height=130,
                         key=f"outreach_msg_{hash(script_text)}")

        st.markdown("---")
        st.subheader(":material/edit_note: Статусы и заметки")
        for s in sel:
            with st.expander(f"{s['name']} — {STATUS_BADGES.get(s.get('user_status'), s.get('user_status', 'Не связывались'))}"):
                c_st, c_nt, c_save, c_pass = st.columns([2, 3.5, 1, 1.5])
                cur_st = s.get("user_status", "Не связывались")
                idx_st = STATUS_OPTS.index(cur_st) if cur_st in STATUS_OPTS else 0
                c_st.selectbox("Статус", STATUS_OPTS, index=idx_st, key=f"crm_st_{s['id']}")
                c_nt.text_input("Заметка", placeholder="Договорённости, скидки, номер счёта…", key=f"crm_nt_{s['id']}")
                
                c_save.button("Сохранить", key=f"crm_btn_{s['id']}", on_click=save_crm,
                              args=(s["id"], f"crm_st_{s['id']}", f"crm_nt_{s['id']}", s.get("user_note", ""),
                                    current_org, current_author))

                if c_pass.button("Подробнее", icon=":material/open_in_full:", key=f"crm_pass_btn_{s['id']}",
                                 use_container_width=True, help="Условия, ассортимент, контакты и заметки"):
                    show_b2b_passport_dialog(s["id"])

                hist = db.get_note_history(con, s["id"], org_name=current_org)
                if hist:
                    st.markdown("**История**")
                    for h in hist:
                        st.caption(history_line(h, "%d.%m.%Y %H:%M"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from functools import lru_cache
from typing import Dict, Optional

SOURCES = [
    "Organic",
    "Paid Search",
    "Social Media",
    "Email",
    "Marketplaces",
]

LANDINGS = [
    "Homepage",
    "Sale",
    "New Arrivals",
    "Lookbook",
    "Blog",
]

PRODUCT_CATALOG = [
    {"name": "Velocity Sneakers", "category": "Shoes", "marketplace_base": 0.14},
    {"name": "Aurora Midi Dress", "category": "Apparel", "marketplace_base": 0.19},
    {"name": "Luminous Jacket", "category": "Outerwear", "marketplace_base": 0.11},
    {"name": "Eclipse T-Shirt", "category": "Apparel", "marketplace_base": 0.08},
    {"name": "Zenith Tote", "category": "Accessories", "marketplace_base": 0.17},
    {"name": "Pulse Leggings", "category": "Sportswear", "marketplace_base": 0.12},
    {"name": "Noir Sunglasses", "category": "Accessories", "marketplace_base": 0.22},
    {"name": "Atlas Backpack", "category": "Accessories", "marketplace_base": 0.10},
    {"name": "Silk Skyline Scarf", "category": "Accessories", "marketplace_base": 0.20},
    {"name": "Horizon Denim Jacket", "category": "Apparel", "marketplace_base": 0.09},
    {"name": "Nimbus Raincoat", "category": "Outerwear", "marketplace_base": 0.16},
]


def _allocate_integer(total: int, weights: np.ndarray) -> np.ndarray:
    """Distribute an integer total across weights while preserving the sum."""
    if total <= 0:
        return np.zeros_like(weights, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    normalized = weights / weights.sum()
    raw = np.floor(normalized * total).astype(int)
    remainder = int(total - raw.sum())
    if remainder > 0:
        fractions = normalized * total - raw
        for idx in np.argsort(fractions)[::-1][:remainder]:
            raw[idx] += 1
    elif remainder < 0:
        fractions = normalized * total - raw
        for idx in np.argsort(fractions)[:abs(remainder)]:
            if raw[idx] > 0:
                raw[idx] -= 1
    return raw


def _generate_datasets() -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")

    base_rows = []
    source_rows = []
    landing_rows = []
    product_rows = []

    for date in dates:
        visits = max(int(rng.normal(2600, 420)), 1500)
        users = max(int(visits * rng.uniform(0.68, 0.85)), 1)
        page_views = max(int(visits * rng.uniform(3.8, 5.2)), visits)
        catalog_views = max(int(visits * rng.uniform(0.78, 0.92)), 1)
        product_views = max(int(catalog_views * rng.uniform(0.55, 0.72)), 1)
        add_to_cart = max(int(product_views * rng.uniform(0.22, 0.32)), 1)
        cart_opens = max(int(add_to_cart * rng.uniform(0.82, 0.90)), 1)
        checkout_start = max(int(cart_opens * rng.uniform(0.75, 0.88)), 1)
        pay_clicks = max(int(checkout_start * rng.uniform(0.82, 0.94)), 1)
        orders = max(int(pay_clicks * rng.uniform(0.90, 0.98)), 1)
        avg_items = rng.uniform(1.25, 1.75)
        items = max(int(round(orders * avg_items)), orders)
        avg_item_price = rng.uniform(2800, 4300)
        revenue = float(items * avg_item_price)

        base_rows.append(
            {
                "date": date,
                "visits": visits,
                "users": users,
                "page_views": page_views,
                "orders": orders,
                "revenue": revenue,
                "items": items,
                "avg_items_per_order": items / orders if orders else 0,
                "avg_item_price": revenue / items if items else 0,
                "avg_views_per_visit": page_views / visits if visits else 0,
                "catalog_views": catalog_views,
                "product_views": product_views,
                "add_to_cart": add_to_cart,
                "cart_opens": cart_opens,
                "checkout_start": checkout_start,
                "pay_clicks": pay_clicks,
            }
        )

        source_visit_weights = rng.dirichlet(np.ones(len(SOURCES)))
        source_order_weights = rng.dirichlet(np.ones(len(SOURCES)))
        visit_alloc = _allocate_integer(visits, source_visit_weights)
        order_alloc = _allocate_integer(orders, source_order_weights)
        for idx, source in enumerate(SOURCES):
            visits_source = int(visit_alloc[idx])
            orders_source = int(order_alloc[idx])
            conversion = (orders_source / visits_source * 100) if visits_source else 0
            source_rows.append(
                {
                    "date": date,
                    "source": source,
                    "visits": visits_source,
                    "orders": orders_source,
                    "conversion_pct": conversion,
                }
            )

        landing_visit_weights = rng.dirichlet(np.ones(len(LANDINGS)))
        landing_order_weights = rng.dirichlet(np.ones(len(LANDINGS)))
        landing_visits_alloc = _allocate_integer(visits, landing_visit_weights)
        landing_orders_alloc = _allocate_integer(orders, landing_order_weights)
        for idx, landing in enumerate(LANDINGS):
            visits_landing = int(landing_visits_alloc[idx])
            orders_landing = int(landing_orders_alloc[idx])
            conversion = (orders_landing / visits_landing * 100) if visits_landing else 0
            landing_rows.append(
                {
                    "date": date,
                    "landing": landing,
                    "visits": visits_landing,
                    "orders": orders_landing,
                    "conversion_pct": conversion,
                }
            )

        product_view_weights = rng.dirichlet(np.ones(len(PRODUCT_CATALOG)))
        product_cart_weights = rng.dirichlet(np.ones(len(PRODUCT_CATALOG)))
        product_order_weights = rng.dirichlet(np.ones(len(PRODUCT_CATALOG)))
        product_views_alloc = _allocate_integer(product_views, product_view_weights)
        product_cart_alloc = _allocate_integer(add_to_cart, product_cart_weights)
        product_orders_alloc = _allocate_integer(orders, product_order_weights)
        for idx, product in enumerate(PRODUCT_CATALOG):
            marketplace_share = float(
                np.clip(rng.normal(product["marketplace_base"], 0.015), 0.03, 0.35)
            )
            product_rows.append(
                {
                    "date": date,
                    "product": product["name"],
                    "category": product["category"],
                    "views": int(product_views_alloc[idx]),
                    "add_to_cart": int(product_cart_alloc[idx]),
                    "orders": int(product_orders_alloc[idx]),
                    "marketplace_share": marketplace_share,
                }
            )

    base_df = pd.DataFrame(base_rows)
    source_df = pd.DataFrame(source_rows)
    landing_df = pd.DataFrame(landing_rows)
    product_df = pd.DataFrame(product_rows)

    return {
        "base": base_df,
        "sources": source_df,
        "landings": landing_df,
        "products": product_df,
    }


@lru_cache(maxsize=1)
def load_data() -> Dict[str, pd.DataFrame]:
    return _generate_datasets()


def filter_by_dates(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, date_col: str = "date"
) -> pd.DataFrame:
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[mask].copy()


def format_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def format_currency(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "orders": 0,
            "revenue": 0.0,
            "items": 0,
            "visits": 0,
            "users": 0,
            "views": 0,
            "avg_order_value": 0.0,
            "avg_items_per_order": 0.0,
            "avg_item_price": 0.0,
            "overall_conversion": 0.0,
            "avg_views_per_visit": 0.0,
        }

    orders = int(df["orders"].sum())
    revenue = float(df["revenue"].sum())
    items = int(df["items"].sum())
    visits = int(df["visits"].sum())
    users = int(df["users"].sum())
    views = int(df["page_views"].sum())

    avg_order_value = revenue / orders if orders else 0.0
    avg_items_per_order = items / orders if orders else 0.0
    avg_item_price = revenue / items if items else 0.0
    overall_conversion = orders / visits * 100 if visits else 0.0
    avg_views_per_visit = views / visits if visits else 0.0

    return {
        "orders": orders,
        "revenue": revenue,
        "items": items,
        "visits": visits,
        "users": users,
        "views": views,
        "avg_order_value": avg_order_value,
        "avg_items_per_order": avg_items_per_order,
        "avg_item_price": avg_item_price,
        "overall_conversion": overall_conversion,
        "avg_views_per_visit": avg_views_per_visit,
    }


def calc_delta_pct(current: float, previous: float) -> Optional[str]:
    if previous is None:
        return None
    current_value = float(current)
    previous_value = float(previous)

    if not np.isfinite(current_value) or not np.isfinite(previous_value):
        return None

    if np.isclose(previous_value, 0.0):
        if np.isclose(current_value, 0.0):
            return "0.0%"
        return None

    delta_pct = (current_value - previous_value) / previous_value * 100
    return f"{delta_pct:+.1f}%"


def main() -> None:
    st.set_page_config(page_title="Fashion Retail Dashboard", layout="wide")
    st.title("Дашборд fashion-ретейлера")
    st.caption("Синтетические данные за январь-март 2023 года")

    data = load_data()
    base_df = data["base"]

    min_date = base_df["date"].min().date()
    max_date = base_df["date"].max().date()
    default_start = max_date - pd.Timedelta(days=29)

    with st.sidebar:
        st.header("Фильтры")
        date_selection = st.date_input(
            "Период",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_selection, tuple):
            if len(date_selection) == 2:
                start_date, end_date = date_selection
            elif len(date_selection) == 1:
                start_date = date_selection[0]
                end_date = date_selection[0]
            else:
                start_date = min_date
                end_date = max_date
        else:
            start_date = date_selection
            end_date = date_selection

        if start_date > end_date:
            st.error("Дата начала больше даты окончания. Выберите корректный период.")
            st.stop()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered = filter_by_dates(base_df, start_date, end_date)
    if filtered.empty:
        st.warning("За выбранный период данных нет.")
        st.stop()

    period_days = (end_date - start_date).days + 1
    previous_end = start_date - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=period_days - 1)
    previous_slice = filter_by_dates(base_df, previous_start, previous_end)

    current_metrics = aggregate_metrics(filtered)
    previous_metrics = aggregate_metrics(previous_slice)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Количество заказов",
        format_int(current_metrics["orders"]),
        calc_delta_pct(current_metrics["orders"], previous_metrics["orders"]),
    )
    col2.metric(
        "Оборот, руб.",
        format_currency(current_metrics["revenue"]),
        calc_delta_pct(current_metrics["revenue"], previous_metrics["revenue"]),
    )
    col3.metric(
        "Средний чек, руб.",
        format_currency(current_metrics["avg_order_value"]),
        calc_delta_pct(
            current_metrics["avg_order_value"], previous_metrics["avg_order_value"]
        ),
    )
    col4.metric(
        "Среднее количество товаров в заказе",
        f"{current_metrics['avg_items_per_order']:.2f}",
        calc_delta_pct(
            current_metrics["avg_items_per_order"],
            previous_metrics["avg_items_per_order"],
        ),
    )

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "Средняя стоимость 1 товара, руб.",
        format_currency(current_metrics["avg_item_price"]),
        calc_delta_pct(
            current_metrics["avg_item_price"], previous_metrics["avg_item_price"]
        ),
    )
    col6.metric(
        "Количество посещений",
        format_int(current_metrics["visits"]),
        calc_delta_pct(current_metrics["visits"], previous_metrics["visits"]),
    )
    col7.metric(
        "Количество пользователей",
        format_int(current_metrics["users"]),
        calc_delta_pct(current_metrics["users"], previous_metrics["users"]),
    )
    col8.metric(
        "Конверсия в заказ общая, %",
        f"{current_metrics['overall_conversion']:.2f}",
        calc_delta_pct(
            current_metrics["overall_conversion"],
            previous_metrics["overall_conversion"],
        ),
    )

    orders_tab, traffic_tab, conversion_tab, funnel_tab, products_tab = st.tabs(
        [
            "Метрики заказов",
            "Трафик",
            "Конверсии",
            "Воронка оформления заказа",
            "Товарная аналитика",
        ]
    )

    with orders_tab:
        st.subheader("Динамика заказов и оборота")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            fig_orders = px.line(
                filtered,
                x="date",
                y="orders",
                markers=True,
                title="Количество заказов по дням",
                labels={"date": "Дата", "orders": "Заказы"},
            )
            fig_orders.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_orders, use_container_width=True)
        with col_o2:
            fig_revenue = px.line(
                filtered,
                x="date",
                y="revenue",
                markers=True,
                title="Оборот по дням, руб.",
                labels={"date": "Дата", "revenue": "Оборот, руб."},
            )
            fig_revenue.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_revenue, use_container_width=True)

        detailed = filtered.copy()
        detailed["average_check"] = detailed["revenue"] / detailed["orders"]
        detailed["average_items"] = detailed["items"] / detailed["orders"]
        detailed["average_item_price"] = detailed["revenue"] / detailed["items"]
        detailed_table = detailed[
            [
                "date",
                "orders",
                "revenue",
                "average_check",
                "items",
                "average_items",
                "average_item_price",
            ]
        ].rename(
            columns={
                "date": "Дата",
                "orders": "Заказы",
                "revenue": "Оборот, руб.",
                "average_check": "Средний чек, руб.",
                "items": "Товары, шт.",
                "average_items": "Среднее количество товаров",
                "average_item_price": "Средняя стоимость товара, руб.",
            }
        )
        detailed_table["Дата"] = detailed_table["Дата"].dt.date
        st.dataframe(
            detailed_table.style.format(
                {
                    "Заказы": format_int,
                    "Оборот, руб.": format_int,
                    "Средний чек, руб.": format_int,
                    "Товары, шт.": format_int,
                    "Среднее количество товаров": "{:.2f}".format,
                    "Средняя стоимость товара, руб.": format_int,
                }
            ),
            use_container_width=True,
        )

    with traffic_tab:
        st.subheader("Трафик и вовлеченность")
        traffic_trend = filtered.melt(
            id_vars="date",
            value_vars=["visits", "users", "page_views"],
            var_name="metric",
            value_name="value",
        )
        metric_labels = {
            "visits": "Посещения",
            "users": "Пользователи",
            "page_views": "Просмотры",
        }
        traffic_trend["metric"] = traffic_trend["metric"].map(metric_labels)
        fig_traffic = px.line(
            traffic_trend,
            x="date",
            y="value",
            color="metric",
            markers=True,
            title="Динамика трафика",
            labels={"metric": "Метрика", "date": "Дата", "value": "Значение"},
        )
        fig_traffic.update_layout(margin=dict(l=10, r=10, t=60, b=10), legend_title="")
        st.plotly_chart(fig_traffic, use_container_width=True)

        col_t1, col_t2, col_t3 = st.columns(3)
        avg_views_per_visit = current_metrics["avg_views_per_visit"]
        col_t1.metric(
            "Среднее количество просмотров на посещение",
            f"{avg_views_per_visit:.2f}",
            calc_delta_pct(
                current_metrics["avg_views_per_visit"],
                previous_metrics["avg_views_per_visit"],
            ),
        )
        col_t2.metric("Всего просмотров", format_int(current_metrics["views"]))
        user_visit_ratio = (
            current_metrics["users"] / current_metrics["visits"] * 100
            if current_metrics["visits"]
            else 0
        )
        col_t3.metric("Пользователи/посещения, %", f"{user_visit_ratio:.1f}")

    with conversion_tab:
        st.subheader("Конверсии по срезам")
        sources_df = filter_by_dates(data["sources"], start_date, end_date)
        sources_summary = (
            sources_df.groupby("source", as_index=False)[["visits", "orders"]]
            .sum()
        )
        sources_summary["conversion_pct"] = np.where(
            sources_summary["visits"] > 0,
            sources_summary["orders"] / sources_summary["visits"] * 100,
            0,
        )
        fig_sources = px.bar(
            sources_summary,
            x="source",
            y="conversion_pct",
            text_auto=".1f",
            title="Конверсия в заказ по источникам, %",
            labels={"source": "Источник", "conversion_pct": "Конверсия, %"},
        )
        fig_sources.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_sources, use_container_width=True)
        st.dataframe(
            sources_summary.rename(
                columns={
                    "source": "Источник",
                    "visits": "Посещения",
                    "orders": "Заказы",
                    "conversion_pct": "Конверсия, %",
                }
            ).style.format(
                {
                    "Посещения": format_int,
                    "Заказы": format_int,
                    "Конверсия, %": "{:.2f}".format,
                }
            ),
            use_container_width=True,
        )

        landings_df = filter_by_dates(data["landings"], start_date, end_date)
        landings_summary = (
            landings_df.groupby("landing", as_index=False)[["visits", "orders"]]
            .sum()
        )
        landings_summary["conversion_pct"] = np.where(
            landings_summary["visits"] > 0,
            landings_summary["orders"] / landings_summary["visits"] * 100,
            0,
        )
        fig_landings = px.bar(
            landings_summary,
            x="landing",
            y="conversion_pct",
            text_auto=".1f",
            title="Конверсия в заказ по страницам входа, %",
            labels={"landing": "Страница входа", "conversion_pct": "Конверсия, %"},
        )
        fig_landings.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_landings, use_container_width=True)
        st.dataframe(
            landings_summary.rename(
                columns={
                    "landing": "Страница входа",
                    "visits": "Посещения",
                    "orders": "Заказы",
                    "conversion_pct": "Конверсия, %",
                }
            ).style.format(
                {
                    "Посещения": format_int,
                    "Заказы": format_int,
                    "Конверсия, %": "{:.2f}".format,
                }
            ),
            use_container_width=True,
        )

    with funnel_tab:
        st.subheader("Воронка оформления заказа")
        funnel_steps = [
            ("Общее количество посещений сайта", filtered["visits"].sum()),
            ("Каталог", filtered["catalog_views"].sum()),
            ("Карточка товара", filtered["product_views"].sum()),
            ("Добавление в корзину", filtered["add_to_cart"].sum()),
            ("Открытие корзины", filtered["cart_opens"].sum()),
            ("Начало оформления заказа", filtered["checkout_start"].sum()),
            ("Кнопка \"Оплатить заказ\"", filtered["pay_clicks"].sum()),
            ("Успешное оформление заказа", filtered["orders"].sum()),
        ]
        step_names = [step[0] for step in funnel_steps]
        step_values = [step[1] for step in funnel_steps]

        funnel_fig = go.Figure(
            go.Funnel(
                y=step_names,
                x=step_values,
                textposition="inside",
                texttemplate="%{value:,.0f}",
                hovertemplate="%{label}<br>%{value:,.0f}<extra></extra>",
            )
        )
        funnel_fig.update_layout(margin=dict(l=60, r=60, t=60, b=20))
        st.plotly_chart(funnel_fig, use_container_width=True)

        transitions = []
        for idx, value in enumerate(step_values):
            if idx == len(step_values) - 1:
                transitions.append(np.nan)
            else:
                next_value = step_values[idx + 1]
                transition_pct = (next_value / value * 100) if value else 0
                transitions.append(transition_pct)
        funnel_table = pd.DataFrame(
            {
                "Шаг": step_names,
                "Значение": step_values,
                "Переход на следующий шаг, %": transitions,
            }
        )
        st.dataframe(
            funnel_table.style.format(
                {
                    "Значение": format_int,
                    "Переход на следующий шаг, %": "{:.2f}".format,
                }
            ),
            use_container_width=True,
        )

    with products_tab:
        st.subheader("Товарная аналитика")
        product_df = filter_by_dates(data["products"], start_date, end_date)
        product_summary = (
            product_df.groupby(["product", "category"], as_index=False)
            .agg(
                {
                    "views": "sum",
                    "add_to_cart": "sum",
                    "orders": "sum",
                    "marketplace_share": "mean",
                }
            )
        )
        product_summary["add_to_cart_pct"] = np.where(
            product_summary["views"] > 0,
            product_summary["add_to_cart"] / product_summary["views"] * 100,
            0,
        )
        product_summary["marketplace_pct"] = product_summary["marketplace_share"] * 100

        top_views = product_summary.nlargest(10, "views")[
            ["product", "category", "views", "orders"]
        ]
        st.markdown("**Топ товаров по просмотрам**")
        st.dataframe(
            top_views.rename(
                columns={
                    "product": "Товар",
                    "category": "Категория",
                    "views": "Просмотры",
                    "orders": "Заказы",
                }
            ).style.format(
                {
                    "Просмотры": format_int,
                    "Заказы": format_int,
                }
            ),
            use_container_width=True,
        )

        top_add_to_cart = product_summary.nlargest(10, "add_to_cart_pct")[
            ["product", "category", "add_to_cart_pct", "add_to_cart"]
        ]
        st.markdown("**Топ товаров по % добавления в корзину**")
        st.dataframe(
            top_add_to_cart.rename(
                columns={
                    "product": "Товар",
                    "category": "Категория",
                    "add_to_cart_pct": "Добавления в корзину, %",
                    "add_to_cart": "Добавления, шт.",
                }
            ).style.format(
                {
                    "Добавления в корзину, %": "{:.2f}".format,
                    "Добавления, шт.": format_int,
                }
            ),
            use_container_width=True,
        )

        top_orders = product_summary.nlargest(10, "orders")[
            ["product", "category", "orders", "views"]
        ]
        st.markdown("**Топ товаров по оформленным заказам**")
        st.dataframe(
            top_orders.rename(
                columns={
                    "product": "Товар",
                    "category": "Категория",
                    "orders": "Заказы",
                    "views": "Просмотры",
                }
            ).style.format(
                {
                    "Заказы": format_int,
                    "Просмотры": format_int,
                }
            ),
            use_container_width=True,
        )

        top_marketplace = product_summary.nlargest(10, "marketplace_pct")[
            ["product", "category", "marketplace_pct", "orders"]
        ]
        st.markdown("**Доля переходов с карточки на маркетплейсы (Ozon/WB)**")
        st.dataframe(
            top_marketplace.rename(
                columns={
                    "product": "Товар",
                    "category": "Категория",
                    "marketplace_pct": "Доля переходов, %",
                    "orders": "Заказы",
                }
            ).style.format(
                {
                    "Доля переходов, %": "{:.2f}".format,
                    "Заказы": format_int,
                }
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

import io
import socket
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Analýza prodejů 2025", layout="wide")

# ---------- Pomocné funkce ----------
def to_numeric_cz(s: pd.Series) -> pd.Series:
    """Bezpečný převod českých čísel na float (mezery + čárky)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", "."),
        errors="coerce"
    )

@st.cache_data
def load_data(path: str | None, uploaded: bytes | None) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(io.BytesIO(uploaded), encoding="cp1250", sep=";")
    else:
        df = pd.read_csv(path, encoding="cp1250", sep=";")

    for col in ["name", "manufacturer", "supplier", "defaultCategory", "code", "internalNote"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["price", "purchasePrice", "turnover", "margins", "count", "stockAmount", "ean"]:
        if col in df.columns:
            df[col] = to_numeric_cz(df[col])

    for col in ["manufacturer", "supplier", "defaultCategory"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    price = df["price"].fillna(0)
    purchase = df["purchasePrice"].fillna(0)
    count = df["count"].fillna(0)

    df["total_profit"] = (price - purchase) * count
    df["margin_percent"] = np.where(price > 0, (price - purchase) / price * 100, np.nan)

    for col in ["turnover", "margins", "count", "total_profit"]:
        df[col] = df[col].fillna(0)

    return df

METRIC_LABELS = {
    "turnover": "Obrat (Kč)",
    "total_profit": "Zisk (Kč)",
    "margins": "Marže (Kč)",
    "count": "Prodáno (ks)",
    "margin_percent": "Maržovost (%)"
}

# ---------- Postranní panel ----------
st.sidebar.title("⚙️ Nastavení")

uploaded_file = st.sidebar.file_uploader(
    "Nahraj CSV (pokud nechceš použít prodej2025.csv v repu)",
    type=["csv"]
)

default_path = "prodej2025.csv"
use_repo_file = st.sidebar.checkbox("Použít soubor z repozitáře", value=True)

try:
    df = load_data(default_path if use_repo_file else None,
                   uploaded_file.getvalue() if (not use_repo_file and uploaded_file is not None) else None)
except Exception as e:
    st.error(f"Načtení se nepovedlo: {e}")
    st.stop()

# Filtry
cat_sel = st.sidebar.multiselect("Kategorie", sorted(df["defaultCategory"].unique()))
sup_sel = st.sidebar.multiselect("Dodavatel", sorted(df["supplier"].unique()))
man_sel = st.sidebar.multiselect("Výrobce", sorted(df["manufacturer"].unique()))

metric = st.sidebar.selectbox(
    "Metrika",
    ["turnover", "total_profit", "margins", "count", "margin_percent"],
    format_func=lambda m: METRIC_LABELS.get(m, m)
)

top_n = st.sidebar.slider("Počet položek v žebříčku", 10, 50, 30, step=5)

# ---------- Filtrace ----------
f = df.copy()
if cat_sel: f = f[f["defaultCategory"].isin(cat_sel)]
if sup_sel: f = f[f["supplier"].isin(sup_sel)]
if man_sel: f = f[f["manufacturer"].isin(man_sel)]

# ---------- Záhlaví ----------
st.title("📊 Interaktivní analýza prodejů 2025 (Streamlit)")
left, right = st.columns([2, 1])
with left:
    st.caption("Klikni v legendě pro skrývání/ukazování, kolečkem zoomuj. Hodnoty v tooltipu.")
with right:
    st.metric("Počet záznamů po filtraci", len(f))

# ---------- TOP produkty ----------
tmp = f.copy()
tmp["__metric__"] = tmp["margin_percent"].fillna(-np.inf) if metric == "margin_percent" else tmp[metric].fillna(0)
top = tmp.sort_values("__metric__", ascending=False).head(top_n)

fig_top = px.bar(
    top,
    x="name",
    y="__metric__",
    color="defaultCategory",
    hover_data={
        "price": ":.2f",
        "purchasePrice": ":.2f",
        "count": True,
        "turnover": True,
        "margins": True,
        "total_profit": True,
        "margin_percent": ":.1f",
        "defaultCategory": True,
        "supplier": True,
        "manufacturer": True,
        "name": False,
        "__metric__": False
    },
    title=f"TOP {top_n} produktů podle: {METRIC_LABELS[metric]}"
)
fig_top.update_layout(
    xaxis_title="Produkt",
    yaxis_title=METRIC_LABELS[metric],
    xaxis_tickangle=-45,
    template="plotly_white",
    legend_title="Kategorie",
    hovermode="x unified",
    height=650
)

st.plotly_chart(fig_top, use_container_width=True)

# ---------- Souhrn dle kategorií ----------
agg = f.groupby("defaultCategory", as_index=False).agg(
    turnover=("turnover", "sum"),
    total_profit=("total_profit", "sum"),
    margins=("margins", "sum"),
    count=("count", "sum")
)
agg["margin_percent"] = np.where(
    agg["turnover"] > 0, (agg["margins"] / agg["turnover"]) * 100, np.nan
)

fig_cat = px.bar(
    agg.sort_values(metric, ascending=False),
    x="defaultCategory",
    y=metric,
    color="defaultCategory",
    title=f"{METRIC_LABELS[metric]} podle kategorií"
)
fig_cat.update_layout(
    xaxis_tickangle=-45,
    template="plotly_white",
    showlegend=False,
    height=500
)
st.plotly_chart(fig_cat, use_container_width=True)

# ---------- Tabulka + export ----------
st.subheader("📄 Data po filtraci")
st.dataframe(
    f[["name","defaultCategory","supplier","manufacturer","price","purchasePrice","count","turnover","margins","total_profit","margin_percent"]]
)

# Export CSV
csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Stáhnout CSV (po filtraci)", data=csv_bytes, file_name="filtered.csv", mime="text/csv")

# Export Excel
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    f.to_excel(writer, index=False, sheet_name="Filtered")
st.download_button("⬇️ Stáhnout Excel (po filtraci)", data=excel_buffer.getvalue(),
                   file_name="filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Info o přístupu
st.caption("Tip: Na Streamlit Cloud dej soubor `prodej2025.csv` do repozitáře nebo používej nahrání souboru vlevo v panelu.")

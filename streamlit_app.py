import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# ZÁKLADNÍ NASTAVENÍ
# =========================
st.set_page_config(page_title="Interaktivní analýza prodejů 2025", layout="wide")

# ---------- 🔒 OCHRANA HESLEM (pevně v kódu) ----------
APP_PASSWORD = "analyza1234"

def check_password() -> bool:
    """Vrátí True, pokud je uživatel ověřen. Heslo je v proměnné APP_PASSWORD."""
    if st.session_state.get("authed", False):
        return True

    with st.sidebar:
        st.header("🔒 Přihlášení")
        pwd = st.text_input("Heslo", type="password")
        ok = st.button("Přihlásit")

    if ok:
        if pwd == APP_PASSWORD:
            st.session_state["authed"] = True
            return True
        else:
            st.error("Nesprávné heslo.")
            return False

    st.stop()

if not check_password():
    st.stop()

# ---------- POMOCNÉ FUNKCE ----------
def to_numeric_cz(s: pd.Series) -> pd.Series:
    """Bezpečný převod českých čísel na float (mezery + čárky)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str)
         .str.replace("\u00A0", "", regex=False)  # nezlomitelné mezery
         .str.replace(" ", "", regex=False)
         .str.replace(",", "."),
        errors="coerce"
    )

@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1250", sep=";")
    return prepare_df(df)

@st.cache_data
def load_data_from_bytes(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b), encoding="cp1250", sep=";")
    return prepare_df(df)

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # texty
    for col in ["name", "manufacturer", "supplier", "defaultCategory", "code", "internalNote"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # čísla
    for col in ["price", "purchasePrice", "turnover", "margins", "count", "stockAmount", "ean"]:
        if col in df.columns:
            df[col] = to_numeric_cz(df[col])

    # doplnění prázdných
    for col in ["manufacturer", "supplier", "defaultCategory"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # odvozené metriky
    price = df["price"].fillna(0) if "price" in df.columns else 0
    purchase = df["purchasePrice"].fillna(0) if "purchasePrice" in df.columns else 0
    count = df["count"].fillna(0) if "count" in df.columns else 0

    if isinstance(price, pd.Series) and isinstance(purchase, pd.Series) and isinstance(count, pd.Series):
        df["total_profit"] = (price - purchase) * count
        df["margin_percent"] = np.where(price > 0, (price - purchase) / price * 100, np.nan)
    else:
        df["total_profit"] = 0
        df["margin_percent"] = np.nan

    for col in ["turnover", "margins", "count", "total_profit"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

METRIC_LABELS = {
    "turnover": "Obrat (Kč)",
    "total_profit": "Zisk (Kč)",
    "margins": "Marže (Kč)",
    "count": "Prodáno (ks)",
    "margin_percent": "Maržovost (%)"
}

# =========================
# UI – SIDEBAR + DATA
# =========================
st.sidebar.title("⚙️ Nastavení dat")
DEFAULT_CSV = "prodeje2025.csv"  # soubor v repozitáři (když zaškrtneš volbu)

use_repo_file = st.sidebar.checkbox(f"Použít soubor z repozitáře: {DEFAULT_CSV}", value=False)
uploaded = st.sidebar.file_uploader("Nebo nahraj CSV ručně", type=["csv"])

# volitelně vyloučit položky obsahující "DÝŠKO" (defaultně NE – necháváme je)
exclude_dysko = st.sidebar.checkbox("Vyloučit položky obsahující 'DÝŠKO'", value=False)

# Načtení dat
if use_repo_file and uploaded is None:
    try:
        df = load_data_from_path(DEFAULT_CSV)
    except Exception as e:
        st.error(f"Nepovedlo se načíst '{DEFAULT_CSV}': {e}")
        st.stop()
else:
    if uploaded is None:
        st.info("Nahraj CSV vlevo, anebo zaškrtni použití souboru z repozitáře.")
        st.stop()
    df = load_data_from_bytes(uploaded.getvalue())

if exclude_dysko and "name" in df.columns:
    df = df[~df["name"].str.contains("DÝŠKO", case=False, na=False)]

# Filtry
cat_sel = st.sidebar.multiselect("Kategorie", sorted(df["defaultCategory"].unique()))
sup_sel = st.sidebar.multiselect("Dodavatel", sorted(df["supplier"].unique()))
man_sel = st.sidebar.multiselect("Výrobce", sorted(df["manufacturer"].unique()))
metric = st.sidebar.selectbox(
    "Metrika",
    ["turnover", "total_profit", "margins", "count", "margin_percent"],
    format_func=lambda m: METRIC_LABELS[m],
    index=0
)
top_n = st.sidebar.slider("Počet položek v žebříčku", 10, 50, 30, step=5)

# =========================
# FILTRACE + HLAVIČKA
# =========================
f = df.copy()
if cat_sel: f = f[f["defaultCategory"].isin(cat_sel)]
if sup_sel: f = f[f["supplier"].isin(sup_sel)]
if man_sel: f = f[f["manufacturer"].isin(man_sel)]

st.title("📊 Interaktivní analýza prodejů 2025 (Streamlit)")
st.caption("Klikni v legendě pro skrývání/ukazování, kolečkem zoomuj, hodnoty v tooltippu.")

# KPI
k1, k2, k3 = st.columns(3)
k1.metric("Počet záznamů po filtraci", int(len(f)))
k2.metric("Součet obratu (Kč)", int(f["turnover"].sum()) if "turnover" in f.columns else 0)
k3.metric("Součet zisku (Kč)", int(f["total_profit"].sum()) if "total_profit" in f.columns else 0)

# =========================
# TOP PRODUKTY
# =========================
tmp = f.copy()
if metric == "margin_percent":
    tmp["__metric__"] = tmp["margin_percent"].fillna(-np.inf)
else:
    tmp["__metric__"] = tmp[metric].fillna(0)

top = tmp.sort_values("__metric__", ascending=False).head(top_n)

fig_top = px.bar(
    top,
    x="name",
    y="__metric__",
    color="defaultCategory",
    hover_data={
        "price": ":.2f" if "price" in top.columns else True,
        "purchasePrice": ":.2f" if "purchasePrice" in top.columns else True,
        "count": True,
        "turnover": True,
        "margins": True,
        "total_profit": True,
        "margin_percent": ":.1f",
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

# =========================
# SOUHRN DLE KATEGORIÍ
# =========================
agg = f.groupby("defaultCategory", as_index=False).agg(
    turnover=("turnover", "sum") if "turnover" in f.columns else ("name", "count"),
    total_profit=("total_profit", "sum") if "total_profit" in f.columns else ("name", "count"),
    margins=("margins", "sum") if "margins" in f.columns else ("name", "count"),
    count=("count", "sum") if "count" in f.columns else ("name", "count")
)
if "turnover" in agg.columns and "margins" in agg.columns:
    agg["margin_percent"] = np.where(
        agg["turnover"] > 0, (agg["margins"] / agg["turnover"]) * 100, np.nan
    )
else:
    agg["margin_percent"] = np.nan

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

# =========================
# TABULKA + EXPORT
# =========================
st.subheader("📄 Data po filtraci")
show_cols = [
    "name","defaultCategory","supplier","manufacturer","price","purchasePrice",
    "count","turnover","margins","total_profit","margin_percent"
]
show_cols = [c for c in show_cols if c in f.columns]
st.dataframe(f[show_cols], use_container_width=True)

# Exporty
csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Stáhnout CSV (po filtraci)",
                   data=csv_bytes, file_name="filtered.csv", mime="text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    f[show_cols].to_excel(writer, index=False, sheet_name="Filtered")
st.download_button("⬇️ Stáhnout Excel (po filtraci)",
                   data=excel_buffer.getvalue(),
                   file_name="filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

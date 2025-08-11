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

# ---------- OCHRANA HESLEM ----------
def check_password() -> bool:
    """Vrátí True, pokud je uživatel ověřen. Heslo bere z st.secrets['APP_PASSWORD'] nebo z env APP_PASSWORD."""
    # už dřív úspěšně přihlášen?
    if st.session_state.get("authed", False):
        return True

    # zjistit očekávané heslo (secrets má přednost)
    expected = st.secrets.get("APP_PASSWORD", None)
    if expected is None:
        expected = os.environ.get("APP_PASSWORD", None)

    # pokud není nastaveno žádné heslo, aplikace je volně přístupná
    if not expected:
        st.warning("⚠️ Heslo není nastaveno (APP_PASSWORD). Aplikace je momentálně veřejná.")
        return True

    # login UI v sidebaru
    with st.sidebar:
        st.header("🔒 Přihlášení")
        pwd = st.text_input("Heslo", type="password")
        ok = st.button("Přihlásit")

    if ok:
        if pwd == expected:
            st.session_state["authed"] = True
            return True
        else:
            st.error("Nesprávné heslo.")
            return False

    # Pozastavit vykreslování, dokud se nepřihlásí
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

# =========================
# UI – SIDEBAR + DATA
# =========================
st.sidebar.title("⚙️ Nastavení dat")
default_csv = "prodeje2025.csv"  # název souboru v repozitáři (nahraješ si ho tam sám)

use_repo_file = st.sidebar.checkbox(f"Použít soubor z repozitáře: {default_csv}", value=False)
uploaded = st.sidebar.file_uploader("Nebo nahraj CSV ručně", type=["csv"])

# volitelně vyloučit položky obsahující "DÝŠKO" (defaultně ponecháváme)
exclude_dysko = st.sidebar.checkbox("Vyloučit položky obsahující 'DÝŠKO'", value=False)

# Načtení dat
if use_repo_file and uploaded is None:
    try:
        df = load_data_from_path(default_csv)
    except Exception as e:
        st.error(f"Nepovedlo se načíst '{default_csv}': {e}")
        st.stop()
else:
    if uploaded is None:
        st.info("Nahraj CSV vlevo, anebo zaškrtni použití souboru z repozitáře.")
        st.stop()
    df = load_data_from_bytes(uploaded.getvalue())

if exclude_dysko:
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
k1.metric("Počet záznamů po filtraci", len(f))
k2.metric("Součet obratu (Kč)", f["turnover"].sum())
k3.metric("Součet zisku (Kč)", int(f["total_profit"].sum()))

# =========================
# TOP PRODUKTY
# =========================
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

# =========================
# TABULKA + EXPORT
# =========================
st.subheader("📄 Data po filtraci")
show_cols = ["name","defaultCategory","supplier","manufacturer","price","purchasePrice",
             "count","turnover","margins","total_profit","margin_percent"]
show_cols = [c for c in show_cols if c in f.columns]
st.dataframe(f[show_cols])

csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Stáhnout CSV (po filtraci)",
                   data=csv_bytes, file_name="filtered.csv", mime="text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    f.to_excel(writer, index=False, sheet_name="Filtered")
st.download_button("⬇️ Stáhnout Excel (po filtraci)",
                   data=excel_buffer.getvalue(),
                   file_name="filtered.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

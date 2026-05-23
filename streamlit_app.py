"""
Stock Screener — Streamlit web frontend.

Reads daily CSV outputs from the data/ directory (committed to the repo after
each GitHub Actions run) and displays them with filtering, charting, and a
manual trigger button.

Streamlit secrets required for the trigger button:
  GH_TOKEN       — GitHub personal access token (scope: workflow)
  GITHUB_OWNER   — GitHub username / org that owns this repo
  GITHUB_REPO    — repo name (default: stock-screener)
"""

import glob
import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

INSIDER_COLORS = {
    "BUYING":  "#198754",
    "SELLING": "#dc3545",
    "NEUTRAL": "#6c757d",
    "—":       "#adb5bd",
}

VALUE_COLS = [
    "Ticker", "Company", "Price", "% Off 52W High", "% Below SMA",
    "P/B Ratio", "Div Yield", "ROE",
    "Short % Float", "Short Ratio (Days)",
    "Insider Signal", "Insider Net Value (6mo)", "Insider Buys", "Insider Sells",
    "Market Cap", "Sector", "Exchange",
]

MOM_COLS = [
    "Ticker", "Company", "Price", "% Above 20Q SMA",
    "20Q SMA", "30Q SMA", "50Q SMA",
    "P/B Ratio", "Div Yield", "ROE",
    "Short % Float", "Short Ratio (Days)",
    "Insider Signal", "Insider Net Value (6mo)", "Insider Buys", "Insider Sells",
    "Market Cap", "Sector", "Exchange",
]

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ───────────────────────────────────────────────────────────────

def available_dates() -> list[str]:
    """Return dates (newest first) for which at least one screen CSV exists."""
    files = glob.glob(os.path.join(DATA_DIR, "screen_*_below30Q.csv"))
    dates = []
    for f in files:
        m = re.search(r"screen_(\d{4}-\d{2}-\d{2})_", os.path.basename(f))
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@st.cache_data(ttl=300)
def load_csv(date_str: str, screen_type: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"screen_{date_str}_{screen_type}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str).fillna("—")
    return df


@st.cache_data(ttl=300)
def load_reports(date_str: str) -> dict[str, str]:
    """Return {ticker: report_text} for all reports on the given date."""
    paths = glob.glob(os.path.join(DATA_DIR, f"report_{date_str}_*.txt"))
    result: dict[str, str] = {}
    for path in sorted(paths):
        m = re.search(r"report_\d{4}-\d{2}-\d{2}_(.+)\.txt$", os.path.basename(path))
        if m:
            with open(path, encoding="utf-8") as f:
                result[m.group(1)] = f.read()
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_float(s) -> float:
    try:
        return float(str(s).replace("%", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def _style_insider(val: str) -> str:
    color = INSIDER_COLORS.get(str(val), INSIDER_COLORS["—"])
    return f"color: {color}; font-weight: 600"


def _apply_filters(df: pd.DataFrame, search: str, sector: str, insider: str) -> pd.DataFrame:
    if search:
        mask = (
            df["Ticker"].str.contains(search, case=False, na=False)
            | df["Company"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if sector and sector != "All":
        df = df[df["Sector"] == sector]
    if insider and insider != "All":
        df = df[df["Insider Signal"] == insider]
    return df


def _render_table(df: pd.DataFrame, col_order: list[str]) -> None:
    display_cols = [c for c in col_order if c in df.columns]
    styled = df[display_cols].style.map(_style_insider, subset=["Insider Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)


def _render_charts(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        sector_data = df["Sector"].value_counts().reset_index()
        sector_data.columns = ["Sector", "Count"]
        fig = px.pie(
            sector_data, values="Count", names="Sector",
            title="Sector Distribution", hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pb_vals = df["P/B Ratio"].apply(_parse_float).dropna()
        if len(pb_vals) > 0:
            fig = px.histogram(
                pb_vals, title="P/B Ratio Distribution",
                nbins=20, color_discrete_sequence=["#0d6efd"],
                labels={"value": "P/B Ratio", "count": "Stocks"},
            )
            fig.update_layout(
                margin=dict(t=40, b=0, l=0, r=0), height=320, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No P/B data to chart.")


def _trigger_workflow(force_reports: bool) -> None:
    try:
        token = st.secrets["GH_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo  = st.secrets.get("GITHUB_REPO", "stock-screener")
    except KeyError as exc:
        st.error(f"Missing Streamlit secret: {exc}. Add it under Settings → Secrets.")
        return

    url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        "/actions/workflows/daily_screener.yml/dispatches"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={"ref": "master", "inputs": {"force_reports": str(force_reports).lower()}},
        timeout=10,
    )
    if resp.status_code == 204:
        st.success("Run triggered! Results appear in ~30–45 min once the Action completes.")
    else:
        st.error(f"GitHub API returned {resp.status_code}: {resp.text[:300]}")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Screener")
    st.caption("NYSE + NASDAQ · Market Cap ≥ $2B · P/B < 2×")

    dates = available_dates()

    if not dates:
        st.warning("No data yet. Trigger a run below or wait for the daily job.")
        selected_date = None
    else:
        selected_date = st.selectbox(
            "Date",
            options=dates,
            index=0,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y"),
        )

    if selected_date:
        df_30q  = load_csv(selected_date, "below30Q")
        df_50q  = load_csv(selected_date, "below50Q")
        df_100q = load_csv(selected_date, "below100Q")
        df_mom  = load_csv(selected_date, "momentum")

        st.divider()
        st.subheader("Screen counts")
        c1, c2 = st.columns(2)
        c1.metric("Below 30Q",  len(df_30q)  if df_30q  is not None else 0)
        c1.metric("Below 50Q",  len(df_50q)  if df_50q  is not None else 0)
        c2.metric("Below 100Q", len(df_100q) if df_100q is not None else 0)
        c2.metric("Momentum",   len(df_mom)  if df_mom  is not None else 0)

    st.divider()
    st.subheader("Manual Trigger")
    force = st.toggle("Force research reports", value=False)
    if st.button("Run Screener Now", type="primary", use_container_width=True):
        _trigger_workflow(force)
    st.caption("Needs GH_TOKEN + GITHUB_OWNER in Streamlit secrets.")

# ── Guard: no data ─────────────────────────────────────────────────────────────

if not selected_date:
    st.info(
        "No screening data found in `data/`. "
        "Trigger a run from the sidebar, or wait for the daily 7 AM ET GitHub Actions job."
    )
    st.stop()

# ── Global filters ─────────────────────────────────────────────────────────────

st.subheader(
    f"Results — {datetime.strptime(selected_date, '%Y-%m-%d').strftime('%B %d, %Y')}"
)

all_sectors: set[str] = set()
for _df in [df_30q, df_50q, df_100q, df_mom]:
    if _df is not None and "Sector" in _df.columns:
        all_sectors.update(_df["Sector"].dropna().tolist())
sector_opts = ["All"] + sorted(s for s in all_sectors if s and s != "—")

fc1, fc2, fc3 = st.columns([2, 2, 2])
search_q      = fc1.text_input("Search", placeholder="Ticker or company name", label_visibility="collapsed")
sector_filter = fc2.selectbox("Sector", sector_opts, label_visibility="collapsed")
insider_filter = fc3.selectbox(
    "Insider signal", ["All", "BUYING", "NEUTRAL", "SELLING", "—"],
    label_visibility="collapsed",
)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔵 Below 30Q SMA", "🟠 Below 50Q SMA", "🔴 Below 100Q SMA", "🚀 Momentum"]
)


def _value_tab(df: pd.DataFrame | None, label: str, sma_col: str) -> None:
    if df is None or df.empty:
        st.info(f"No {label} data for {selected_date}.")
        return

    filtered = _apply_filters(df, search_q, sector_filter, insider_filter)
    st.caption(f"{len(filtered):,} of {len(df):,} stocks")

    cols = [sma_col if c == "% Below SMA" else c for c in VALUE_COLS]
    _render_table(filtered, cols)

    with st.expander("Charts"):
        _render_charts(filtered)


with tab1:
    st.markdown(
        "Price below its **30-quarter (7.5 yr) SMA** — mild undervaluation vs long-run average."
    )
    _value_tab(df_30q, "Below 30Q", "30Q SMA")

with tab2:
    st.markdown(
        "Price below its **50-quarter (12.5 yr) SMA** — moderate undervaluation."
    )
    _value_tab(df_50q, "Below 50Q", "50Q SMA")

with tab3:
    st.markdown(
        "Price below its **100-quarter (25 yr) SMA** — deep undervaluation vs 25-year history."
    )
    _value_tab(df_100q, "Below 100Q", "100Q SMA")

with tab4:
    st.markdown(
        "**Bull-aligned quarterly SMAs** (Price > 20Q > 30Q > 50Q), "
        "within 15% of 20Q support, insider buying or neutral."
    )

    if df_mom is None or df_mom.empty:
        st.info(f"No momentum data for {selected_date}.")
    else:
        filtered_mom = _apply_filters(df_mom, search_q, sector_filter, insider_filter)
        st.caption(f"{len(filtered_mom):,} of {len(df_mom):,} stocks")
        _render_table(filtered_mom, MOM_COLS)

        with st.expander("Charts"):
            _render_charts(filtered_mom)

        # Research reports
        reports = load_reports(selected_date)
        if reports:
            st.divider()
            st.subheader(f"Research Reports ({len(reports)})")
            st.caption("Claude-generated investment memos for new momentum names.")
            for ticker, text in reports.items():
                with st.expander(f"📄 {ticker}"):
                    st.text(text)
        else:
            st.caption(
                "No research reports for this date — memos are generated on Mondays "
                "or when 'Force research reports' is toggled."
            )

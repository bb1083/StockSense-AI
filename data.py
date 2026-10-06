"""ชั้นดึงข้อมูล (Yahoo Finance) - ทุกฟังก์ชันป้องกัน error และมี cache"""
from __future__ import annotations
import numpy as np, pandas as pd, yfinance as yf, streamlit as st
from concurrent.futures import ThreadPoolExecutor

SECTOR_MAP = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "CRM", "ADBE", "QCOM", "INTC"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "F"],
    "Consumer Defensive": ["WMT", "COST", "KO", "PEP"],
    "Healthcare": ["LLY", "JNJ", "ABBV", "MRK", "PFE", "UNH", "ISRG", "ABT", "AMGN", "GILD"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK"],
    "Energy": ["XOM", "CVX"],
    "Industrials": ["CAT", "GE", "BA", "UPS"],
}


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _clean_hist(h: pd.DataFrame) -> pd.DataFrame:
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.copy()
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    h.index = pd.to_datetime(h.index).normalize()
    return h[~h.index.duplicated(keep="last")].dropna(subset=["Close"])


def benchmark_for(ticker: str) -> str:
    return "^SET.BK" if ticker.endswith(".BK") else "SPY"


@st.cache_data(ttl=300, show_spinner=False)
def load_bundle(ticker: str) -> dict:
    """ดึงข้อมูลทั้งหมดที่ใช้วิเคราะห์หุ้น 1 ตัว"""
    t = yf.Ticker(ticker)
    hist = _clean_hist(_safe(lambda: t.history(period="5y", interval="1d", auto_adjust=True), pd.DataFrame()))
    if hist.empty:
        return {"ok": False, "ticker": ticker}
    bench_sym = benchmark_for(ticker)
    bench = _clean_hist(_safe(lambda: yf.Ticker(bench_sym).history(period="5y", interval="1d", auto_adjust=True), pd.DataFrame()))
    return {
        "ok": True, "ticker": ticker, "bench_symbol": bench_sym,
        "hist": hist, "bench": bench,
        "info": _safe(lambda: t.info, {}) or {},
        "income": _safe(lambda: t.income_stmt, pd.DataFrame()),
        "balance": _safe(lambda: t.balance_sheet, pd.DataFrame()),
        "cash": _safe(lambda: t.cashflow, pd.DataFrame()),
        "recs": _safe(lambda: t.recommendations_summary, pd.DataFrame()),
        "targets": _safe(lambda: t.analyst_price_targets, {}) or {},
        "earnings": _safe(lambda: t.earnings_dates, pd.DataFrame()),
        "news": _safe(lambda: t.news, []) or [],
    }


@st.cache_data(ttl=60, show_spinner=False)
def live_price(ticker: str):
    d = _safe(lambda: yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=True), pd.DataFrame())
    if d is None or d.empty:
        return None, None
    d = d.dropna(subset=["Close"])
    return (float(d["Close"].iloc[-1]), d.index[-1]) if len(d) else (None, None)


@st.cache_data(ttl=3600, show_spinner=False)
def search_symbols(query: str) -> list[dict]:
    """ค้นหาจากชื่อบริษัท เช่น 'nvidia' -> NVDA"""
    res = _safe(lambda: yf.Search(query, max_results=6).quotes, [])
    return [{"symbol": q.get("symbol"), "name": q.get("shortname") or q.get("longname", ""), "exch": q.get("exchDisp", "")}
            for q in res if q.get("symbol") and q.get("quoteType") in ("EQUITY", "ETF")]


@st.cache_data(ttl=3600, show_spinner=False)
def peer_table(ticker: str, sector: str, n: int = 5) -> pd.DataFrame:
    peers = [p for p in SECTOR_MAP.get(sector, []) if p != ticker and p != "GOOG"][:n]
    syms = [ticker] + peers

    def one(s):
        i = _safe(lambda: yf.Ticker(s).info, {}) or {}
        return {"Ticker": s, "Market Cap": i.get("marketCap"), "Fwd P/E": i.get("forwardPE"),
                "PEG": i.get("trailingPegRatio") or i.get("pegRatio"), "P/S": i.get("priceToSalesTrailing12Months"),
                "Net Margin": i.get("profitMargins"), "ROE": i.get("returnOnEquity"),
                "Rev Growth": i.get("revenueGrowth"), "Beta": i.get("beta")}
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, syms))
    return pd.DataFrame(rows)

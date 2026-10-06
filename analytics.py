"""เอนจินวิเคราะห์หุ้น: Technical, Risk, Fundamentals, Valuation, Scorecard, Narrative (ไม่พึ่ง Streamlit)"""
from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path
import joblib, numpy as np, pandas as pd
from train_universal_model import build_features

BASE = Path(__file__).resolve().parent
NAN = float("nan")

# ---------- helpers ----------
def f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else NAN
    except (TypeError, ValueError):
        return NAN

def ok(x) -> bool:
    return not (x is None or (isinstance(x, float) and math.isnan(x)))

def lin(x, bad, good):
    """แปลงค่าเป็นคะแนน 0-100 (bad->0, good->100) ใช้ได้ทั้งแบบมากดีและน้อยดี"""
    x = f(x)
    return None if not ok(x) else float(np.clip((x - bad) / (good - bad) * 100, 0, 100))

def pct(x, d=1): return "N/A" if not ok(f(x)) else f"{x*100:.{d}f}%"
def num(x, d=2): return "N/A" if not ok(f(x)) else f"{x:,.{d}f}"
def big(x):
    x = f(x)
    if not ok(x): return "N/A"
    for u, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= u: return f"{x/u:,.2f}{s}"
    return f"{x:,.0f}"

# ---------- technical ----------
def add_indicators(h: pd.DataFrame) -> pd.DataFrame:
    d = h.copy(); c = d["Close"]
    for n in (20, 50, 200): d[f"SMA{n}"] = c.rolling(n).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]
    dl = c.diff()
    rs = dl.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (-dl.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + rs)
    sd = c.rolling(20).std()
    d["BB_UP"], d["BB_LO"] = d["SMA20"] + 2 * sd, d["SMA20"] - 2 * sd
    pc = c.shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return d

def tech_signals(d: pd.DataFrame) -> tuple[list[dict], dict]:
    L = d.iloc[-1]; p = L["Close"]; out = []
    def add(name, val, state, text): out.append(dict(name=name, value=val, state=state, text=text))
    if ok(f(L["SMA200"])):
        gap = p / L["SMA200"] - 1
        add("ราคา vs SMA200", pct(gap), "bull" if gap > 0.02 else "bear" if gap < -0.02 else "neutral",
            "เทรนด์ระยะยาวขาขึ้น" if gap > 0.02 else "ต่ำกว่าเส้นแนวโน้มระยะยาว" if gap < -0.02 else "ใกล้เส้นแนวโน้มระยะยาว")
    if ok(f(L["SMA200"])) and ok(f(L["SMA50"])):
        up = L["SMA50"] > L["SMA200"]
        add("SMA50 vs SMA200", "Golden" if up else "Death", "bull" if up else "bear",
            "SMA50 อยู่เหนือ SMA200" if up else "SMA50 อยู่ใต้ SMA200")
    r = f(L["RSI"])
    if ok(r):
        add("RSI (14)", num(r, 1), "bear" if r > 75 else "bull" if r < 30 else "neutral",
            "ซื้อมากเกินไป เสี่ยงพักตัว" if r > 75 else "ขายมากเกินไป อาจเกิดรีบาวด์" if r < 30 else "โมเมนตัมอยู่ในโซนปกติ")
    if ok(f(L["MACD_HIST"])):
        h = L["MACD_HIST"]
        add("MACD Histogram", num(h, 3), "bull" if h > 0 else "bear", "โมเมนตัมระยะสั้นเป็นบวก" if h > 0 else "โมเมนตัมระยะสั้นเป็นลบ")
    if ok(f(L["BB_UP"])):
        pb = (p - L["BB_LO"]) / (L["BB_UP"] - L["BB_LO"])
        add("Bollinger %B", num(pb, 2), "bear" if pb > 1 else "bull" if pb < 0 else "neutral",
            "ราคาทะลุกรอบบน (ตึงตัว)" if pb > 1 else "ราคาทะลุกรอบล่าง" if pb < 0 else "ราคาอยู่ในกรอบปกติ")
    v20, v60 = d["Volume"].tail(20).mean(), d["Volume"].tail(60).mean()
    if v60 > 0:
        r_ = v20 / v60
        add("Volume 20D/60D", num(r_, 2) + "x", "neutral", "ปริมาณซื้อขายสูงกว่าปกติ" if r_ > 1.2 else "ปริมาณซื้อขายเบาบาง" if r_ < 0.8 else "ปริมาณซื้อขายปกติ")
    last252 = d.tail(252)
    levels = dict(price=p, hi52=last252["High"].max(), lo52=last252["Low"].min(),
                  s1=d["Low"].tail(20).min(), s2=d["Low"].tail(60).min(),
                  r1=d["High"].tail(20).max(), atr=f(L["ATR"]))
    return out, levels

# ---------- risk ----------
def risk_metrics(h: pd.DataFrame, bench: pd.DataFrame, rf: float = 0.04) -> dict:
    c = h["Close"]; r = c.pct_change().dropna()
    n = len(r); tot = c.iloc[-1] / c.iloc[0] - 1
    ann_ret = (1 + tot) ** (252 / max(n, 1)) - 1
    vol = r.std() * np.sqrt(252)
    dn = r[r < 0].std() * np.sqrt(252)
    dd = c / c.cummax() - 1
    out = dict(ann_ret=ann_ret, vol=vol, sharpe=(r.mean() * 252 - rf) / vol if vol else NAN,
               sortino=(r.mean() * 252 - rf) / dn if dn else NAN, max_dd=dd.min(), cur_dd=dd.iloc[-1],
               var95=-r.quantile(0.05), cvar95=-r[r <= r.quantile(0.05)].mean(),
               skew=r.skew(), kurt=r.kurt(), worst=r.min(), best=r.max(),
               beta=NAN, corr=NAN, alpha=NAN, dd_series=dd, returns=r)
    if bench is not None and not bench.empty:
        br = bench["Close"].pct_change().dropna()
        j = pd.concat([r, br], axis=1, join="inner").dropna().tail(756)
        if len(j) > 120:
            j.columns = ["a", "b"]
            cov = np.cov(j["a"], j["b"])
            out["beta"] = cov[0, 1] / cov[1, 1]
            out["corr"] = j["a"].corr(j["b"])
            out["alpha"] = (j["a"].mean() - out["beta"] * j["b"].mean()) * 252
    return out

def monte_carlo(h: pd.DataFrame, horizon=252, n=4000, mu=0.08, seed=7) -> dict:
    """Bootstrap จากผลตอบแทนจริง 3 ปีล่าสุด แต่ปรับ drift เป็น mu เพื่อไม่ให้ผลในอดีตครอบงำ"""
    lr = np.log(h["Close"]).diff().dropna().values[-756:]
    rng = np.random.default_rng(seed)
    draw = (lr - lr.mean())[rng.integers(0, len(lr), (n, horizon))] + math.log(1 + mu) / 252
    paths = h["Close"].iloc[-1] * np.exp(np.cumsum(draw, axis=1))
    qs = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fin = paths[:, -1] / h["Close"].iloc[-1] - 1
    return dict(q=qs, final=fin, p_loss=(fin < 0).mean(), p_loss20=(fin < -0.2).mean(),
                p_gain30=(fin > 0.3).mean(), median=np.median(fin), p5=np.percentile(fin, 5), p95=np.percentile(fin, 95))

@lru_cache(maxsize=1)
def _vol_pack():
    p = BASE / "universal_vol_model.joblib"
    try:
        return joblib.load(p) if p.exists() else None
    except Exception:  # เช่น scikit-learn คนละเวอร์ชัน -> ให้รัน train_universal_model.py ใหม่ แอปยังทำงานต่อได้
        return None

def forecast_vol(h: pd.DataFrame, rev_growth=NAN, de=NAN) -> dict | None:
    pack = _vol_pack()
    if pack is None or len(h) < 260: return None
    g = build_features(h[["Close", "Volume"]])
    last = g.iloc[[-1]].copy()
    last["Revenue_Growth"] = float(np.clip(rev_growth, -1, 3)) if ok(f(rev_growth)) else NAN
    last["Debt_to_Equity"] = float(np.clip(de, -50, 50)) if ok(f(de)) else NAN
    X = last[pack["features"]]
    core = X.drop(columns=["Revenue_Growth", "Debt_to_Equity"])
    if core.isna().any(axis=None) or not np.isfinite(core.values).all(): return None
    m = pack["metrics"]; anchor = float(last["Volatility_1Y"].iloc[0])
    pred = anchor * math.exp(float(pack["model"].predict(X)[0]))
    lo, hi = pred * math.exp(m["log_interval80"][0]), pred * math.exp(m["log_interval80"][1])
    cls = "Low" if pred <= m["low_threshold"] else "Medium" if pred <= m["high_threshold"] else "High"
    return dict(pred=pred, lo=lo, hi=hi, cls=cls, anchor=anchor, vol21=float(last["Vol_21D"].iloc[0]), metrics=m)

# ---------- fundamentals ----------
def row(df, *names, col=0):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return NAN
    for n in names:
        if n in df.index:
            try: return f(df.loc[n].iloc[col])
            except IndexError: return NAN
    return NAN

def _ratio(a, b): return a / b if ok(a) and ok(b) and b != 0 else NAN
def _chg(a, b): return a / b - 1 if ok(a) and ok(b) and b > 0 else NAN

def piotroski(inc, bal, cf):
    ni0, ni1 = row(inc, "Net Income", col=0), row(inc, "Net Income", col=1)
    ta0, ta1 = row(bal, "Total Assets", col=0), row(bal, "Total Assets", col=1)
    cfo0 = row(cf, "Operating Cash Flow", col=0)
    rev0, rev1 = row(inc, "Total Revenue", col=0), row(inc, "Total Revenue", col=1)
    gp0, gp1 = row(inc, "Gross Profit", col=0), row(inc, "Gross Profit", col=1)
    ltd0, ltd1 = row(bal, "Long Term Debt", col=0), row(bal, "Long Term Debt", col=1)
    ca0, cl0 = row(bal, "Current Assets", col=0), row(bal, "Current Liabilities", col=0)
    ca1, cl1 = row(bal, "Current Assets", col=1), row(bal, "Current Liabilities", col=1)
    sh0, sh1 = row(bal, "Ordinary Shares Number", col=0), row(bal, "Ordinary Shares Number", col=1)
    roa0, roa1 = _ratio(ni0, ta0), _ratio(ni1, ta1)
    tests = [
        ("กำไรสุทธิเป็นบวก (ROA>0)", roa0 > 0 if ok(roa0) else None),
        ("กระแสเงินสดดำเนินงานเป็นบวก", cfo0 > 0 if ok(cfo0) else None),
        ("ROA ดีขึ้นจากปีก่อน", roa0 > roa1 if ok(roa0) and ok(roa1) else None),
        ("เงินสดจากงาน > กำไรสุทธิ (คุณภาพกำไร)", cfo0 > ni0 if ok(cfo0) and ok(ni0) else None),
        ("สัดส่วนหนี้ระยะยาวต่อสินทรัพย์ลดลง", _ratio(ltd0, ta0) < _ratio(ltd1, ta1) if ok(_ratio(ltd0, ta0)) and ok(_ratio(ltd1, ta1)) else None),
        ("Current Ratio ดีขึ้น", _ratio(ca0, cl0) > _ratio(ca1, cl1) if ok(_ratio(ca0, cl0)) and ok(_ratio(ca1, cl1)) else None),
        ("ไม่ออกหุ้นเพิ่ม (ไม่ dilute)", sh0 <= sh1 * 1.005 if ok(sh0) and ok(sh1) else None),
        ("Gross Margin ดีขึ้น", _ratio(gp0, rev0) > _ratio(gp1, rev1) if ok(_ratio(gp0, rev0)) and ok(_ratio(gp1, rev1)) else None),
        ("Asset Turnover ดีขึ้น", _ratio(rev0, ta0) > _ratio(rev1, ta1) if ok(_ratio(rev0, ta0)) and ok(_ratio(rev1, ta1)) else None),
    ]
    known = [t for t in tests if t[1] is not None]
    return dict(score=sum(bool(t[1]) for t in known), n=len(known), tests=tests)

def altman_z(inc, bal, mcap):
    ta = row(bal, "Total Assets"); tl = row(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
    wc = row(bal, "Working Capital")
    if not ok(wc): wc = row(bal, "Current Assets") - row(bal, "Current Liabilities")
    re_, ebit = row(bal, "Retained Earnings"), row(inc, "EBIT", "Operating Income")
    sales = row(inc, "Total Revenue")
    parts = [_ratio(wc, ta), _ratio(re_, ta), _ratio(ebit, ta), _ratio(mcap, tl), _ratio(sales, ta)]
    if not all(ok(x) for x in parts): return NAN
    return 1.2 * parts[0] + 1.4 * parts[1] + 3.3 * parts[2] + 0.6 * parts[3] + 1.0 * parts[4]

def fundamentals(inc, bal, cf, info, mcap) -> dict:
    n = min(4, inc.shape[1]) if isinstance(inc, pd.DataFrame) and not inc.empty else 0
    years = [pd.Timestamp(inc.columns[i]).year for i in range(n)][::-1] if n else []
    rev = [row(inc, "Total Revenue", col=i) for i in range(n)]
    ni = [row(inc, "Net Income", col=i) for i in range(n)]
    opi = [row(inc, "Operating Income", "EBIT", col=i) for i in range(n)]
    fcfs = []
    for i in range(min(4, cf.shape[1]) if isinstance(cf, pd.DataFrame) and not cf.empty else 0):
        v = row(cf, "Free Cash Flow", col=i)
        if not ok(v): v = row(cf, "Operating Cash Flow", col=i) + row(cf, "Capital Expenditure", col=i)
        fcfs.append(v)
    eq, ta = row(bal, "Stockholders Equity", "Total Equity Gross Minority Interest"), row(bal, "Total Assets")
    tl = row(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
    debt, cash = row(bal, "Total Debt"), row(bal, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    ebitda = row(inc, "EBITDA", "Normalized EBITDA"); ebit = row(inc, "EBIT", "Operating Income")
    intexp = abs(row(inc, "Interest Expense"))
    ocf = row(cf, "Operating Cash Flow")
    gm = _ratio(row(inc, "Gross Profit"), rev[0] if rev else NAN)
    out = dict(
        years=years, rev=rev[::-1], ni=ni[::-1], opi=opi[::-1], fcf_series=fcfs[::-1],
        gross_margin=gm, op_margin=_ratio(opi[0] if opi else NAN, rev[0] if rev else NAN),
        net_margin=_ratio(ni[0] if ni else NAN, rev[0] if rev else NAN),
        roe=_ratio(ni[0] if ni else NAN, eq) if ok(eq) and eq > 0 else NAN, roa=_ratio(ni[0] if ni else NAN, ta),
        rev_growth=_chg(rev[0], rev[1]) if len(rev) > 1 else NAN,
        rev_cagr=(rev[0] / rev[3]) ** (1 / 3) - 1 if len(rev) > 3 and ok(rev[0]) and ok(rev[3]) and rev[3] > 0 and rev[0] > 0 else NAN,
        ni_growth=_chg(ni[0], ni[1]) if len(ni) > 1 else NAN,
        fcf=fcfs[0] if fcfs else NAN, fcf_margin=_ratio(fcfs[0] if fcfs else NAN, rev[0] if rev else NAN),
        ocf_ni=_ratio(ocf, ni[0] if ni else NAN) if ok(ni[0] if ni else NAN) and ni[0] > 0 else NAN,
        de=_ratio(tl, eq) if ok(eq) and eq > 0 else NAN, debt=debt, cash=cash,
        nd_ebitda=_ratio(debt - cash, ebitda) if ok(debt) and ok(cash) and ok(ebitda) and ebitda > 0 else NAN,
        int_cov=_ratio(ebit, intexp) if ok(intexp) and intexp > 0 else NAN,
        current_ratio=_ratio(row(bal, "Current Assets"), row(bal, "Current Liabilities")),
        piotroski=piotroski(inc, bal, cf), altman=altman_z(inc, bal, mcap), equity=eq,
    )
    return out

def reverse_dcf(mcap, fcf, disc=0.10, tg=0.025, years=10):
    """ตลาดกำลัง 'ตีราคา' ว่า FCF จะโตปีละกี่ % ต่อเนื่อง 10 ปี (สมมติฐานง่าย ๆ)"""
    if not (ok(f(mcap)) and ok(f(fcf)) and mcap > 0 and fcf > 0): return None
    def val(g):
        cf, pv = fcf, 0.0
        for t in range(1, years + 1):
            cf *= 1 + g; pv += cf / (1 + disc) ** t
        return pv + cf * (1 + tg) / (disc - tg) / (1 + disc) ** years
    lo, hi = -0.5, 1.0
    if val(hi) < mcap: return hi
    if val(lo) > mcap: return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if val(mid) < mcap else (lo, mid)
    return (lo + hi) / 2

# ---------- street & news ----------
POS = set("beat beats surge surges soar soars jump jumps rally rallies record growth strong upgrade upgrades raises raised profit gain gains rise rises bullish outperform buy expands expansion breakthrough win wins approval approved boost boosts higher tops".split())
NEG = set("miss misses plunge plunges fall falls drop drops slump downgrade downgrades cut cuts loss losses weak lawsuit probe investigation recall fraud warning warns bearish underperform sell layoffs decline declines risk risks fine fined delay delays lower slumps tumbles sinks".split())

def parse_news(items, k=10):
    out = []
    for it in items[:k]:
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = c.get("title") or ""
        if not title: continue
        summ = c.get("summary") or ""
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        pub = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        words = [w.strip(".,:;!?'\"()").lower() for w in (title + " " + summ).split()]
        p, n_ = sum(w in POS for w in words), sum(w in NEG for w in words)
        out.append(dict(title=title, link=link, source=pub or "", date=c.get("pubDate") or "", tone=(p - n_) / (p + n_ + 1)))
    return out

def street_view(recs, targets, info, price):
    out = dict(mean_rating=NAN, counts=None, target_mean=f(targets.get("mean")), target_hi=f(targets.get("high")),
               target_lo=f(targets.get("low")), upside=NAN, n_analysts=f(info.get("numberOfAnalystOpinions")))
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        r0 = recs.iloc[0]; cnt = {k: int(f(r0.get(k, 0)) if ok(f(r0.get(k, 0))) else 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
        tot = sum(cnt.values())
        if tot: out["mean_rating"] = (5*cnt["strongBuy"] + 4*cnt["buy"] + 3*cnt["hold"] + 2*cnt["sell"] + cnt["strongSell"]) / tot; out["counts"] = cnt
    if not ok(out["mean_rating"]) and ok(f(info.get("recommendationMean"))): out["mean_rating"] = 6 - f(info["recommendationMean"])
    if ok(out["target_mean"]) and price: out["upside"] = out["target_mean"] / price - 1
    return out

def earnings_view(df):
    out = dict(next=None, surprises=[], avg_surprise=NAN)
    if not isinstance(df, pd.DataFrame) or df.empty: return out
    d = df.copy()
    if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
    now = pd.Timestamp.now().normalize()
    fut = d[(d.index >= now) & d["Reported EPS"].isna()] if "Reported EPS" in d else d.iloc[0:0]
    if len(fut): out["next"] = fut.index.min()
    if "Surprise(%)" in d:
        hist = d[d["Surprise(%)"].notna()].sort_index().tail(8)
        out["surprises"] = [(i, f(v)) for i, v in zip(hist.index, hist["Surprise(%)"])]
        last4 = hist["Surprise(%)"].tail(4)
        if len(last4): out["avg_surprise"] = f(last4.mean()) / 100
    return out

# ---------- scorecard ----------
PILLARS = {"Quality": "คุณภาพธุรกิจ", "Growth": "การเติบโต", "Valuation": "ความถูก/แพง",
           "Trend": "เทรนด์ & โมเมนตัม", "Safety": "ความปลอดภัย", "Street": "ฉันทามติ & ข่าว"}
PROFILES = {
    "สมดุล (Balanced)": dict(Quality=20, Growth=15, Valuation=20, Trend=15, Safety=20, Street=10),
    "อนุรักษ์นิยม (Conservative)": dict(Quality=25, Growth=5, Valuation=20, Trend=10, Safety=30, Street=10),
    "เน้นเติบโต (Aggressive)": dict(Quality=10, Growth=30, Valuation=10, Trend=25, Safety=10, Street=15),
}

def build_scorecard(ctx: dict, weights: dict) -> dict:
    fu, va, rk, tl, sv, nw, ev, vf = (ctx[k] for k in ("fund", "val", "risk", "tech_last", "street", "news", "earn", "volf"))
    info = ctx["info"]; items = []
    def add(p, name, value, score, note=""):
        items.append(dict(pillar=p, name=name, value=value, score=score, note=note))
    # Quality
    add("Quality", "ROE", pct(fu["roe"]), lin(fu["roe"], 0, 0.25), "ผลตอบแทนต่อส่วนผู้ถือหุ้น")
    add("Quality", "Operating Margin", pct(fu["op_margin"]), lin(fu["op_margin"], 0, 0.30), "กำไรจากการดำเนินงาน")
    add("Quality", "Gross Margin", pct(fu["gross_margin"]), lin(fu["gross_margin"], 0.15, 0.60), "อำนาจต่อรอง/ความได้เปรียบ")
    pf = fu["piotroski"]
    add("Quality", "Piotroski F-Score", f"{pf['score']}/{pf['n']}" if pf["n"] >= 6 else "N/A",
        lin(pf["score"] / pf["n"] * 9, 3, 8) if pf["n"] >= 6 else None, "คะแนนงบการเงิน 9 ข้อ")
    add("Quality", "OCF / Net Income", num(fu["ocf_ni"], 2) + "x" if ok(fu["ocf_ni"]) else "N/A", lin(fu["ocf_ni"], 0.5, 1.2), "กำไรเป็นเงินสดจริงแค่ไหน")
    # Growth
    add("Growth", "Revenue Growth YoY", pct(fu["rev_growth"]), lin(fu["rev_growth"], -0.05, 0.25))
    add("Growth", "Revenue CAGR 3Y", pct(fu["rev_cagr"]), lin(fu["rev_cagr"], 0, 0.20))
    eg = f(info.get("earningsGrowth")); eg = fu["ni_growth"] if not ok(eg) else eg
    add("Growth", "Earnings Growth", pct(eg), lin(eg, -0.10, 0.30))
    feps, teps = f(info.get("forwardEps")), f(info.get("trailingEps"))
    fg = _chg(feps, teps) if ok(feps) and ok(teps) and teps > 0 else NAN
    add("Growth", "Forward EPS vs Trailing", pct(fg), lin(fg, -0.05, 0.30), "นักวิเคราะห์คาดกำไรโตต่อ")
    # Valuation
    add("Valuation", "Forward P/E", num(va["fpe"], 1), lin(va["fpe"], 40, 12) if ok(va["fpe"]) and va["fpe"] > 0 else None)
    add("Valuation", "PEG", num(va["peg"], 2), lin(va["peg"], 3, 1) if ok(va["peg"]) and va["peg"] > 0 else None, "P/E เทียบการเติบโต")
    add("Valuation", "EV/EBITDA", num(va["ev_ebitda"], 1), lin(va["ev_ebitda"], 30, 10) if ok(va["ev_ebitda"]) and va["ev_ebitda"] > 0 else None)
    add("Valuation", "FCF Yield", pct(va["fcf_yield"]), lin(va["fcf_yield"], 0, 0.06), "กระแสเงินสดอิสระต่อมูลค่าตลาด")
    add("Valuation", "Price/Sales", num(va["ps"], 1), lin(va["ps"], 15, 2) if ok(va["ps"]) and va["ps"] > 0 else None)
    # Trend
    L = tl
    if ok(f(L.get("SMA200"))): add("Trend", "ราคา vs SMA200", pct(L["Close"] / L["SMA200"] - 1), lin(L["Close"] / L["SMA200"] - 1, -0.15, 0.15))
    if ok(f(L.get("SMA200"))) and ok(f(L.get("SMA50"))): add("Trend", "SMA50 vs SMA200", "Golden" if L["SMA50"] > L["SMA200"] else "Death", 100.0 if L["SMA50"] > L["SMA200"] else 0.0)
    if ok(f(L.get("RSI"))): add("Trend", "RSI (14)", num(L["RSI"], 1), float(np.clip(100 - abs(L["RSI"] - 58) * 3.5, 0, 100)), "โซนแข็งแรงแต่ไม่ร้อนแรง")
    add("Trend", "Momentum 12-1M", pct(ctx["mom12_1"]), lin(ctx["mom12_1"], -0.20, 0.40), "ผลตอบแทน 12 เดือน ไม่รวมเดือนล่าสุด")
    if ok(f(L.get("MACD_HIST"))): add("Trend", "MACD", "บวก" if L["MACD_HIST"] > 0 else "ลบ", 70.0 if L["MACD_HIST"] > 0 else 30.0)
    # Safety
    add("Safety", "Volatility 1Y", pct(rk["vol"]), lin(rk["vol"], 0.60, 0.15))
    add("Safety", "Max Drawdown 5Y", pct(rk["max_dd"]), lin(rk["max_dd"], -0.65, -0.15))
    add("Safety", "Beta", num(rk["beta"], 2), lin(rk["beta"], 1.8, 0.7))
    add("Safety", "Altman Z", num(fu["altman"], 2), lin(fu["altman"], 1.8, 3.0) if not ctx["is_financial"] else None, "ความเสี่ยงล้มละลาย (ไม่ใช้กับสถาบันการเงิน)")
    add("Safety", "Net Debt/EBITDA", num(fu["nd_ebitda"], 1) + "x" if ok(fu["nd_ebitda"]) else "N/A", lin(fu["nd_ebitda"], 4, 0.5) if not ctx["is_financial"] else None)
    add("Safety", "Current Ratio", num(fu["current_ratio"], 2), lin(fu["current_ratio"], 0.8, 1.8) if not ctx["is_financial"] else None)
    if vf: add("Safety", "Vol Forecast 63D", pct(vf["pred"]), lin(vf["pred"], 0.60, 0.15), "จากโมเดล ML (ดู Model Card)")
    # Street
    add("Street", "Analyst Rating (1-5)", num(sv["mean_rating"], 2), lin(sv["mean_rating"], 2.5, 4.5))
    add("Street", "Target Upside", pct(sv["upside"]), lin(sv["upside"], -0.10, 0.25))
    add("Street", "EPS Surprise 4Q", pct(ev["avg_surprise"]), lin(ev["avg_surprise"], -0.05, 0.08))
    if nw: add("Street", "News Tone", num(float(np.mean([n["tone"] for n in nw])), 2), lin(float(np.mean([n["tone"] for n in nw])), -0.3, 0.3), "ประเมินด้วยพจนานุกรมคำ (หยาบ)")
    pillars = {}
    for p in PILLARS:
        its = [i for i in items if i["pillar"] == p]; sc = [i["score"] for i in its if i["score"] is not None]
        pillars[p] = dict(score=float(np.mean(sc)) if sc else None, items=its, n_ok=len(sc), n_all=len(its))
    avail = {p: v for p, v in pillars.items() if v["score"] is not None}
    wsum = sum(weights[p] for p in avail)
    overall = sum(weights[p] * v["score"] for p, v in avail.items()) / wsum if wsum else NAN
    coverage = sum(v["n_ok"] for v in pillars.values()) / max(1, sum(v["n_all"] for v in pillars.values()))
    spread = float(np.std([v["score"] for v in avail.values()])) if avail else NAN
    conf = "สูง" if coverage >= 0.8 and spread < 20 else "ปานกลาง" if coverage >= 0.6 else "ต่ำ"
    return dict(pillars=pillars, items=items, overall=overall, coverage=coverage, spread=spread, confidence=conf)

def verdict(score, coverage=1.0, has_quality=True):
    """ป้ายสรุป - ถ้าข้อมูลไม่ครบจะไม่ให้ป้ายเชิงบวกชัดเจน เพื่อไม่ให้ตัดสินเกินหลักฐาน"""
    if not ok(f(score)) or coverage < 0.4: return ("ข้อมูลไม่เพียงพอต่อการสรุป", "#94A3B8")
    if score >= 70: lab, col = "หลักฐานเชิงบวกชัดเจน", "#22D3A6"
    elif score >= 58: lab, col = "โน้มเอียงเชิงบวก", "#7DD87F"
    elif score >= 45: lab, col = "สัญญาณผสม ควรศึกษาเพิ่ม", "#FBBF24"
    elif score >= 32: lab, col = "ควรระวัง", "#FB923C"
    else: lab, col = "หลักฐานเชิงลบเด่นชัด", "#F43F5E"
    if not has_quality or coverage < 0.6:
        return ("บางส่วน: " + lab + " (ข้อมูลงบไม่ครบ)", "#FBBF24" if score >= 58 else col)
    return (lab, col)


# ---------- narrative / checklist / sizing ----------
def red_flags(ctx, sc) -> list[str]:
    fu, rk, vf, ev = ctx["fund"], ctx["risk"], ctx["volf"], ctx["earn"]; fl = []
    if ok(fu["altman"]) and fu["altman"] < 1.8 and not ctx["is_financial"]: fl.append(f"Altman Z = {fu['altman']:.2f} (<1.8 โซนเสี่ยงทางการเงิน)")
    if fu["piotroski"]["n"] >= 6 and fu["piotroski"]["score"] <= 3: fl.append(f"Piotroski F-Score ต่ำ ({fu['piotroski']['score']}/{fu['piotroski']['n']})")
    if ok(fu["fcf"]) and fu["fcf"] < 0: fl.append("กระแสเงินสดอิสระ (FCF) ติดลบ")
    if ok(fu["equity"]) and fu["equity"] <= 0: fl.append("ส่วนของผู้ถือหุ้นติดลบ")
    if ok(fu["nd_ebitda"]) and fu["nd_ebitda"] > 4 and not ctx["is_financial"]: fl.append(f"หนี้สุทธิ/EBITDA สูง ({fu['nd_ebitda']:.1f}x)")
    if ok(fu["int_cov"]) and fu["int_cov"] < 2: fl.append(f"ความสามารถจ่ายดอกเบี้ยต่ำ ({fu['int_cov']:.1f}x)")
    if ok(rk["cur_dd"]) and rk["cur_dd"] < -0.30: fl.append(f"ราคาอยู่ต่ำกว่าจุดสูงสุด 5 ปี {abs(rk['cur_dd'])*100:.0f}%")
    if vf and vf["cls"] == "High": fl.append(f"โมเดลคาดความผันผวนอีก 3 เดือนสูง (~{vf['pred']*100:.0f}% ต่อปี)")
    if ev["next"] is not None and 0 <= (ev["next"] - pd.Timestamp.now().normalize()).days <= 14:
        fl.append(f"ประกาศงบใน {(ev['next'] - pd.Timestamp.now().normalize()).days} วัน ({ev['next']:%d %b %Y}) ราคามักแกว่งแรง")
    return fl

def checklist(ctx, sc) -> list[tuple[str, str, str]]:
    """(สถานะ ok/warn/bad, หัวข้อ, รายละเอียด)"""
    fu, va, rk, L, ev, vf = ctx["fund"], ctx["val"], ctx["risk"], ctx["tech_last"], ctx["earn"], ctx["volf"]
    c = []
    def add(cond, t, good, bad, warn=None):
        if cond is None: c.append(("warn", t, "ข้อมูลไม่พอสำหรับข้อนี้"))
        elif cond: c.append(("ok", t, good))
        else: c.append((("warn" if warn else "bad"), t, warn or bad))
    add(fu["fcf"] > 0 if ok(fu["fcf"]) else None, "ธุรกิจสร้างเงินสดอิสระ", f"FCF {big(fu['fcf'])}", "FCF ติดลบ")
    add(fu["piotroski"]["score"] >= 6 if fu["piotroski"]["n"] >= 6 else None, "คุณภาพงบการเงิน", f"F-Score {fu['piotroski']['score']}/{fu['piotroski']['n']}", "F-Score ต่ำกว่าเกณฑ์", f"F-Score {fu['piotroski']['score']}/{fu['piotroski']['n']} (กลาง ๆ)" if fu["piotroski"]["score"] >= 4 else None)
    add(fu["altman"] >= 1.8 if ok(fu["altman"]) and not ctx["is_financial"] else None, "ความเสี่ยงล้มละลาย", f"Altman Z {fu['altman']:.2f}" if ok(fu["altman"]) else "", "Altman Z ในโซนเสี่ยง")
    add((va["fpe"] <= 35) if ok(va["fpe"]) and va["fpe"] > 0 else None, "ราคาไม่ตึงเกินไป", f"Forward P/E {num(va['fpe'],1)}", f"Forward P/E {num(va['fpe'],1)} สูง คาดหวังการเติบโตมาก")
    add(L["Close"] > L["SMA200"] if ok(f(L.get("SMA200"))) else None, "เทรนด์ระยะยาวเป็นขาขึ้น", "ราคาอยู่เหนือ SMA200", "ราคาอยู่ต่ำกว่า SMA200")
    add((rk["cur_dd"] > -0.20), "ไม่ได้ร่วงหนักจากจุดสูงสุด", f"ต่ำกว่าจุดสูงสุด {abs(rk['cur_dd'])*100:.0f}%", f"ร่วงจากจุดสูงสุด {abs(rk['cur_dd'])*100:.0f}% (ตรวจสาเหตุ)")
    nd = (ev["next"] - pd.Timestamp.now().normalize()).days if ev["next"] is not None else None
    add(None if nd is None else nd > 14, "ไม่ใกล้วันประกาศงบ", f"งบถัดไปอีก {nd} วัน" if nd is not None else "", "ประกาศงบภายใน 14 วัน ความผันผวนอาจสูง")
    add(rk["vol"] < 0.45, "ความผันผวนรับได้", f"Volatility {pct(rk['vol'],0)}", f"Volatility {pct(rk['vol'],0)} สูง ควรลดขนาดไม้")
    return c

def position_size(price, atr, capital, risk_pct, atr_mult, max_pos_pct):
    if not (ok(f(price)) and ok(f(atr)) and atr > 0): return None
    stop_dist = atr * atr_mult; risk_amt = capital * risk_pct / 100
    shares = math.floor(min(risk_amt / stop_dist, capital * max_pos_pct / 100 / price))
    return dict(stop=price - stop_dist, stop_pct=stop_dist / price, shares=shares, value=shares * price,
                risk_amt=shares * stop_dist, weight=shares * price / capital if capital else NAN)

def summary_text(ctx, sc, flags) -> str:
    v, _ = verdict(sc["overall"], sc["coverage"], sc["pillars"]["Quality"]["score"] is not None); P = sc["pillars"]
    ranked = sorted([(PILLARS[p], x["score"]) for p, x in P.items() if x["score"] is not None], key=lambda t: -t[1])
    s = f"คะแนนรวม **{sc['overall']:.0f}/100** ({v}) ความเชื่อมั่นของผลวิเคราะห์ระดับ **{sc['confidence']}** (ครอบคลุมข้อมูล {sc['coverage']*100:.0f}%). "
    if len(ranked) >= 2:
        s += f"ด้านที่เด่นที่สุดคือ **{ranked[0][0]}** ({ranked[0][1]:.0f}) ส่วนด้านที่อ่อนที่สุดคือ **{ranked[-1][0]}** ({ranked[-1][1]:.0f}). "
    vf = ctx["volf"]
    if vf: s += f"โมเดลประเมินความผันผวนอีก ~3 เดือนข้างหน้าอยู่ราว {vf['pred']*100:.0f}% ต่อปี (ช่วง 80%: {vf['lo']*100:.0f}–{vf['hi']*100:.0f}%). "
    s += f"พบจุดที่ควรตรวจสอบเพิ่ม {len(flags)} ข้อ." if flags else "ไม่พบ Red Flag ตามเกณฑ์ที่ตั้งไว้."
    return s


# ---------- valuation + orchestration ----------
def valuation(info, fund, mcap) -> dict:
    return dict(
        pe=f(info.get("trailingPE")), fpe=f(info.get("forwardPE")),
        peg=f(info.get("trailingPegRatio") if info.get("trailingPegRatio") is not None else info.get("pegRatio")),
        ps=f(info.get("priceToSalesTrailing12Months")), pb=f(info.get("priceToBook")),
        ev_ebitda=f(info.get("enterpriseToEbitda")),
        fcf_yield=_ratio(fund["fcf"], mcap) if ok(mcap) and mcap > 0 else NAN,
        div_yield=f(info.get("trailingAnnualDividendYield")), mcap=mcap)

def analyze(b: dict) -> dict:
    """รวมทุกโมดูลเป็น context เดียว (ใช้ได้ทั้งใน app และเทสต์)"""
    h, info = b["hist"], b["info"]
    d = add_indicators(h); sigs, levels = tech_signals(d)
    price = float(h["Close"].iloc[-1]); mcap = f(info.get("marketCap"))
    fu = fundamentals(b["income"], b["balance"], b["cash"], info, mcap)
    ctx = dict(info=info, hist=h, ind=d, tech_last=d.iloc[-1], signals=sigs, levels=levels, price=price,
               fund=fu, val=valuation(info, fu, mcap), risk=risk_metrics(h, b["bench"]),
               news=parse_news(b["news"]), street=street_view(b["recs"], b["targets"], info, price),
               earn=earnings_view(b["earnings"]),
               mom12_1=float(h["Close"].iloc[-22] / h["Close"].iloc[-253] - 1) if len(h) > 253 else NAN,
               is_financial=info.get("sector") == "Financial Services")
    ctx["volf"] = forecast_vol(h, fu["rev_growth"], fu["de"])
    ctx["rdcf"] = reverse_dcf(mcap, fu["fcf"])
    return ctx

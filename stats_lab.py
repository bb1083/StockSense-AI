"""Stats Lab: เครื่องมือสถิติเชิงอนุมานที่ใช้กับข้อมูลหุ้นจริง (ไม่พึ่ง Streamlit)"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import xlogy

TD = 252


def _partial_year(idx) -> bool:
    return not (idx[-1].month == 12 and idx[-1].day >= 24)


def return_methods(close: pd.Series) -> dict | None:
    """HPR, ค่าเฉลี่ยเลขคณิต, ค่าเฉลี่ยเรขาคณิต, CAGR (ใช้ผลตอบแทนปีปฏิทินที่ครบปี)"""
    c = close.dropna()
    ye = c.groupby(c.index.year).last()
    ann = ye.pct_change().dropna()
    if _partial_year(c.index) and len(ann):
        ann = ann.iloc[:-1]
    if len(ann) < 2:
        return None
    days = (c.index[-1] - c.index[0]).days
    arith = float(ann.mean())
    geo = float((1 + ann).prod() ** (1 / len(ann)) - 1)
    return dict(annual=ann, hpr=float(c.iloc[-1] / c.iloc[0] - 1), arith=arith, geo=geo,
                cagr=float((c.iloc[-1] / c.iloc[0]) ** (365.25 / days) - 1), drag=arith - geo,
                years=len(ann), first=str(c.index[0].date()), last=str(c.index[-1].date()))


def ttest_mean(r: pd.Series) -> dict:
    """H0: ผลตอบแทนเฉลี่ยรายวัน = 0"""
    n = len(r); t, p = stats.ttest_1samp(r, 0.0)
    se = r.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return dict(n=n, t=float(t), p=float(p), mean_ann=float(r.mean() * TD),
                ci_lo=float((r.mean() - tc * se) * TD), ci_hi=float((r.mean() + tc * se) * TD))


def capm(r: pd.Series, b: pd.Series, rf: float = 0.04, window: int = 756) -> dict | None:
    j = pd.concat([r, b], axis=1, join="inner").dropna().tail(window)
    if len(j) < 120:
        return None
    y, x = j.iloc[:, 0] - rf / TD, j.iloc[:, 1] - rf / TD
    res = stats.linregress(x, y); n = len(j); tc = stats.t.ppf(0.975, n - 2)
    ta = res.intercept / res.intercept_stderr
    return dict(n=n, beta=float(res.slope), beta_lo=float(res.slope - tc * res.stderr), beta_hi=float(res.slope + tc * res.stderr),
                beta_p=float(res.pvalue), alpha_ann=float(res.intercept * TD),
                alpha_p=float(2 * stats.t.sf(abs(ta), n - 2)), r2=float(res.rvalue ** 2), x=x.values, y=y.values)


def normality(r: pd.Series) -> dict:
    jb = stats.jarque_bera(r)
    return dict(jb=float(jb.statistic), p=float(jb.pvalue), skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r)))


def anova_by_year(r: pd.Series, min_obs: int = 60) -> dict | None:
    """H0: ผลตอบแทนเฉลี่ยรายวันเท่ากันทุกปี | F-test + Tukey HSD (post hoc) + Kruskal-Wallis (ทนต่อความไม่ปกติ)"""
    groups = {int(y): g.values for y, g in r.groupby(r.index.year) if len(g) >= min_obs}
    if len(groups) < 3:
        return None
    ys = list(groups); arr = [groups[y] for y in ys]
    f, p = stats.f_oneway(*arr); kw = stats.kruskal(*arr)
    tk = stats.tukey_hsd(*arr); pairs = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if tk.pvalue[i, j] < 0.05:
                pairs.append((ys[i], ys[j], float((arr[i].mean() - arr[j].mean()) * TD), float(tk.pvalue[i, j])))
    return dict(f=float(f), p=float(p), kw_p=float(kw.pvalue), k=len(ys), n=int(sum(len(a) for a in arr)),
                means={y: float(groups[y].mean() * TD) for y in ys}, groups=groups, pairs=pairs)


def sharpe_bootstrap(r: pd.Series, rf: float = 0.04, n: int = 2000, seed: int = 11) -> dict:
    v = r.values; rng = np.random.default_rng(seed)
    s = v[rng.integers(0, len(v), (n, len(v)))]
    sh = (s.mean(axis=1) * TD - rf) / (s.std(axis=1, ddof=1) * np.sqrt(TD))
    pt = (v.mean() * TD - rf) / (v.std(ddof=1) * np.sqrt(TD))
    return dict(point=float(pt), lo=float(np.percentile(sh, 2.5)), hi=float(np.percentile(sh, 97.5)), p_pos=float((sh > 0).mean()))


def kupiec(r: pd.Series, alpha: float = 0.05, win: int = 252, test_days: int = 504) -> dict | None:
    """ทดสอบ VaR 95% แบบ Historical: จำนวนวันที่ขาดทุนเกิน VaR ควรอยู่ราว 5% (Kupiec POF test)"""
    var = (-r.rolling(win).quantile(alpha)).shift(1)
    d = pd.concat([r, var], axis=1).dropna().tail(test_days)
    if len(d) < 250:
        return None
    n = len(d); x = int((d.iloc[:, 0] < -d.iloc[:, 1]).sum()); ph = x / n
    lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)) + 2 * (xlogy(n - x, 1 - ph) + xlogy(x, ph))
    return dict(n=n, x=x, rate=ph, lr=float(lr), p=float(stats.chi2.sf(lr, 1)))


def dca(monthly: float, years: int, annual: float) -> dict:
    """ลงทุนรายเดือน: เทียบ 'ทบต้นรายเดือน (nominal/12)' กับ 'ผลตอบแทนที่แท้จริงต่อปี (effective)' """
    m = years * 12; out = {}
    for k, i in (("nominal", annual / 12), ("effective", (1 + annual) ** (1 / 12) - 1)):
        out[k] = monthly * m if abs(i) < 1e-12 else monthly * (((1 + i) ** m - 1) / i)
    out["paid"] = monthly * m
    return out


def run_all(h: pd.DataFrame, bench: pd.DataFrame, rf: float = 0.04) -> dict:
    r = h["Close"].pct_change().dropna()
    br = bench["Close"].pct_change().dropna() if bench is not None and not bench.empty else None
    def safe(fn, *a):
        try: return fn(*a)
        except Exception: return None
    return dict(methods=safe(return_methods, h["Close"]), ttest=safe(ttest_mean, r),
                capm=safe(capm, r, br, rf) if br is not None else None, normality=safe(normality, r),
                anova=safe(anova_by_year, r), sharpe=safe(sharpe_bootstrap, r, rf), kupiec=safe(kupiec, r))

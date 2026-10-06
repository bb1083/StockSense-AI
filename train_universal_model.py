"""
StockSense AI v2 - Universal Volatility Model
- ไม่ใช้ Ticker / ราคา Close ดิบ เป็น Feature => ใช้ได้กับหุ้นทุกตัว (รวมหุ้นที่ไม่อยู่ใน 50 ตัวที่ใช้ฝึก)
- ทำนาย "ความผันผวนล่วงหน้า 63 วันทำการ" เป็นตัวเลขจริง (ผ่านอัตราส่วนต่อ Vol 1 ปี) แล้วแปลงเป็น Low/Medium/High
- เปรียบเทียบกับ Baseline (ใช้ Volatility ย้อนหลังตรง ๆ) และใช้ Walk-forward ตามเวลา + gap 63 วัน
ใช้งาน:  python train_universal_model.py [path/to/stock_risk_dataset.csv]
"""
import sys, json
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

FEATURES = ["Vol_21D", "Vol_63D", "Volatility_1Y", "Downside_Vol_63D", "Return_1M",
            "Momentum_3M", "Return_1Y", "Max_Drawdown", "Volume_Ratio",
            "Revenue_Growth", "Debt_to_Equity"]
GAP = 63
BASE = Path(__file__).resolve().parent


def build_features(g: pd.DataFrame) -> pd.DataFrame:
    """g: ต้องมี Close, Volume, Daily_Return (เรียงตามวันที่) -> เพิ่มคอลัมน์ Feature (ใช้ซ้ำใน app.py)"""
    g = g.copy()
    r = g["Close"].pct_change()
    g["Daily_Return"] = r
    g["Vol_21D"] = r.rolling(21).std() * np.sqrt(252)
    g["Vol_63D"] = r.rolling(63).std() * np.sqrt(252)
    g["Volatility_1Y"] = r.rolling(252, min_periods=126).std() * np.sqrt(252)
    g["Downside_Vol_63D"] = r.clip(upper=0).rolling(63).std() * np.sqrt(252)
    g["Return_1M"] = g["Close"].pct_change(21)
    g["Momentum_3M"] = g["Close"].pct_change(63)
    g["Return_1Y"] = g["Close"].pct_change(252)
    g["Max_Drawdown"] = g["Close"] / g["Close"].rolling(252, min_periods=126).max() - 1
    g["Volume_Ratio"] = g["Volume"] / g["Volume"].rolling(60).mean()
    return g


def main(csv):
    df = pd.read_csv(csv, parse_dates=["Date"]).sort_values(["Ticker", "Date"])
    base_cols = ["Date", "Ticker", "Close", "Volume", "Daily_Return", "Future_Volatility",
                 "Revenue_Growth", "Debt_to_Equity"]
    df = df[base_cols]
    parts = [build_features(g).assign(Ticker=t) for t, g in df.groupby("Ticker")]
    df = pd.concat(parts, ignore_index=True)
    df["Debt_to_Equity"] = df["Debt_to_Equity"].clip(-50, 50)
    df["Revenue_Growth"] = df["Revenue_Growth"].clip(-1, 3)
    core = [f for f in FEATURES if f not in ("Revenue_Growth", "Debt_to_Equity")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=core + ["Future_Volatility"])

    dates = np.array(sorted(df["Date"].unique()))
    cut = int(len(dates) * 0.8)
    train = df[df["Date"] < dates[cut - GAP]]
    test = df[df["Date"] >= dates[cut]]

    lo, hi = train["Future_Volatility"].quantile([0.33, 0.67])
    to_cls = lambda v: np.where(v <= lo, "Low", np.where(v <= hi, "Medium", "High"))

    # ทำนาย "อัตราส่วน" log(Future_Vol / Vol_1Y) แทนระดับ Volatility ตรง ๆ
    # => ไม่ผูกกับระดับ Volatility ของช่วงฝึก (เช่น COVID-2020) และใช้ได้กับหุ้นทุกตัว
    params = dict(max_depth=3, learning_rate=0.04, max_iter=200, l2_regularization=5.0,
                  min_samples_leaf=500, random_state=42)
    ratio = lambda d: np.log(d["Future_Volatility"] / d["Volatility_1Y"])
    model = HistGradientBoostingRegressor(**params).fit(train[FEATURES], ratio(train))
    anchor_te = test["Volatility_1Y"].values
    pred = anchor_te * np.exp(model.predict(test[FEATURES]))
    y = test["Future_Volatility"].values
    base = anchor_te  # Baseline: "อนาคตผันผวนเท่ากับปีที่ผ่านมา"

    # ช่วงความเชื่อมั่น 80% จาก residual ของ hold-out (ในสเกล log-ratio)
    resid = np.log(y / pred)
    q10, q90 = np.quantile(resid, [0.10, 0.90])

    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_start": str(pd.Timestamp(dates[cut]).date()),
        "r2_model": float(r2_score(y, pred)), "r2_baseline": float(r2_score(y, base)),
        "mae_model": float(mean_absolute_error(y, pred)), "mae_baseline": float(mean_absolute_error(y, base)),
        "acc_model": float(accuracy_score(to_cls(y), to_cls(pred))),
        "acc_baseline": float(accuracy_score(to_cls(y), to_cls(base))),
        "f1_model": float(f1_score(to_cls(y), to_cls(pred), average="macro")),
        "f1_baseline": float(f1_score(to_cls(y), to_cls(base), average="macro")),
        "low_threshold": float(lo), "high_threshold": float(hi),
        "log_interval80": [float(q10), float(q90)],
        "n_tickers_train": int(df["Ticker"].nunique()),
    }
    final = HistGradientBoostingRegressor(**params).fit(df[FEATURES], ratio(df))
    joblib.dump({"model": final, "features": FEATURES, "metrics": metrics},
                BASE / "universal_vol_model.joblib", compress=3)
    json.dump(metrics, open(BASE / "universal_model_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BASE / "data" / "stock_risk_dataset.csv")

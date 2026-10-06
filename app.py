import streamlit as st
import math, re
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data
import analytics as A
import ai_assistant
import ui
from ui import html, kpi, chip, bar, stretch, PLOT, GRID

st.set_page_config(page_title="StockSense AI", page_icon="⚡", layout="wide")

st.markdown(ui.CSS, unsafe_allow_html=True)
CUR = {"USD": "$", "THB": "฿", "EUR": "€", "GBP": "£", "JPY": "¥", "HKD": "HK$"}
EXAMPLES = ["NVDA", "AAPL", "MSFT", "TSLA", "JPM", "PTT.BK"]

# ====================== cached wrappers ======================
@st.cache_data(ttl=300, show_spinner=False)
def get_ctx(ticker: str):
    return A.analyze(data.load_bundle(ticker))


@st.cache_data(ttl=300, show_spinner=False)
def get_mc(ticker: str, mu: float):
    return A.monte_carlo(data.load_bundle(ticker)["hist"], mu=mu)


# ====================== sidebar ======================
st.sidebar.markdown("### ⚡ StockSense AI")
st.sidebar.caption("พิมพ์ชื่อหุ้นแล้วรับการวิเคราะห์")
with st.sidebar.form("search", clear_on_submit=False):
    q = st.text_input("Ticker / ชื่อบริษัท", value=st.session_state.get("ticker", ""), placeholder="เช่น NVDA, AAPL, PTT.BK")
    submitted = st.form_submit_button("วิเคราะห์ ⚡", **stretch(st.form_submit_button))
if submitted and q.strip():
    st.session_state["ticker"] = q.strip().upper()

profile = st.sidebar.selectbox("โปรไฟล์นักลงทุน", list(A.PROFILES), help="เปลี่ยนน้ำหนักของแต่ละมิติในคะแนนรวม")
weights = dict(A.PROFILES[profile])
with st.sidebar.expander("ปรับน้ำหนักคะแนนเอง"):
    for p, th in A.PILLARS.items():
        weights[p] = st.slider(th, 0, 40, weights[p], key=f"w_{profile}_{p}")
mu = st.sidebar.slider("ผลตอบแทนคาดหวังต่อปี (สำหรับ Monte Carlo)", 0, 20, 8, help="ใช้ปรับ drift ของการจำลอง ไม่ใช้ผลตอบแทนในอดีตของหุ้น") / 100
if st.sidebar.toggle("รีเฟรชราคาอัตโนมัติ (60 วินาที)", value=False):
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="auto")
    except ImportError:
        st.sidebar.caption("ติดตั้งก่อน: pip install streamlit-autorefresh")
st.sidebar.divider()
st.sidebar.caption("⚠️ เครื่องมือนี้ให้ข้อมูลเพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน")


# ====================== landing ======================
def landing():
    html("""<div class="glass glow" style="padding:34px 36px">
    <div class="sub mono">STOCKSENSE AI · 360° DUE DILIGENCE ENGINE</div>
    <div class="tick" style="font-size:3.2rem;margin:10px 0 8px">พิมพ์แค่ชื่อหุ้น<br>ได้ทุกอย่างที่ต้องใช้ตัดสินใจ</div>
    <div class="sub" style="max-width:760px"> ให้คะแนน 6 มิติ พร้อมบอกระดับความเชื่อมั่นของผลวิเคราะห์ทุกครั้ง</div></div>""")
    st.write("")
    feats = [("🎯 Decision Hub", "คะแนน 6 มิติ + Bull/Bear case + Red Flags + Checklist ก่อนซื้อ "),
             ("📈 Technical Lab", "SMA/EMA, RSI, MACD, Bollinger, ATR และแนวรับ-แนวต้านจากข้อมูลจริง"),
             ("🏢 Fundamental X-Ray", "Piotroski F-Score, Altman Z, ROE, margin, FCF, หนี้สิน จากงบการเงินย้อนหลัง"),
             ("💰 Valuation & Reverse DCF", "ตลาดกำลังคาดหวังการเติบโตเท่าไรจากราคานี้ พร้อมตารางสถานการณ์ผลตอบแทน"),
             ("🛡️ Risk Lab", "Beta, VaR/CVaR, Drawdown, โมเดล ML คาดความผันผวน และ Monte Carlo 1 ปี"),
             ("🧪 Model Card", "เปิดเผยผล backtest จริงของโมเดลเทียบ baseline ไม่โอ้อวดเกินความจริง")]
    cols = st.columns(3)
    for i, (t, d) in enumerate(feats):
        with cols[i % 3]:
            html(f'<div class="glass feat"><h4>{t}</h4><p>{d}</p></div>')
            st.write("")
    st.markdown("**ลองเลย:**")
    cs = st.columns(len(EXAMPLES))
    for c, t in zip(cs, EXAMPLES):
        if c.button(t, key=f"ex_{t}", **stretch(st.button)):
            st.session_state["ticker"] = t
            st.rerun()


ticker = st.session_state.get("ticker")
if not ticker:
    landing()
    st.stop()

# ====================== load ======================
with st.spinner(f"กำลังดึงข้อมูลและคำนวณ {ticker} ..."):
    bundle = data.load_bundle(ticker)
    if not bundle["ok"]:
        st.error(f"ไม่พบข้อมูลสำหรับ “{ticker}”")
        sug = data.search_symbols(ticker)
        if sug:
            st.markdown("**คุณหมายถึงตัวใดตัวหนึ่งนี้หรือไม่?**")
            for s in sug:
                if st.button(f"{s['symbol']} · {s['name']} ({s['exch']})", key=f"sg_{s['symbol']}"):
                    st.session_state["ticker"] = s["symbol"]
                    st.rerun()
        else:
            st.caption("หุ้นไทยใส่ .BK ต่อท้าย เช่น PTT.BK, หุ้นสหรัฐใส่ชื่อย่อ เช่น AAPL")
        st.stop()
    if len(bundle["hist"]) < 60:
        st.warning("หุ้นนี้มีประวัติราคาสั้นเกินไป (<60 วัน) จึงวิเคราะห์ได้ไม่น่าเชื่อถือ")
        st.stop()
    ctx = get_ctx(ticker)

info, h, rk, fu, va, L = ctx["info"], ctx["hist"], ctx["risk"], ctx["fund"], ctx["val"], ctx["tech_last"]
sym = CUR.get(info.get("currency", "USD"), info.get("currency", "") + " ")
sc = A.build_scorecard(ctx, weights)
flags = A.red_flags(ctx, sc)
vlabel, vcolor = A.verdict(sc["overall"], sc["coverage"], sc["pillars"]["Quality"]["score"] is not None)

try:
    lp, lt = data.live_price(ticker)
except Exception:
    lp, lt = None, None
price = lp if lp else ctx["price"]
prev = float(h["Close"].iloc[-2]) if (lt is not None and h.index[-1].date() == lt.date()) or (lt is None and len(h) > 1) else float(h["Close"].iloc[-1])
chg = price / prev - 1
lv = ctx["levels"]
pos52 = (price - lv["lo52"]) / (lv["hi52"] - lv["lo52"]) * 100 if lv["hi52"] > lv["lo52"] else 50
name = info.get("longName") or info.get("shortName") or ticker

# ====================== hero ======================
c1, c2 = st.columns([3.2, 1.2])
with c1:
    html(f"""<div class="glass glow">
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
      <div><div class="tick">{ticker}</div><div class="sub" style="margin-top:6px">{name}</div>
      <div style="margin-top:8px">{chip(info.get('sector','—'))}{chip(info.get('industry','—'))}{chip('Mkt Cap ' + A.big(va['mcap']))}{chip('Benchmark ' + bundle['bench_symbol'])}</div></div>
      <div style="text-align:right"><div class="price">{sym}{price:,.2f}</div>
      <div class="{'up' if chg >= 0 else 'dn'} mono">{'▲' if chg >= 0 else '▼'} {chg*100:+.2f}% วันนี้</div>
      <div class="sub">{'Live · ' + lt.strftime('%d %b %H:%M') if lt is not None else 'ราคาปิดล่าสุด'}</div></div></div>
    <div class="sub" style="margin-top:14px;display:flex;justify-content:space-between"><span>52W Low {sym}{lv['lo52']:,.2f}</span><span>52W High {sym}{lv['hi52']:,.2f}</span></div>
    <div class="rng"><i style="left:calc({pos52:.0f}% - 2px)"></i></div></div>""")
with c2:
    html(f"""<div class="glass" style="display:flex;flex-direction:column;align-items:center;gap:8px;height:100%;justify-content:center">
    <div class="ring" style="--p:{0 if not A.ok(sc['overall']) else sc['overall']:.0f};--c:{vcolor}"><div><b>{'—' if not A.ok(sc['overall']) else format(sc['overall'], '.0f')}</b><small>/ 100</small></div></div>
    <div style="color:{vcolor};font-weight:600;text-align:center;font-size:.95rem">{vlabel}</div>
    <div class="sub">ความเชื่อมั่นผลวิเคราะห์: <b>{sc['confidence']}</b></div></div>""")

st.write("")
brief = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", A.summary_text(ctx, sc, flags))
html(f"""<div class="glass"><div class="sub mono" style="margin-bottom:6px">STOCKSENSE BRIEF</div>
<div style="font-size:1.0rem;line-height:1.7">{brief}</div></div>""")
st.write("")

tabs = st.tabs(["🎯 Decision Hub", "📈 Technical", "🏢 Fundamentals", "💰 Valuation", "🛡️ Risk Lab", "📰 Street & News", "🆚 Peers", "🧪 Model Card"])

# ====================== 1. decision hub ======================
with tabs[0]:
    a, b = st.columns([1, 1.15])
    avail = {p: v for p, v in sc["pillars"].items() if v["score"] is not None}
    with a:
        names = [A.PILLARS[p] for p in avail] + [A.PILLARS[list(avail)[0]]]
        vals = [v["score"] for v in avail.values()] + [list(avail.values())[0]["score"]]
        fig = go.Figure(go.Scatterpolar(r=vals, theta=names, fill="toself", line=dict(color="#22D3EE", width=2), fillcolor="rgba(34,211,238,.18)"))
        fig.update_layout(**PLOT, height=380, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="rgba(120,150,220,.18)", tickfont=dict(size=9)),
                                                         angularaxis=dict(gridcolor="rgba(120,150,220,.18)")), showlegend=False)
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    with b:
        rows = ""
        for p, th in A.PILLARS.items():
            v = sc["pillars"][p]
            if v["score"] is None:
                rows += f"<div class='prow'><span>{th}</span><span class='mono sub'>ไม่มีข้อมูล</span></div>"
            else:
                rows += bar(v["score"], f"{th} <span class='sub'>(น้ำหนัก {weights[p]}%)</span>", f"{v['score']:.0f}")
        html(f"<div class='glass'>{rows}<div style='margin-top:14px'>{chip('ครอบคลุมข้อมูล ' + format(sc['coverage']*100, '.0f') + '%', 'ok' if sc['coverage'] >= .8 else 'warn')}"
             f"{chip('สัญญาณสอดคล้องกัน' if A.ok(sc['spread']) and sc['spread'] < 20 else 'สัญญาณขัดแย้งกัน', 'ok' if A.ok(sc['spread']) and sc['spread'] < 20 else 'warn')}</div></div>")

    its = [i for i in sc["items"] if i["score"] is not None]
    top = sorted([i for i in its if i["score"] >= 65], key=lambda i: -i["score"])[:5]
    low = sorted([i for i in its if i["score"] <= 35], key=lambda i: i["score"])[:5]
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("#### 🟢 Bull Case")
        html("".join(f'<div class="pt g"><b>{i["name"]}</b> · <span class="mono">{i["value"]}</span><br><span class="sub">{i["note"]}</span></div>' for i in top) or '<div class="pt">ไม่มีตัวชี้วัดที่โดดเด่นเป็นพิเศษ</div>')
    with g2:
        st.markdown("#### 🔴 Bear Case")
        html("".join(f'<div class="pt r"><b>{i["name"]}</b> · <span class="mono">{i["value"]}</span><br><span class="sub">{i["note"]}</span></div>' for i in low) or '<div class="pt">ไม่พบจุดอ่อนที่ชัดเจน</div>')
    with g3:
        st.markdown("#### 🚩 Red Flags")
        html("".join(f'<div class="pt a">{f_}</div>' for f_ in flags) or '<div class="pt g">ไม่พบ Red Flag ตามเกณฑ์ที่ตั้งไว้</div>')

    st.markdown("#### ✅ Pre-Buy Checklist")
    cl = A.checklist(ctx, sc)
    cc = st.columns(2)
    icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}
    for i, (s_, t_, d_) in enumerate(cl):
        with cc[i % 2]:
            html(f'<div class="pt {"g" if s_ == "ok" else "a" if s_ == "warn" else "r"}">{icon[s_]} <b>{t_}</b><br><span class="sub">{d_}</span></div>')

    st.markdown("#### 🧮 คำนวณขนาดไม้ (Position Sizing)")
    st.caption("ตั้ง Stop จาก ATR เพื่อให้ขาดทุนสูงสุดต่อไม้ไม่เกินที่คุณกำหนด")
    p1, p2, p3, p4 = st.columns(4)
    cap = p1.number_input("เงินลงทุนรวม", 100.0, 1e9, 10000.0, 500.0)
    rp = p2.number_input("ยอมขาดทุนต่อไม้ (%)", 0.1, 10.0, 1.0, 0.1)
    am = p3.number_input("Stop = ATR ×", 0.5, 6.0, 2.0, 0.5)
    mp = p4.number_input("สูงสุดต่อตัว (% พอร์ต)", 1.0, 100.0, 20.0, 1.0)
    ps = A.position_size(price, lv["atr"], cap, rp, am, mp)
    if ps:
        k = st.columns(5)
        with k[0]: kpi("จำนวนหุ้น", f"{ps['shares']:,}")
        with k[1]: kpi("มูลค่าตำแหน่ง", f"{sym}{ps['value']:,.0f}", f"{ps['weight']*100:.1f}% ของพอร์ต")
        with k[2]: kpi("ราคา Stop", f"{sym}{ps['stop']:,.2f}", f"ห่าง {ps['stop_pct']*100:.1f}%")
        with k[3]: kpi("ขาดทุนสูงสุด", f"{sym}{ps['risk_amt']:,.0f}", f"{ps['risk_amt']/cap*100:.2f}% ของพอร์ต")
        with k[4]: kpi("ATR (14)", f"{sym}{lv['atr']:,.2f}")

    with st.expander("ดูตัวชี้วัดทั้งหมดที่ใช้คำนวณคะแนน (ตรวจสอบย้อนกลับได้)"):
        df = pd.DataFrame([{"มิติ": A.PILLARS[i["pillar"]], "ตัวชี้วัด": i["name"], "ค่า": i["value"], "คะแนน": None if i["score"] is None else round(i["score"]), "หมายเหตุ": i["note"]} for i in sc["items"]])
        st.dataframe(df, hide_index=True, **stretch(st.dataframe))
        st.caption("คะแนนแต่ละตัวชี้วัดแปลงเป็น 0–100 ด้วยเกณฑ์คงที่ (เช่น ROE 0%→0, 25%→100) แล้วเฉลี่ยเป็นคะแนนรายมิติ และถ่วงน้ำหนักตามโปรไฟล์ เกณฑ์เป็นแนวทางทั่วไป ไม่ได้ปรับตามอุตสาหกรรม")

# ====================== 2. technical ======================
with tabs[1]:
    d = ctx["ind"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.5, .12, .19, .19], vertical_spacing=.02)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="ราคา",
                                 increasing_line_color="#22D3A6", decreasing_line_color="#F43F5E"), 1, 1)
    for col, color in (("SMA20", "#FBBF24"), ("SMA50", "#22D3EE"), ("SMA200", "#8B5CF6")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1.3, color=color)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_UP"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_LO"], fill="tonexty", fillcolor="rgba(139,92,246,.07)", line=dict(width=0), name="Bollinger", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color="rgba(143,160,192,.45)", name="Volume"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], line=dict(color="#22D3EE", width=1.4), name="RSI"), 3, 1)
    for y_, c_ in ((70, "#F43F5E"), (30, "#22D3A6")): fig.add_hline(y=y_, line=dict(color=c_, width=1, dash="dot"), row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], marker_color=np.where(d["MACD_HIST"] >= 0, "#22D3A6", "#F43F5E"), name="MACD Hist"), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], line=dict(color="#FBBF24", width=1.2), name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIG"], line=dict(color="#8B5CF6", width=1.2), name="Signal"), 4, 1)
    fig.update_layout(**PLOT, height=780, xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(range=[d.index[max(0, len(d) - 252)], d.index[-1]], **GRID,
                     rangeselector=dict(buttons=[dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                                                 dict(count=1, label="1Y", step="year", stepmode="backward"), dict(count=3, label="3Y", step="year", stepmode="backward"), dict(step="all", label="5Y")],
                                        bgcolor="#0E1730", font=dict(color="#C9D4EE")), row=1, col=1)
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, **stretch(st.plotly_chart))

    k = st.columns(5)
    for col_, (lab, val, sub) in zip(k, [("แนวรับ 1 (20D Low)", lv["s1"], ""), ("แนวรับ 2 (60D Low)", lv["s2"], ""), ("ราคาปัจจุบัน", price, ""),
                                         ("แนวต้าน 1 (20D High)", lv["r1"], ""), ("แนวต้าน 2 (52W High)", lv["hi52"], "")]):
        with col_: kpi(lab, f"{sym}{val:,.2f}", sub)
    st.markdown("#### สัญญาณเทคนิค")
    sg = st.columns(2)
    for i, s_ in enumerate(ctx["signals"]):
        cls = {"bull": "g", "bear": "r", "neutral": "a"}[s_["state"]]
        with sg[i % 2]:
            html(f'<div class="pt {cls}"><b>{s_["name"]}</b> · <span class="mono">{s_["value"]}</span><br><span class="sub">{s_["text"]}</span></div>')
    st.caption("สัญญาณเทคนิคสะท้อนพฤติกรรมราคาในอดีต ใช้ประกอบการจับจังหวะ ไม่ใช่ตัวทำนายอนาคต")

# ====================== 3. fundamentals ======================
with tabs[2]:
    if not fu["years"]:
        st.warning("ไม่พบงบการเงินของหุ้นนี้จาก Yahoo Finance (พบบ่อยในหุ้นนอกสหรัฐ/หุ้นขนาดเล็ก) คะแนนด้านคุณภาพและการเติบโตจึงถูกคำนวณเท่าที่มี")
    else:
        k = st.columns(6)
        for col_, (l_, v_, s_) in zip(k, [("Gross Margin", A.pct(fu["gross_margin"]), ""), ("Operating Margin", A.pct(fu["op_margin"]), ""), ("Net Margin", A.pct(fu["net_margin"]), ""),
                                           ("ROE", A.pct(fu["roe"]), ""), ("Revenue YoY", A.pct(fu["rev_growth"]), f"CAGR 3Y {A.pct(fu['rev_cagr'])}"), ("FCF Margin", A.pct(fu["fcf_margin"]), "")]):
            with col_: kpi(l_, v_, s_)
        st.write("")
        fa, fb = st.columns([1.4, 1])
        with fa:
            fig = go.Figure()
            fig.add_bar(x=fu["years"], y=fu["rev"], name="Revenue", marker_color="#22D3EE")
            fig.add_bar(x=fu["years"], y=fu["opi"], name="Operating Income", marker_color="#8B5CF6")
            fig.add_bar(x=fu["years"], y=fu["ni"], name="Net Income", marker_color="#22D3A6")
            if fu["fcf_series"]: fig.add_bar(x=fu["years"][-len(fu["fcf_series"]):], y=fu["fcf_series"], name="Free Cash Flow", marker_color="#FBBF24")
            fig.update_layout(**PLOT, height=380, barmode="group", title="ผลประกอบการรายปี", xaxis=dict(type="category"))
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, **stretch(st.plotly_chart))
        with fb:
            pf = fu["piotroski"]
            st.markdown(f"#### Piotroski F-Score: {pf['score']}/{pf['n']}")
            html("".join(f'<div class="pt {"g" if r_ else "r" if r_ is False else "a"}">{"✅" if r_ else "❌" if r_ is False else "➖"} {t_}</div>' for t_, r_ in pf["tests"]))
        st.markdown("#### สุขภาพงบดุล")
        k = st.columns(5)
        z = fu["altman"]
        zl = "Safe" if A.ok(z) and z >= 3 else "Grey" if A.ok(z) and z >= 1.8 else "Distress" if A.ok(z) else ""
        for col_, (l_, v_, s_) in zip(k, [("Debt / Equity", A.num(fu["de"], 2), ""), ("Net Debt / EBITDA", A.num(fu["nd_ebitda"], 1) + "x" if A.ok(fu["nd_ebitda"]) else "N/A", ""),
                                           ("Interest Coverage", A.num(fu["int_cov"], 1) + "x" if A.ok(fu["int_cov"]) else "N/A", ""), ("Current Ratio", A.num(fu["current_ratio"], 2), ""),
                                           ("Altman Z", A.num(z, 2) if not ctx["is_financial"] else "N/A", zl if not ctx["is_financial"] else "ไม่ใช้กับสถาบันการเงิน")]):
            with col_: kpi(l_, v_, s_)
        st.caption("ข้อมูลงบมาจาก Yahoo Finance (รายปี) อาจมีความล่าช้าหรือขาดบางรายการ")

# ====================== 4. valuation ======================
with tabs[3]:
    k = st.columns(6)
    for col_, (l_, v_) in zip(k, [("P/E (TTM)", A.num(va["pe"], 1)), ("Forward P/E", A.num(va["fpe"], 1)), ("PEG", A.num(va["peg"], 2)),
                                  ("EV/EBITDA", A.num(va["ev_ebitda"], 1)), ("P/S", A.num(va["ps"], 1)), ("FCF Yield", A.pct(va["fcf_yield"]))] ):
        with col_: kpi(l_, v_)
    st.write("")
    ra, rb = st.columns([1, 1.2])
    with ra:
        st.markdown("#### 🔬 Reverse DCF — ตลาดกำลังคาดหวังอะไร?")
        disc = st.slider("อัตราคิดลด (Discount rate)", 6.0, 15.0, 10.0, 0.5) / 100
        tg = st.slider("การเติบโตระยะยาว (Terminal growth)", 0.0, 4.0, 2.5, 0.5) / 100
        g = A.reverse_dcf(va["mcap"], fu["fcf"], disc, tg)
        if g is None:
            st.info("คำนวณไม่ได้: FCF เป็นลบหรือไม่มีข้อมูล (Reverse DCF ใช้ได้กับธุรกิจที่มี FCF เป็นบวก)")
        else:
            hist_g = [x for x in (fu["rev_cagr"], A.f(info.get("earningsGrowth"))) if A.ok(x)]
            ref = max(hist_g) if hist_g else None
            kpi("FCF ต้องโตต่อเนื่อง 10 ปี", f"{g*100:.1f}% / ปี", "เพื่อให้สมเหตุสมผลกับมูลค่าตลาดปัจจุบัน", "#22D3EE")
            if ref is not None:
                msg = ("ตลาดคาดหวัง **สูงกว่า** การเติบโตที่ผ่านมามาก ราคาเผื่อความผิดพลาดน้อย" if g > ref + 0.05 else
                       "ตลาดคาดหวังใกล้เคียงกับการเติบโตที่ผ่านมา" if g > ref - 0.03 else "ตลาดคาดหวัง **ต่ำกว่า** การเติบโตที่ผ่านมา อาจมี Margin of Safety หากธุรกิจรักษาแรงโตได้")
                st.markdown(f"เทียบกับการเติบโตที่ผ่านมา ({ref*100:.1f}%): {msg}")
            st.caption("ใช้ FCF ปีล่าสุดเทียบมูลค่าตลาด (ไม่หักหนี้สุทธิ) เป็นการประมาณแบบง่าย เหมาะกับการเปรียบเทียบเชิงเปรียบเทียบมากกว่าหามูลค่าที่แท้จริง")
    with rb:
        st.markdown("#### 🎲 ตารางสถานการณ์: ผลตอบแทนต่อปี (3 ปี)")
        feps = A.f(info.get("forwardEps"))
        base_pe = va["fpe"] if A.ok(va["fpe"]) and va["fpe"] > 0 else (va["pe"] if A.ok(va["pe"]) and va["pe"] > 0 else None)
        if not (A.ok(feps) and feps > 0 and base_pe):
            st.info("ต้องมี Forward EPS เป็นบวกและค่า P/E จึงจะสร้างตารางได้")
        else:
            gs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
            pes = [round(base_pe * m_, 1) for m_ in (0.6, 0.8, 1.0, 1.2)]
            z_ = [[((feps * (1 + g_) ** 2 * pe_) / price) ** (1 / 3) - 1 for pe_ in pes] for g_ in gs]
            fig = go.Figure(go.Heatmap(z=z_, x=[f"P/E {p_}" for p_ in pes], y=[f"EPS โต {int(g_*100)}%" for g_ in gs], text=[[f"{v*100:.0f}%" for v in r_] for r_ in z_],
                                       texttemplate="%{text}", colorscale=[[0, "#7F1D1D"], [.5, "#1E293B"], [1, "#065F46"]], zmid=0, showscale=False))
            fig.update_layout(**PLOT, height=330, title="EPS โตปีละ × P/E ตอนจบปีที่ 3")
            st.plotly_chart(fig, **stretch(st.plotly_chart))
            st.caption(f"คำนวณจาก Forward EPS {sym}{feps:.2f}, ราคาปัจจุบัน {sym}{price:,.2f}, ไม่รวมเงินปันผล เป็นสถานการณ์สมมติ ไม่ใช่การพยากรณ์")

# ====================== 5. risk lab ======================
with tabs[4]:
    k = st.columns(6)
    for col_, (l_, v_, s_) in zip(k, [("Volatility (5Y)", A.pct(rk["vol"]), ""), ("Beta", A.num(rk["beta"], 2), f"vs {bundle['bench_symbol']}"), ("Sharpe", A.num(rk["sharpe"], 2), "rf 4%"),
                                       ("Max Drawdown", A.pct(rk["max_dd"]), f"ตอนนี้ {A.pct(rk['cur_dd'], 0)}"), ("VaR 95% (1 วัน)", A.pct(rk["var95"]), "ขาดทุนวันที่แย่"), ("CVaR 95%", A.pct(rk["cvar95"]), "เฉลี่ยวันที่แย่สุด 5%")]):
        with col_: kpi(l_, v_, s_)
    st.write("")
    vf = ctx["volf"]
    va_, vb_ = st.columns([1, 1.3])
    with va_:
        st.markdown("#### 🤖 โมเดล ML: คาดความผันผวน 63 วันข้างหน้า")
        if vf:
            col = {"Low": "#22D3A6", "Medium": "#FBBF24", "High": "#F43F5E"}[vf["cls"]]
            kpi("ระดับความเสี่ยงที่คาด", {"Low": "ต่ำ", "Medium": "ปานกลาง", "High": "สูง"}[vf["cls"]] + f" · {vf['pred']*100:.0f}%", f"ช่วง 80%: {vf['lo']*100:.0f}% – {vf['hi']*100:.0f}% ต่อปี", col)
            st.caption(f"เทียบ: ความผันผวน 1 เดือนล่าสุด {vf['vol21']*100:.0f}%, 1 ปีล่าสุด {vf['anchor']*100:.0f}%. เกณฑ์ Low/Medium/High มาจาก 33/67 เปอร์เซนไทล์ของข้อมูลฝึก ({vf['metrics']['low_threshold']*100:.0f}% / {vf['metrics']['high_threshold']*100:.0f}%)")
            st.info("โมเดลนี้ทำนายระดับความผันผวน ไม่ใช่ทิศทางราคา ดูผล backtest ที่แท้จริงในแท็บ Model Card")
        else:
            st.info("ข้อมูลย้อนหลังไม่พอสำหรับโมเดล (ต้องมีราคาอย่างน้อย ~1 ปี)")
    with vb_:
        fig = go.Figure(go.Scatter(x=rk["dd_series"].index, y=rk["dd_series"] * 100, fill="tozeroy", line=dict(color="#F43F5E", width=1.2), fillcolor="rgba(244,63,94,.18)"))
        fig.update_layout(**PLOT, height=300, title="Drawdown จากจุดสูงสุด (%)")
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    r1, r2 = st.columns([1, 1.5])
    with r1:
        rr = rk["returns"] * 100
        fig = go.Figure(go.Histogram(x=rr, nbinsx=70, marker_color="#22D3EE", opacity=.8))
        fig.add_vline(x=-rk["var95"] * 100, line=dict(color="#F43F5E", dash="dash"), annotation_text="VaR 95%")
        fig.update_layout(**PLOT, height=360, title="การกระจายผลตอบแทนรายวัน (%)", showlegend=False)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    with r2:
        mc = get_mc(ticker, mu)
        fut = pd.bdate_range(h.index[-1] + pd.Timedelta(days=1), periods=mc["q"].shape[1])
        hh = h["Close"].tail(252)
        fig = go.Figure()
        fig.add_scatter(x=hh.index, y=hh.values, name="ราคาจริง", line=dict(color="#E6ECFA", width=1.5))
        fig.add_scatter(x=fut, y=mc["q"][4], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fut, y=mc["q"][0], fill="tonexty", fillcolor="rgba(139,92,246,.14)", line=dict(width=0), name="90% band")
        fig.add_scatter(x=fut, y=mc["q"][3], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fut, y=mc["q"][1], fill="tonexty", fillcolor="rgba(34,211,238,.20)", line=dict(width=0), name="50% band")
        fig.add_scatter(x=fut, y=mc["q"][2], name="Median", line=dict(color="#22D3EE", width=2))
        fig.update_layout(**PLOT, height=360, title=f"Monte Carlo 1 ปี (4,000 เส้นทาง, drift {mu*100:.0f}%/ปี)")
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    k = st.columns(4)
    with k[0]: kpi("โอกาสขาดทุน (1 ปี)", f"{mc['p_loss']*100:.0f}%")
    with k[1]: kpi("โอกาสขาดทุน > 20%", f"{mc['p_loss20']*100:.0f}%", "", "#F43F5E")
    with k[2]: kpi("โอกาสกำไร > 30%", f"{mc['p_gain30']*100:.0f}%", "", "#22D3A6")
    with k[3]: kpi("ช่วง 90% ของผลตอบแทน", f"{mc['p5']*100:.0f}% ถึง {mc['p95']*100:+.0f}%")
    st.caption("Monte Carlo สุ่มผลตอบแทนรายวันจากพฤติกรรม 3 ปีล่าสุดของหุ้นนี้ แต่ปรับค่าเฉลี่ยเป็น drift ที่คุณตั้งไว้ จึงสะท้อน 'ความกว้างของความเป็นไปได้' มากกว่าการทำนาย")

# ====================== 6. street & news ======================
with tabs[5]:
    sv, ev = ctx["street"], ctx["earn"]
    k = st.columns(4)
    with k[0]: kpi("Analyst Rating", (A.num(sv["mean_rating"], 2) + " / 5") if A.ok(sv["mean_rating"]) else "N/A", f"{int(sv['n_analysts'])} นักวิเคราะห์" if A.ok(sv["n_analysts"]) else "")
    with k[1]: kpi("ราคาเป้าหมายเฉลี่ย", f"{sym}{sv['target_mean']:,.2f}" if A.ok(sv["target_mean"]) else "N/A", f"Upside {A.pct(sv['upside'])}" if A.ok(sv["upside"]) else "")
    with k[2]: kpi("งบถัดไป", ev["next"].strftime("%d %b %Y") if ev["next"] is not None else "N/A", f"อีก {(ev['next'] - pd.Timestamp.now().normalize()).days} วัน" if ev["next"] is not None else "")
    with k[3]: kpi("EPS Surprise เฉลี่ย 4Q", A.pct(ev["avg_surprise"]))
    st.write("")
    sa, sb = st.columns(2)
    with sa:
        if sv["counts"]:
            cn = sv["counts"]
            fig = go.Figure()
            for key, lab, colr in (("strongBuy", "Strong Buy", "#059669"), ("buy", "Buy", "#22D3A6"), ("hold", "Hold", "#FBBF24"), ("sell", "Sell", "#FB923C"), ("strongSell", "Strong Sell", "#F43F5E")):
                fig.add_bar(y=["ปัจจุบัน"], x=[cn[key]], name=lab, orientation="h", marker_color=colr, text=[cn[key]], textposition="inside")
            fig.update_layout(**PLOT, height=200, barmode="stack", title="สัดส่วนคำแนะนำนักวิเคราะห์")
            st.plotly_chart(fig, **stretch(st.plotly_chart))
        if A.ok(sv["target_lo"]) and A.ok(sv["target_hi"]):
            fig = go.Figure()
            fig.add_scatter(x=[sv["target_lo"], sv["target_hi"]], y=[0, 0], mode="lines", line=dict(color="rgba(143,160,192,.5)", width=10), showlegend=False)
            fig.add_scatter(x=[sv["target_mean"]], y=[0], mode="markers", marker=dict(size=16, color="#22D3EE"), name="เป้าเฉลี่ย")
            fig.add_scatter(x=[price], y=[0], mode="markers", marker=dict(size=16, color="#FBBF24", symbol="diamond"), name="ราคาปัจจุบัน")
            fig.update_layout(**PLOT, height=170, title="ช่วงราคาเป้าหมาย (ต่ำสุด–สูงสุด)", yaxis=dict(visible=False))
            st.plotly_chart(fig, **stretch(st.plotly_chart))
    with sb:
        if ev["surprises"]:
            xs = [f"{i:%b %y}" for i, _ in ev["surprises"]]; ys = [v for _, v in ev["surprises"]]
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=["#22D3A6" if y >= 0 else "#F43F5E" for y in ys]))
            fig.update_layout(**PLOT, height=380, title="EPS Surprise (%) ย้อนหลัง")
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, **stretch(st.plotly_chart))
    st.markdown("#### 📰 ข่าวล่าสุด")
    if ctx["news"]:
        avg = float(np.mean([n["tone"] for n in ctx["news"]]))
        st.caption(f"Tone เฉลี่ย {avg:+.2f} (ประเมินจากคำในพาดหัว/สรุปข่าวด้วยพจนานุกรมคำอย่างง่าย ไม่ใช่ความเห็นเชิงลึก)")
        for n in ctx["news"]:
            t_ = n["tone"]; kind = "ok" if t_ > .15 else "bad" if t_ < -.15 else ""
            title = f'<a href="{n["link"]}" target="_blank" style="color:#E6ECFA;text-decoration:none">{n["title"]}</a>' if n["link"] else n["title"]
            html(f'<div class="pt {"g" if kind == "ok" else "r" if kind == "bad" else ""}">{title}<br><span class="sub">{n["source"]} · {str(n["date"])[:10]}</span> {chip("เชิงบวก" if kind == "ok" else "เชิงลบ" if kind == "bad" else "กลาง", kind)}</div>')
    else:
        st.info("ไม่พบข่าวล่าสุด")

# ====================== 7. peers ======================
with tabs[6]:
    sector = info.get("sector")
    if sector not in data.SECTOR_MAP:
        st.info("การเปรียบเทียบคู่แข่งรองรับกลุ่มหุ้นสหรัฐที่อยู่ในฐานข้อมูลของแอปเท่านั้น (ตอนนี้ยังไม่รองรับหุ้นนอกฐาน/หุ้นไทย)")
    else:
        with st.spinner("กำลังดึงข้อมูลคู่แข่ง..."):
            pt = data.peer_table(ticker, sector)
        med = pt.iloc[1:][["Fwd P/E", "PEG", "P/S", "Net Margin", "ROE", "Rev Growth", "Beta"]].apply(pd.to_numeric, errors="coerce").median()
        me = pt.iloc[0]
        st.markdown(f"#### {ticker} เทียบคู่แข่งในกลุ่ม {sector}")
        k = st.columns(4)
        for col_, (lab, key, lower_better, fmt) in zip(k, [("Forward P/E", "Fwd P/E", True, A.num), ("Net Margin", "Net Margin", False, A.pct), ("ROE", "ROE", False, A.pct), ("Revenue Growth", "Rev Growth", False, A.pct)]):
            mv = A.f(me[key]); md = A.f(med[key])
            if A.ok(mv) and A.ok(md) and md != 0:
                better = (mv < md) if lower_better else (mv > md)
                with col_: kpi(lab, fmt(mv), f"มัธยฐานคู่แข่ง {fmt(md)}", "#22D3A6" if better else "#FB923C")
            else:
                with col_: kpi(lab, "N/A")
        show = pt.copy()
        for c_ in ("Net Margin", "ROE", "Rev Growth"): show[c_] = show[c_].map(A.pct)
        for c_ in ("Fwd P/E", "PEG", "P/S", "Beta"): show[c_] = show[c_].map(lambda v: A.num(A.f(v), 2))
        show["Market Cap"] = show["Market Cap"].map(lambda v: A.big(A.f(v)))
        st.dataframe(show, hide_index=True, **stretch(st.dataframe))
        fp = pd.to_numeric(pt["Fwd P/E"], errors="coerce")
        if fp.notna().sum() >= 2:
            fig = go.Figure(go.Bar(x=pt["Ticker"], y=fp, marker_color=["#22D3EE"] + ["#475569"] * (len(pt) - 1)))
            fig.update_layout(**PLOT, height=300, title="Forward P/E เทียบคู่แข่ง")
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, **stretch(st.plotly_chart))
        st.caption("สีเขียว = ดีกว่ามัธยฐานของคู่แข่ง สีส้ม = ด้อยกว่า (P/E ยิ่งต่ำยิ่งถูก) คู่แข่งคัดจากหุ้นในกลุ่มเดียวกันที่อยู่ในฐานข้อมูลของแอป")

# ====================== 8. model card ======================
with tabs[7]:
    pack = A._vol_pack()
    st.markdown("#### 🧪 Model Card — โมเดลคาดความผันผวน")
    if pack is None:
        st.warning("ไม่พบไฟล์ universal_vol_model.joblib กรุณารัน train_universal_model.py")
    else:
        m = pack["metrics"]
        df = pd.DataFrame({"ตัวชี้วัด (Hold-out)": ["R²", "MAE (ความผันผวนรายปี)", "Accuracy 3 ระดับ", "Macro F1"],
                           "โมเดล StockSense": [f"{m['r2_model']:.3f}", f"{m['mae_model']:.4f}", f"{m['acc_model']:.3f}", f"{m['f1_model']:.3f}"],
                           "Baseline (ใช้ Vol 1 ปีย้อนหลัง)": [f"{m['r2_baseline']:.3f}", f"{m['mae_baseline']:.4f}", f"{m['acc_baseline']:.3f}", f"{m['f1_baseline']:.3f}"]})
        st.dataframe(df, hide_index=True, **stretch(st.dataframe))
        st.markdown(f"""
**อ่านผลอย่างตรงไปตรงมา:** โมเดลให้ความแม่นยำ **ใกล้เคียง** baseline ที่ใช้ความผันผวนย้อนหลังตรง ๆ (Accuracy {m['acc_model']:.1%} เทียบ {m['acc_baseline']:.1%}) และผิดพลาดเฉลี่ยต่ำกว่าเล็กน้อย
จึงถือเป็นตัวช่วยประกอบหนึ่งในหลายสิบตัวชี้วัด ไม่ใช่ตัวตัดสิน ระบบจึงให้น้ำหนักกับ Forecast ในมิติ Safety เพียงหนึ่งในเจ็ดตัวชี้วัด

**ออกแบบอย่างไร**
- ฝึกด้วยหุ้นสหรัฐ {m['n_tickers_train']} ตัว ({m['train_rows']:,} ตัวอย่าง) ทดสอบกับช่วงเวลาที่ไม่เคยเห็น (เริ่ม {m['test_start']}) โดยเว้น 63 วันกันข้อมูลรั่ว
- ไม่ใช้ชื่อหุ้นหรือราคาดิบเป็นฟีเจอร์ จึงใช้ได้กับหุ้นตัวอื่นนอกชุดฝึก แต่ความแม่นยำนอกชุดฝึกยังไม่ได้วัด
- ทำนายอัตราส่วนความผันผวนอนาคตต่อความผันผวน 1 ปี จึงไม่ติดระดับความผันผวนของช่วงฝึก

**ข้อจำกัดที่ควรรู้**
- คะแนน 6 มิติใช้เกณฑ์ตายตัวเชิงหลักการ ยังไม่ได้ backtest ว่าคะแนนสูงให้ผลตอบแทนดีกว่าตลาดจริงหรือไม่
- ข้อมูลจาก Yahoo Finance อาจล่าช้า ผิดพลาด หรือขาดบางรายการ โดยเฉพาะหุ้นนอกสหรัฐ
- ไม่รวมปัจจัยมหภาค ข่าวเชิงลึก ธุรกรรมผู้บริหาร หรือราคาออปชัน
- ชุดฝึกเป็นหุ้นใหญ่ที่รอดมาจนปัจจุบัน (Survivorship bias)
""")

# ====================== 9. AI Analyst ======================
st.divider()
st.subheader("💬 สอบถาม AI Analyst")

# รวบรวมข้อมูลสำหรับ Context (ใช้ข้อมูลที่ดึงมาแล้วในหน้าเว็บ)
try:
    context_data = f"""
    ชื่อหุ้น: {name} ({ticker})
    ราคาปัจจุบัน: {sym}{price:,.2f} (เปลี่ยนแปลง {chg*100:+.2f}%)
    คะแนนรวม: {'—' if not A.ok(sc['overall']) else format(sc['overall'], '.0f')}/100 ({vlabel})
    
    มิติทั้ง 6:
    """
    for p, th in A.PILLARS.items():
        v = sc["pillars"][p]
        context_data += f"- {th}: {'ไม่มีข้อมูล' if v['score'] is None else format(v['score'], '.0f')}\n"
    
    context_data += f"\nรายละเอียดอื่นๆ:\n{brief}\n"
    context_data += f"\nความผันผวนที่คาดการณ์ (ML Model): {'ไม่มีข้อมูล' if not ctx['volf'] else ctx['volf']['cls']}"

except Exception:
    context_data = "ไม่สามารถดึงข้อมูลสรุปได้ กรุณาตรวจสอบข้อมูลหุ้นอีกครั้ง"

# แสดง UI ช่องแชทและประมวลผล
if user_question := st.chat_input(f"พิมพ์คำถามเกี่ยวกับ {ticker} ที่นี่..."):
    with st.chat_message("user"):
        st.write(user_question)
        
    with st.chat_message("assistant"):
        with st.spinner("AI กำลังวิเคราะห์..."):
            answer = ai_assistant.ask_ai(question=user_question, stock_context=context_data)
            st.write(answer)

st.divider()
st.caption("StockSense AI · ข้อมูลเพื่อการศึกษาและประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน ผลในอดีตไม่รับประกันอนาคต · แหล่งข้อมูล: Yahoo Finance (อาจล่าช้า)")
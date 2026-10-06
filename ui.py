"""UI helpers: CSS ธีม futuristic + HTML components"""
import inspect
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#060A14;--card:rgba(17,26,46,.62);--line:rgba(120,150,220,.16);--cy:#22D3EE;--vi:#8B5CF6;--gr:#22D3A6;--rd:#F43F5E;--am:#FBBF24;--tx:#E6ECFA;--mu:#8FA0C0}
html,body,.stApp{font-family:'Space Grotesk',sans-serif!important}
.stApp{background:radial-gradient(1200px 600px at 10% -10%,rgba(139,92,246,.16),transparent 60%),radial-gradient(1000px 500px at 100% 0%,rgba(34,211,238,.12),transparent 55%),var(--bg);color:var(--tx)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(120,150,220,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(120,150,220,.045) 1px,transparent 1px);background-size:44px 44px;mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%)}
.block-container{max-width:1400px;padding-top:1.4rem;padding-bottom:3rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:rgba(9,14,28,.92);border-right:1px solid var(--line)}
h1,h2,h3,h4{color:#F4F7FF!important;letter-spacing:-.3px;font-weight:600!important}
p,label,li,span{color:var(--tx)}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--mu)!important}
hr{border-color:var(--line)}
.glass{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 20px;backdrop-filter:blur(14px);box-shadow:0 10px 40px rgba(0,0,0,.28)}
.glass.glow{box-shadow:0 0 0 1px rgba(34,211,238,.18),0 10px 50px rgba(34,211,238,.08)}
.mono{font-family:'JetBrains Mono',monospace}
.tick{font-size:3rem;font-weight:700;line-height:1;background:linear-gradient(90deg,var(--cy),var(--vi));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mu);font-size:.92rem}
.price{font-size:2.2rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.up{color:var(--gr)}.dn{color:var(--rd)}
.chip{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;border:1px solid var(--line);color:var(--mu);margin:2px 4px 2px 0}
.chip.ok{color:var(--gr);border-color:rgba(34,211,166,.4);background:rgba(34,211,166,.08)}
.chip.bad{color:var(--rd);border-color:rgba(244,63,94,.4);background:rgba(244,63,94,.08)}
.chip.warn{color:var(--am);border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.08)}
.ring{--p:50;--c:var(--cy);width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.07) 0);position:relative;animation:spin 1.2s ease-out}
.ring:before{content:"";position:absolute;inset:11px;border-radius:50%;background:#0A1122}
.ring b{position:relative;font-size:2.4rem;font-family:'JetBrains Mono',monospace}
.ring small{position:relative;display:block;text-align:center;color:var(--mu);font-size:.7rem;margin-top:-6px}
@keyframes spin{from{--p:0;opacity:.3}to{opacity:1}}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;height:100%}
.kpi .l{color:var(--mu);font-size:.78rem;text-transform:uppercase;letter-spacing:.8px}
.kpi .v{font-size:1.55rem;font-weight:600;font-family:'JetBrains Mono',monospace;margin-top:2px}
.kpi .s{color:var(--mu);font-size:.78rem;margin-top:2px}
.bar{height:8px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden}
.bar>i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--vi),var(--cy))}
.prow{display:flex;justify-content:space-between;font-size:.88rem;margin:10px 0 4px}
.pt{border-left:3px solid var(--line);padding:6px 12px;margin:7px 0;border-radius:0 10px 10px 0;background:rgba(255,255,255,.03);font-size:.9rem}
.pt.g{border-color:var(--gr)}.pt.r{border-color:var(--rd)}.pt.a{border-color:var(--am)}
.rng{position:relative;height:6px;border-radius:9px;background:linear-gradient(90deg,rgba(244,63,94,.5),rgba(251,191,36,.5),rgba(34,211,166,.5));margin:12px 0 4px}
.rng i{position:absolute;top:-5px;width:4px;height:16px;border-radius:3px;background:#fff;box-shadow:0 0 10px #fff}
.feat{min-height:150px}
.feat h4{margin:.2rem 0 .4rem 0;font-size:1.05rem}
.feat p{color:var(--mu);font-size:.88rem;margin:0}
.stButton>button,.stFormSubmitButton>button{background:linear-gradient(90deg,#2563EB,#7C3AED);color:#fff;border:0;border-radius:11px;font-weight:600}
.stButton>button:hover,.stFormSubmitButton>button:hover{filter:brightness(1.12);color:#fff;border:0}
.stTextInput input{background:#0E1730;border:1px solid var(--line);border-radius:11px;color:#fff;font-family:'JetBrains Mono',monospace;letter-spacing:1px}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:10px 10px 0 0;padding:8px 14px;color:var(--mu)}
.stTabs [aria-selected="true"]{color:#fff!important;background:rgba(34,211,238,.10)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line);border-radius:14px}
</style>
"""


def H(s: str) -> str:
    """ตัด indent เพื่อไม่ให้ Markdown มองเป็น code block"""
    return "\n".join(l.strip() for l in s.strip().splitlines())


def html(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


def kpi(label, value, sub="", color=""):
    st.markdown(H(f'<div class="kpi"><div class="l">{label}</div><div class="v" style="color:{color or "var(--tx)"}">{value}</div><div class="s">{sub}</div></div>'), unsafe_allow_html=True)


def chip(text, kind=""):
    return f'<span class="chip {kind}">{text}</span>'


def bar(pct_, label, value_txt):
    return (f'<div class="prow"><span>{label}</span><span class="mono">{value_txt}</span></div>'
            f'<div class="bar"><i style="width:{max(2, min(100, pct_)):.0f}%"></i></div>')


def stretch(fn):
    """รองรับทั้ง Streamlit เก่า/ใหม่ (use_container_width vs width='stretch')"""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif", color="#C9D4EE"), margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.08, x=0))
GRID = dict(gridcolor="rgba(120,150,220,.10)", zeroline=False)

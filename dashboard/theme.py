"""Design system for the dashboard.

A dense dark analytics console: hairline borders, one accent, JetBrains Mono for
every number, tight aligned rows over big cards, real horizontal bar charts.

`inject()` writes the stylesheet once. Pure-string helpers (`pill`, `delta`, `mono`,
`meter`) return markup for embedding; `render`/section helpers write to the page.
"""

from __future__ import annotations

import html as _html
from typing import Callable, Iterable, Optional, Sequence

import streamlit as st

ACCENT = "#7C8CF8"
STAGE_COLORS = {
    "embedding": "#7C8CF8", "retrieval": "#4CC9C0", "reranking": "#8A909B",
    "generation": "#E0A458", "evaluation": "#C77DE0", "total": "#8A909B",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#08090B; --bg-1:#0D0E11; --panel:#111318; --panel-2:#161920; --panel-3:#1B1F27;
  --line:rgba(255,255,255,.06); --line-2:rgba(255,255,255,.11); --line-3:rgba(255,255,255,.17);
  --ink:#EDEEF1; --ink-2:#A2A7B2; --ink-3:#666B76; --ink-4:#474C56;
  --accent:#7C8CF8; --accent-2:#4CC9C0; --accent-ink:#B7C0FF;
  --pos:#3FBE6B; --pos-ink:#79DA9C; --neg:#E5544B; --neg-ink:#F0A49E; --warn:#D9A441; --warn-ink:#ECC784;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --r:11px; --r-sm:8px; --r-xs:6px;
}

/* shell ------------------------------------------------------------------------ */
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:var(--bg);}
.stApp{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink);
  -webkit-font-smoothing:antialiased;font-feature-settings:'cv05','ss01';}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stDeployButton"],#MainMenu,footer{display:none!important;}
[data-testid="stMainBlockContainer"]{max-width:1180px;padding:1.6rem 2.2rem 5rem;}
[data-testid="stMain"]{background:var(--bg);}
[data-testid="stVerticalBlock"]{gap:.55rem;}
hr{border-color:var(--line);}

/* typography ----------------------------------------------------------------------- */
[data-testid="stMarkdownContainer"] p{color:var(--ink-2);font-size:13px;line-height:1.55;}
a,a:visited{color:var(--accent-ink);text-decoration:none;} a:hover{text-decoration:underline;}
code,kbd,pre{font-family:var(--mono)!important;}
[data-testid="stCode"]{border:1px solid var(--line);border-radius:var(--r-sm);background:var(--bg-1)!important;}
[data-testid="stCode"] pre{background:transparent!important;font-size:12px!important;line-height:1.6;}

/* sidebar --------------------------------------------------------------------------- */
[data-testid="stSidebar"]{background:var(--bg-1);border-right:1px solid var(--line);width:230px!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:1.25rem .85rem 1rem;}
.brand{display:flex;align-items:center;gap:11px;padding:2px 4px 20px;}
.brand .mk{width:30px;height:30px;border-radius:9px;flex-shrink:0;color:#0B0C10;
  background:linear-gradient(145deg,#93A0FF,#6E7CF1 52%,#5A67DF);
  display:grid;place-items:center;
  box-shadow:0 0 0 1px rgba(124,140,248,.4),0 8px 18px -8px rgba(124,140,248,.55),inset 0 1px 0 rgba(255,255,255,.28);}
.brand .mk svg{width:19px;height:19px;display:block;}
.brand .nm{font-weight:650;font-size:15px;letter-spacing:-.015em;line-height:1.12;color:var(--ink);}
.brand .nm span{display:block;font-weight:450;font-size:9.5px;color:var(--ink-3);letter-spacing:.02em;margin-top:2px;}
.s-lbl{font-size:9.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-4);margin:16px 4px 6px;}
.s-foot{margin:14px 4px 0;font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;color:var(--ink-4);
  display:flex;align-items:center;gap:6px;}
.s-foot::before{content:"";width:5px;height:5px;border-radius:99px;background:var(--accent);opacity:.7;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:1px;}
[data-testid="stSidebar"] [role="radiogroup"]>label{padding:6px 9px;border-radius:7px;margin:0;border:1px solid transparent;cursor:pointer;transition:background .1s;}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover{background:rgba(255,255,255,.03);}
[data-testid="stSidebar"] [role="radiogroup"]>label>div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] div[data-testid="stMarkdownContainer"] p{
  font-size:13px!important;font-weight:450!important;color:var(--ink-2)!important;}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked){background:rgba(124,140,248,.13);border-color:rgba(124,140,248,.26);}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked) div[data-testid="stMarkdownContainer"] p{color:var(--accent-ink)!important;font-weight:550!important;}
.mini{display:flex;flex-direction:column;gap:5px;margin:8px 4px 4px;padding:10px;border:1px solid var(--line);border-radius:var(--r-sm);background:var(--panel);}
.mini .mrow{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--ink-3);}
.mini .mrow b{font-family:var(--mono);font-weight:500;color:var(--ink);font-size:11.5px;}

/* form controls ---------------------------------------------------------------------- */
label p,[data-testid="stWidgetLabel"] p{color:var(--ink-3)!important;font-size:10.5px!important;font-weight:600!important;letter-spacing:.05em;text-transform:uppercase;}
[data-baseweb="select"]>div,[data-baseweb="base-input"]{background:var(--panel)!important;border-color:var(--line-2)!important;border-radius:var(--r-sm)!important;font-size:12.5px!important;}
[data-baseweb="select"]>div:hover{border-color:var(--accent)!important;}
[data-baseweb="popover"] [role="listbox"]{background:var(--panel)!important;border:1px solid var(--line-2)!important;}
[data-baseweb="tag"]{background:rgba(124,140,248,.16)!important;border-radius:5px!important;}
.stButton>button{background:var(--panel-2);border:1px solid var(--line-2);color:var(--ink-2);border-radius:var(--r-sm);font-weight:500;font-size:12px;padding:.32rem .75rem;transition:all .1s;}
.stButton>button:hover{border-color:var(--accent);color:var(--accent-ink);}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{border-color:var(--accent)!important;}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:var(--r-sm)!important;background:var(--panel)!important;}
[data-testid="stExpander"] summary{font-size:12px;color:var(--ink-2);}
[data-testid="stAlert"]{border-radius:var(--r-sm);border:1px solid var(--line-2);background:var(--panel)!important;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{color:var(--ink)!important;}

/* topbar --------------------------------------------------------------------------- */
.tb{display:flex;align-items:baseline;justify-content:space-between;gap:16px;padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:16px;}
.tb .h{font-size:21px;font-weight:600;letter-spacing:-.02em;}
.tb .h small{font-size:10px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-4);margin-right:9px;}
.tb .rt{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--ink-3);flex-shrink:0;}
.dot{width:6px;height:6px;border-radius:99px;background:var(--pos);box-shadow:0 0 0 3px rgba(63,190,107,.15);}
.mc{font-family:var(--mono);font-size:11px;color:var(--ink-2);background:var(--panel-2);border:1px solid var(--line-2);padding:3px 8px;border-radius:5px;}

/* vitals strip ------------------------------------------------------------------- */
.vitals{display:flex;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;margin-bottom:18px;}
.vt{flex:1;min-width:0;min-height:56px;background:var(--panel);padding:8px 12px 9px;display:flex;flex-direction:column;gap:5px;}
.vt .l{font-size:9px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-4);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.vt .v{font-family:var(--mono);font-size:15px;font-weight:500;color:var(--ink);letter-spacing:-.02em;line-height:1;font-feature-settings:'tnum';}
.vt .v.dim{color:var(--ink-4);}

/* section header ----------------------------------------------------------------- */
.sec{display:flex;align-items:baseline;gap:10px;margin:22px 0 10px;}
.sec:first-of-type{margin-top:2px;}
.sec h2{font-size:13px;font-weight:600;letter-spacing:.01em;margin:0;}
.sec .eb{font-size:9.5px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-4);}
.sec .ds{font-size:11.5px;color:var(--ink-3);}

/* stat list --------------------------------------------------------------------------- */
.sl{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--panel);}
.sl-h{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-4);padding:9px 14px;border-bottom:1px solid var(--line);background:var(--bg-1);}
.sl-row{display:grid;grid-template-columns:1fr 52px 58px auto;align-items:center;gap:11px;padding:6px 14px;border-bottom:1px solid var(--line);}
.sl-row:last-child{border-bottom:0;}
.sl-row:hover{background:rgba(255,255,255,.018);}
.sl-l{font-size:12px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sl-v{font-family:var(--mono);font-size:12.5px;font-weight:500;color:var(--ink);text-align:right;font-feature-settings:'tnum';}
.sl-v.dim{color:var(--ink-4);}
.sl-d{font-family:var(--mono);font-size:11px;font-weight:500;text-align:right;}

/* meter -------------------------------------------------------------------------------- */
.mtr{position:relative;height:4px;border-radius:99px;background:var(--line-2);overflow:hidden;display:block;}
.mtr>i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:var(--accent);}
.mtr.good>i{background:#43B36C;} .mtr.warn>i{background:var(--warn);} .mtr.bad>i{background:var(--neg);}
.mtr.t2>i{background:var(--accent-2);}
.mtr-i{display:inline-block;width:44px;vertical-align:middle;}

/* hbar chart ------------------------------------------------------------------------ */
.hbc{border:1px solid var(--line);border-radius:var(--r);background:var(--panel);padding:12px 14px;}
.hbc-h{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-4);margin-bottom:10px;}
.hbc-row{display:grid;grid-template-columns:82px 1fr 68px;align-items:center;gap:10px;padding:4px 0;}
.hbc-l{font-size:11.5px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hbc-trk{height:16px;border-radius:4px;background:rgba(255,255,255,.04);position:relative;}
.hbc-trk>i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;min-width:2px;}
.hbc-v{font-family:var(--mono);font-size:11.5px;color:var(--ink);text-align:right;font-feature-settings:'tnum';}

/* donut ----------------------------------------------------------------------------- */
.dn{border:1px solid var(--line);border-radius:var(--r);background:var(--panel);padding:14px;display:flex;gap:16px;align-items:center;}
.dn .lg{display:flex;flex-direction:column;gap:6px;font-size:11.5px;color:var(--ink-2);}
.dn .lg span{display:flex;align-items:center;gap:7px;}
.dn .lg i{width:8px;height:8px;border-radius:2px;flex-shrink:0;}
.dn .lg b{font-family:var(--mono);color:var(--ink);font-weight:500;margin-left:auto;padding-left:14px;}

/* kv --------------------------------------------------------------------------------- */
.kv{border:1px solid var(--line);border-radius:var(--r);background:var(--panel);overflow:hidden;}
.kv-h{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-4);padding:9px 14px;border-bottom:1px solid var(--line);background:var(--bg-1);}
.kv-row{display:flex;justify-content:space-between;gap:12px;padding:6px 14px;border-bottom:1px solid var(--line);font-size:12px;}
.kv-row:last-child{border-bottom:0;}
.kv-row .k{color:var(--ink-3);} .kv-row .v{font-family:var(--mono);color:var(--ink);font-size:11.5px;}

/* table ---------------------------------------------------------------------------- */
.rt-wrap{border:1px solid var(--line);border-radius:var(--r-sm);overflow:auto;}
table.rt{width:100%;border-collapse:separate;border-spacing:0;font-size:12px;}
table.rt th{text-align:left;font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-4);background:var(--bg-1);padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0;}
table.rt td{padding:8px 12px;border-bottom:1px solid var(--line);color:var(--ink);vertical-align:middle;}
table.rt tr:last-child td{border-bottom:0;}
table.rt tbody tr:hover td{background:rgba(255,255,255,.02);}
table.rt td.num,table.rt th.num{text-align:right;font-family:var(--mono);font-feature-settings:'tnum';}
table.rt td.mut{color:var(--ink-3);}
table.rt tr.hl td{background:rgba(124,140,248,.07);}
table.rt tr.hl td:first-child{box-shadow:inset 2px 0 0 var(--accent);font-weight:600;}

/* compare grid --------------------------------------------------------------------- */
.cmp{border:1px solid var(--line);border-radius:var(--r-sm);overflow:hidden;width:fit-content;max-width:100%;}
.cmp-fam{font-size:9.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-4);padding:9px 14px 7px;background:var(--bg-1);border-bottom:1px solid var(--line);}
.cmp-row{display:grid;align-items:center;gap:14px;padding:7px 14px;border-bottom:1px solid var(--line);font-size:12px;}
.cmp-row:last-child{border-bottom:0;}
.cmp-row:hover{background:rgba(255,255,255,.018);}
.cmp-m{color:var(--ink-2);}
.cmp-c{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-feature-settings:'tnum';color:var(--ink);}
.cmp-c .mtr-i{width:46px;}
.cmp-hd{color:var(--ink-4)!important;font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;background:var(--bg-1);}
.cmp-hd .cmp-c{color:var(--ink-4);font-family:'Inter';}

/* pills / deltas ------------------------------------------------------------------- */
.pill{display:inline-flex;align-items:center;gap:5px;font-size:9.5px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:99px;border:1px solid var(--line-2);color:var(--ink-2);background:var(--panel-2);white-space:nowrap;text-transform:uppercase;}
.pill--pos{color:var(--pos-ink);background:rgba(63,190,107,.11);border-color:rgba(63,190,107,.3);}
.pill--neg{color:var(--neg-ink);background:rgba(229,84,75,.11);border-color:rgba(229,84,75,.3);}
.pill--warn{color:var(--warn-ink);background:rgba(217,164,65,.11);border-color:rgba(217,164,65,.3);}
.pill--accent{color:var(--accent-ink);background:rgba(124,140,248,.12);border-color:rgba(124,140,248,.32);}
.dl{font-family:var(--mono);font-size:11px;font-weight:500;white-space:nowrap;}
.dl--up{color:var(--pos-ink);} .dl--down{color:var(--neg-ink);} .dl--flat{color:var(--ink-4);}

/* cards ----------------------------------------------------------------------------- */
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:13px 15px;}
.card h4{margin:0 0 7px;font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:8px;}
.card .li{display:flex;gap:8px;align-items:flex-start;font-size:12px;color:var(--ink-2);margin-top:5px;line-height:1.5;}
.card .li .b{width:5px;height:5px;border-radius:99px;margin-top:6px;flex-shrink:0;background:var(--ink-4);}
.card .li .b.pos{background:var(--pos);} .card .li .b.neg{background:var(--neg);}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;}
.cols2>div{background:var(--panel);padding:13px 15px;}
.cols2 .ch{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;margin-bottom:8px;}
.cols2 .ch.pos{color:var(--pos-ink);} .cols2 .ch.neg{color:var(--neg-ink);}
.tro{border:1px solid var(--line-2);border-radius:var(--r);padding:12px 15px;margin-bottom:8px;background:var(--panel);}
.tro .t{font-size:12.5px;font-weight:600;margin-bottom:2px;}
.tro .s{font-size:11.5px;color:var(--ink-3);}

.ans{background:var(--panel);border:1px solid var(--line);border-left:2px solid var(--accent);border-radius:var(--r-sm);padding:11px 14px;font-size:13px;line-height:1.55;color:var(--ink);}
.ans.exp{border-left-color:var(--ink-4);color:var(--ink-2);}
.ans .h{font-size:9px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-4);display:block;margin-bottom:5px;}

.qlist{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--panel);}
.qrow{display:grid;grid-template-columns:56px 152px 1fr;gap:12px;align-items:baseline;padding:8px 14px;border-bottom:1px solid var(--line);font-size:12px;}
.qrow:last-child{border-bottom:0;} .qrow:hover{background:rgba(255,255,255,.018);}
.qrow .qi{font-family:var(--mono);font-size:11px;color:var(--ink-3);}
.qrow .qw{color:var(--ink-2);}

.empty{text-align:center;padding:56px 24px;border:1px dashed var(--line-2);border-radius:var(--r);}
.empty h3{color:var(--ink);font-weight:600;margin:0 0 7px;font-size:14px;}
.empty p{color:var(--ink-3);font-size:12.5px;margin:0;}
.empty code{background:var(--panel-2);padding:2px 6px;border-radius:5px;font-family:var(--mono);font-size:11.5px;color:var(--accent-ink);}

@media (max-width:900px){.vitals{flex-wrap:wrap;} .vt{flex-basis:33%;}}
"""


def inject() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def _e(v) -> str:
    return _html.escape(str(v))


# --- string helpers --------------------------------------------------------------------
def mono(v) -> str:
    return f'<span style="font-family:var(--mono);font-feature-settings:\'tnum\'">{_e(v)}</span>'


def pill(text: str, tone: str = "") -> str:
    return f'<span class="pill{" pill--" + tone if tone else ""}">{_e(text)}</span>'


def delta(text: str, direction: str) -> str:
    cls = {"improved": "up", "regressed": "down", "neutral": "flat"}.get(direction, "flat")
    arrow = {"up": "▲", "down": "▼", "flat": "–"}[cls]
    return f'<span class="dl dl--{cls}">{arrow}&nbsp;{_e(text)}</span>'


def _tone_for(frac: Optional[float]) -> str:
    if frac is None:
        return ""
    if frac >= 0.85:
        return "good"
    if frac >= 0.55:
        return ""
    if frac >= 0.35:
        return "warn"
    return "bad"


def meter(frac: Optional[float], tone: Optional[str] = None, cls: str = "mtr-i") -> str:
    if frac is None:
        return '<span style="color:var(--ink-4)">—</span>'
    pct = max(0.0, min(1.0, float(frac))) * 100
    t = tone if tone is not None else _tone_for(frac)
    return f'<span class="mtr {t} {cls}"><i style="width:{pct:.1f}%"></i></span>'


# --- render helpers -------------------------------------------------------------------
def topbar(section: str, experiment_id: Optional[str] = None, right: str = "") -> None:
    rt = (f'<span class="dot"></span><span class="mc">{_e(experiment_id)}</span>'
          if experiment_id else f'<span class="mc">{_e(right)}</span>')
    st.markdown(
        f'<div class="tb"><div class="h"><small>ASSAY</small>{_e(section)}</div>'
        f'<div class="rt">{rt}</div></div>', unsafe_allow_html=True)


def vitals(items: list[dict]) -> None:
    cells = []
    for it in items:
        v = it.get("value")
        vcls = "v dim" if v in (None, "—") else "v"
        m = f'<span class="mtr {_tone_for(it.get("frac"))}" style="margin-top:2px"><i style="width:{max(0,min(1,it.get("frac",0)))*100:.0f}%"></i></span>' if it.get("frac") is not None else ""
        d = it.get("delta", "")
        cells.append(f'<div class="vt"><span class="l">{_e(it["label"])}</span>'
                     f'<span class="{vcls}">{_e(v if v is not None else "—")}</span>{m}{d}</div>')
    st.markdown(f'<div class="vitals">{"".join(cells)}</div>', unsafe_allow_html=True)


def sec(title: str, eyebrow: str = "", desc: str = "") -> None:
    eb = f'<span class="eb">{_e(eyebrow)}</span>' if eyebrow else ""
    ds = f'<span class="ds">{_e(desc)}</span>' if desc else ""
    st.markdown(f'<div class="sec">{eb}<h2>{_e(title)}</h2>{ds}</div>', unsafe_allow_html=True)


def statlist_html(header: str, rows: list[dict]) -> str:
    body = []
    for r in rows:
        v = r.get("value")
        vcls = "sl-v dim" if v in (None, "—") else "sl-v"
        frac = r.get("frac")
        mt = meter(frac, r.get("tone")) if frac is not None else '<span style="color:var(--ink-4)">—</span>'
        d = r.get("delta", "")
        body.append(f'<div class="sl-row"><span class="sl-l">{_e(r["label"])}</span>'
                    f'<span class="{vcls}">{_e(v if v is not None else "—")}</span>'
                    f'<span>{mt}</span><span class="sl-d">{d}</span></div>')
    return f'<div class="sl"><div class="sl-h">{_e(header)}</div>{"".join(body)}</div>'


def statlist(header: str, rows: list[dict]) -> None:
    st.markdown(statlist_html(header, rows), unsafe_allow_html=True)


def hbar_html(header: str, rows: list[tuple], fmt: Callable = str,
              colors: Optional[Sequence[str]] = None, peak: Optional[float] = None) -> str:
    vals = [v for _, v in rows] or [1.0]
    hi = peak or max(vals) or 1.0
    out = []
    for i, (label, v) in enumerate(rows):
        w = max(0.0, (v / hi) * 100) if hi else 0
        c = (colors[i % len(colors)] if colors else "var(--accent)")
        out.append(f'<div class="hbc-row"><span class="hbc-l">{_e(label)}</span>'
                   f'<span class="hbc-trk"><i style="width:{w:.2f}%;background:{c}"></i></span>'
                   f'<span class="hbc-v">{_e(fmt(v))}</span></div>')
    return f'<div class="hbc"><div class="hbc-h">{_e(header)}</div>{"".join(out)}</div>'


def hbar(header: str, rows: list[tuple], fmt: Callable = str,
         colors: Optional[Sequence[str]] = None, peak: Optional[float] = None) -> None:
    st.markdown(hbar_html(header, rows, fmt, colors, peak), unsafe_allow_html=True)


def donut_html(header: str, segments: list[tuple]) -> str:
    """segments: (label, value, color)."""
    total = sum(v for _, v, _ in segments) or 1.0
    circ, off, arcs = 100.0, 25.0, []
    for _, v, c in segments:
        frac = v / total
        arcs.append(f'<circle cx="18" cy="18" r="15.915" fill="none" stroke="{c}" stroke-width="4.2" '
                    f'stroke-dasharray="{frac*circ:.2f} {circ-frac*circ:.2f}" stroke-dashoffset="{off:.2f}"/>')
        off -= frac * circ
    legend = "".join(f'<span><i style="background:{c}"></i>{_e(l)}<b>{_e(f"${v:.6f}")}</b></span>'
                     for l, v, c in segments)
    return (
        f'<div class="dn"><svg viewBox="0 0 36 36" width="88" height="88" style="transform:rotate(-90deg)">'
        f'<circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--line-2)" stroke-width="4.2"/>{"".join(arcs)}'
        f'</svg><div class="lg"><div class="hbc-h" style="margin:0 0 2px">{_e(header)}</div>{legend}</div></div>')


def donut(header: str, segments: list[tuple]) -> None:
    st.markdown(donut_html(header, segments), unsafe_allow_html=True)


def kv_html(header: str, pairs: list[tuple]) -> str:
    body = "".join(f'<div class="kv-row"><span class="k">{_e(k)}</span><span class="v">{_e(v)}</span></div>'
                   for k, v in pairs)
    return f'<div class="kv"><div class="kv-h">{_e(header)}</div>{body}</div>'


def kv(header: str, pairs: list[tuple]) -> None:
    st.markdown(kv_html(header, pairs), unsafe_allow_html=True)


def grid(cells: list[str], cols: str, gap: str = "14px") -> None:
    """One CSS-grid row from pre-rendered HTML strings — avoids st.columns quirks."""
    inner = "".join(f"<div>{c}</div>" for c in cells if c)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:{cols};gap:{gap};'
        f'align-items:start;margin:2px 0 14px">{inner}</div>', unsafe_allow_html=True)


def table(headers: list[str], rows: list[list[str]], num_cols: Iterable[int] = (),
          max_w: Optional[int] = None) -> None:
    num = set(num_cols)
    head = "".join(f'<th class="num">{_e(h)}</th>' if i in num else f"<th>{_e(h)}</th>"
                   for i, h in enumerate(headers))
    body = []
    for r in rows:
        hl = ' class="hl"' if r and r[0] == "__HL__" else ""
        cells = r[1:] if hl else r
        tds = "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
                      for i, c in enumerate(cells))
        body.append(f"<tr{hl}>{tds}</tr>")
    style = f' style="max-width:{max_w}px"' if max_w else ""
    st.markdown(f'<div class="rt-wrap"{style}><table class="rt"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


def compare(family_blocks: list[tuple], baseline: str, others: list[str]) -> None:
    """family_blocks: (family_name, [row dicts]) where a row dict has
    label, base:(text, frac), cols:[(text, frac, delta_html)]."""
    n = 1 + len(others)
    grid = f"grid-template-columns:160px repeat({n},minmax(150px,224px))"
    head = (f'<div class="cmp-row cmp-hd" style="{grid}"><span></span>'
            f'<span class="cmp-c">{_e(baseline)}·base</span>'
            + "".join(f'<span class="cmp-c">{_e(o)}</span>' for o in others) + "</div>")
    blocks = [head]
    for fam, rows in family_blocks:
        blocks.append(f'<div class="cmp-fam">{_e(fam)}</div>')
        for r in rows:
            bt, bf = r["base"]
            bcell = f'<span class="cmp-c">{_e(bt)}{meter(bf) if bf is not None else ""}</span>'
            ocells = "".join(
                f'<span class="cmp-c">{_e(t)}{meter(f) if f is not None else ""}&nbsp;{d}</span>'
                for t, f, d in r["cols"])
            blocks.append(f'<div class="cmp-row" style="{grid}">'
                          f'<span class="cmp-m">{_e(r["label"])}</span>{bcell}{ocells}</div>')
    st.markdown(f'<div class="cmp">{"".join(blocks)}</div>', unsafe_allow_html=True)


def card(title_html: str, lines: list[tuple]) -> None:
    body = "".join(f'<div class="li"><span class="b {t}"></span><span>{_e(x)}</span></div>' for t, x in lines)
    st.markdown(f'<div class="card"><h4>{title_html}</h4>{body}</div>', unsafe_allow_html=True)


def tradeoff(experiment_id: str, summary: str, gains: list[str], losses: list[str]) -> None:
    g = "".join(f'<div class="li"><span class="b pos"></span><span>{_e(x)}</span></div>' for x in gains) or \
        '<div class="li" style="color:var(--ink-4)">none</div>'
    l = "".join(f'<div class="li"><span class="b neg"></span><span>{_e(x)}</span></div>' for x in losses) or \
        '<div class="li" style="color:var(--ink-4)">none</div>'
    st.markdown(
        f'<div class="tro"><div class="t">{_e(experiment_id)}</div><div class="s">{_e(summary)}</div></div>'
        f'<div class="cols2"><div><div class="ch pos">Gains</div>{g}</div>'
        f'<div><div class="ch neg">Losses</div>{l}</div></div>', unsafe_allow_html=True)


def answer_block(label: str, text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ans{(" " + kind) if kind else ""}"><span class="h">{_e(label)}</span>{_e(text)}</div>',
                unsafe_allow_html=True)


def qrows(rows: list[tuple]) -> None:
    """rows: (qid_html, tag_html, text)."""
    body = "".join(f'<div class="qrow"><span class="qi">{qi}</span><span>{tg}</span>'
                   f'<span class="qw">{_e(tx)}</span></div>' for qi, tg, tx in rows)
    st.markdown(f'<div class="qlist">{body}</div>', unsafe_allow_html=True)


def empty(title: str, hint_html: str) -> None:
    st.markdown(f'<div class="empty"><h3>{_e(title)}</h3><p>{hint_html}</p></div>', unsafe_allow_html=True)


def esc(v) -> str:
    return _e(v)

"""Design system for the dashboard.

`inject()` writes the stylesheet once; the rest are small HTML component helpers.
Pure-string helpers (``pill``, ``delta``, ``bar``) return markup for embedding in
table cells; ``render_*`` helpers write to the page.

Aesthetic: layered near-black, hairline borders, one restrained accent, Inter for
UI + JetBrains Mono for every number. No gradients in the chrome, minimal shadow.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Optional

import streamlit as st

# stage -> colour, used by the trace timeline
STAGE_COLORS = {
    "embedding": "#7C8CF8",
    "retrieval": "#4CC9C0",
    "reranking": "#8A909B",
    "generation": "#E0A458",
    "evaluation": "#C77DE0",
    "total": "#8A909B",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:#0A0B0D; --bg-elev:#101114; --panel:#131519; --panel-2:#171A20;
  --line:rgba(255,255,255,.07); --line-2:rgba(255,255,255,.13);
  --ink:#ECEDEF; --ink-2:#A7ACB5; --ink-3:#6C6F79;
  --accent:#7C8CF8; --accent-ink:#BAC2FF;
  --pos:#46B26B; --pos-ink:#7FD99C; --pos-bg:rgba(70,178,107,.12);
  --neg:#E5544B; --neg-ink:#F0A49E; --neg-bg:rgba(229,84,75,.12);
  --warn:#D9A441; --warn-ink:#EBC583; --warn-bg:rgba(217,164,65,.12);
  --r:12px; --r-sm:9px;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}

/* ---- shell ---------------------------------------------------------------- */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] { background:var(--bg); }
.stApp { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--ink);
  -webkit-font-smoothing:antialiased; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stDeployButton"], #MainMenu, footer { display:none !important; }
[data-testid="stMain"] .block-container { max-width:1240px; padding:2.4rem 2rem 5rem; }
[data-testid="stMainBlockContainer"] { max-width:1240px; padding:2.4rem 2rem 5rem; }

/* ---- typography --------------------------------------------------------------- */
h1,h2,h3,h4 { font-family:'Inter',sans-serif; color:var(--ink); letter-spacing:-.02em; }
[data-testid="stMarkdownContainer"] p { color:var(--ink-2); }
a, a:visited { color:var(--accent-ink); text-decoration:none; }
a:hover { text-decoration:underline; }
code, kbd, pre { font-family:var(--mono) !important; }
[data-testid="stCode"] { border:1px solid var(--line); border-radius:var(--r-sm); background:var(--bg-elev) !important; }
[data-testid="stCode"] pre { background:transparent !important; font-size:12.5px !important; }

/* ---- sidebar --------------------------------------------------------------- */
[data-testid="stSidebar"] { background:var(--bg-elev); border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding:1.4rem 1rem 1rem; }
.brand { display:flex; align-items:center; gap:11px; padding:2px 4px 20px; }
.brand .mark { width:28px; height:28px; border-radius:8px; background:var(--accent); color:#0A0B0D;
  display:grid; place-items:center; font-weight:800; font-size:14px; letter-spacing:-.03em; }
.brand .name { font-weight:600; font-size:14.5px; letter-spacing:-.01em; }
.brand .name span { display:block; font-weight:500; font-size:11px; color:var(--ink-3); letter-spacing:.02em; }
.side-label { font-size:10.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:var(--ink-3); margin:18px 4px 8px; }

/* radio -> nav list */
[data-testid="stSidebar"] [role="radiogroup"] { gap:2px; }
[data-testid="stSidebar"] [role="radiogroup"] > label { padding:8px 10px; border-radius:8px; margin:0;
  border:1px solid transparent; cursor:pointer; transition:background .12s,border-color .12s; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background:rgba(255,255,255,.03); }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
  font-size:13.5px !important; font-weight:500 !important; color:var(--ink-2) !important; letter-spacing:0; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background:rgba(124,140,248,.12); border-color:rgba(124,140,248,.28); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
  color:var(--accent-ink) !important; font-weight:600 !important; }

/* ---- form controls -------------------------------------------------------- */
label p, [data-testid="stWidgetLabel"] p { color:var(--ink-2) !important; font-size:11.5px !important;
  font-weight:600 !important; letter-spacing:.03em; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"] {
  background:var(--panel) !important; border-color:var(--line-2) !important; border-radius:var(--r-sm) !important; }
[data-baseweb="select"] > div:hover { border-color:var(--accent) !important; }
[data-baseweb="popover"] [role="listbox"] { background:var(--panel) !important; border:1px solid var(--line-2) !important; }
[data-baseweb="tag"] { background:rgba(124,140,248,.16) !important; border-radius:6px !important; }
.stButton > button { background:var(--panel-2); border:1px solid var(--line-2); color:var(--ink-2);
  border-radius:var(--r-sm); font-weight:500; font-size:12.5px; padding:.35rem .8rem; transition:all .12s; }
.stButton > button:hover { border-color:var(--accent); color:var(--accent-ink); background:var(--panel); }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { border-color:var(--accent) !important; }

/* ---- alerts ------------------------------------------------------------------- */
[data-testid="stAlert"], [data-testid="stNotification"] { border-radius:var(--r-sm); border:1px solid var(--line-2);
  background:var(--panel) !important; }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color:var(--ink) !important; }

/* ---- streamlit dataframe frame ------------------------------------------------ */
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:var(--r-sm); overflow:hidden; }
[data-testid="stJson"] { border:1px solid var(--line); border-radius:var(--r-sm); background:var(--bg-elev) !important;
  padding:6px 10px; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:var(--r-sm) !important;
  background:var(--panel) !important; }
[data-testid="stExpander"] summary { font-size:12.5px; color:var(--ink-2); }

/* ---- topbar --------------------------------------------------------------- */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding-bottom:16px; border-bottom:1px solid var(--line); margin-bottom:26px; }
.topbar .tt { font-size:22px; font-weight:600; letter-spacing:-.02em; }
.topbar .tt small { display:block; font-size:11px; font-weight:700; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ink-3); margin-bottom:3px; }
.topbar .ctx { display:flex; align-items:center; gap:9px; flex-shrink:0; }
.livedot { width:7px; height:7px; border-radius:999px; background:var(--pos); box-shadow:0 0 0 3px rgba(70,178,107,.16); }
.mono-chip { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); background:var(--panel-2);
  border:1px solid var(--line-2); padding:4px 9px; border-radius:6px; }

/* ---- section header ----------------------------------------------------------- */
.sec { margin:34px 0 14px; }
.sec:first-of-type { margin-top:6px; }
.sec .eyebrow { font-size:10.5px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; color:var(--ink-3); }
.sec h2 { font-size:17px; font-weight:600; margin:3px 0 0; }
.sec p { font-size:12.5px; color:var(--ink-2); margin:4px 0 0; }

/* ---- context chips --------------------------------------------------------- */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 4px; }
.chip { display:inline-flex; align-items:baseline; gap:7px; font-size:12px; background:var(--panel);
  border:1px solid var(--line); border-radius:8px; padding:6px 11px; color:var(--ink); }
.chip b { color:var(--ink-3); font-weight:600; font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; }
.chip .v { font-family:var(--mono); font-size:12px; }

/* ---- kpi grid ------------------------------------------------------------------ */
.kpis { display:grid; gap:12px; margin:6px 0 2px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:15px 17px;
  display:flex; flex-direction:column; gap:9px; min-width:0; }
.kpi .l { font-size:10.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3); }
.kpi .v { font-family:var(--mono); font-size:26px; font-weight:600; line-height:1; letter-spacing:-.02em;
  color:var(--ink); font-feature-settings:'tnum'; }
.kpi .v.dim { color:var(--ink-3); }
.kpi .s { font-size:11.5px; color:var(--ink-2); }

/* ---- pills / deltas / bars --------------------------------------------------- */
.pill { display:inline-flex; align-items:center; gap:6px; font-size:10.5px; font-weight:600; letter-spacing:.04em;
  padding:3px 9px; border-radius:999px; border:1px solid var(--line-2); color:var(--ink-2); background:var(--panel-2);
  white-space:nowrap; text-transform:uppercase; }
.pill--pos  { color:var(--pos-ink);  background:var(--pos-bg);  border-color:rgba(70,178,107,.32); }
.pill--neg  { color:var(--neg-ink);  background:var(--neg-bg);  border-color:rgba(229,84,75,.32); }
.pill--warn { color:var(--warn-ink); background:var(--warn-bg); border-color:rgba(217,164,65,.32); }
.pill--accent { color:var(--accent-ink); background:rgba(124,140,248,.13); border-color:rgba(124,140,248,.34); }
.delta { font-family:var(--mono); font-size:11.5px; font-weight:600; white-space:nowrap; }
.delta--up { color:var(--pos-ink); } .delta--down { color:var(--neg-ink); } .delta--flat { color:var(--ink-3); }
.bar { position:relative; display:inline-block; vertical-align:middle; width:70px; height:6px; border-radius:999px;
  background:var(--line-2); overflow:hidden; }
.bar > i { position:absolute; left:0; top:0; bottom:0; background:var(--accent); border-radius:999px; }
.bar.pos > i { background:var(--pos); } .bar.neg > i { background:var(--neg); }

/* ---- custom table ------------------------------------------------------------- */
.rt-wrap { border:1px solid var(--line); border-radius:var(--r-sm); overflow:hidden; }
table.rt { width:100%; border-collapse:separate; border-spacing:0; font-size:12.5px; }
table.rt th { text-align:left; font-size:10px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
  color:var(--ink-3); background:var(--bg-elev); padding:10px 14px; border-bottom:1px solid var(--line); }
table.rt td { padding:11px 14px; border-bottom:1px solid var(--line); color:var(--ink); vertical-align:middle; }
table.rt tr:last-child td { border-bottom:0; }
table.rt tbody tr:hover td { background:rgba(255,255,255,.022); }
table.rt td.num, table.rt th.num { text-align:right; font-family:var(--mono); font-feature-settings:'tnum'; }
table.rt td.mut { color:var(--ink-3); }
table.rt tr.hl td { background:rgba(124,140,248,.06); }
table.rt tr.hl td:first-child { box-shadow:inset 2px 0 0 var(--accent); font-weight:600; }

/* ---- trace timeline ------------------------------------------------------------ */
.timeline { display:flex; height:11px; border-radius:999px; overflow:hidden; border:1px solid var(--line);
  background:var(--bg-elev); margin:4px 0 10px; }
.timeline > span { height:100%; }
.tl-legend { display:flex; flex-wrap:wrap; gap:16px; font-size:11.5px; color:var(--ink-2); margin-bottom:6px; }
.tl-legend i { width:9px; height:9px; border-radius:3px; display:inline-block; margin-right:6px; vertical-align:middle; }
.tl-legend b { font-family:var(--mono); color:var(--ink); font-weight:600; margin-left:5px; }

/* ---- cards --------------------------------------------------------------------- */
.card { background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:15px 17px; margin-bottom:10px; }
.card h4 { margin:0 0 8px; font-size:13.5px; font-weight:600; }
.card .li { display:flex; gap:9px; align-items:flex-start; font-size:12.5px; color:var(--ink-2); margin-top:6px; }
.card .li .dot { width:6px; height:6px; border-radius:999px; margin-top:6px; flex-shrink:0; background:var(--ink-3); }
.card .li .dot.pos { background:var(--pos); } .card .li .dot.neg { background:var(--neg); }
.answer { background:var(--panel); border:1px solid var(--line); border-left:2px solid var(--accent);
  border-radius:var(--r-sm); padding:13px 16px; font-size:13.5px; line-height:1.6; color:var(--ink); }
.answer.exp { border-left-color:var(--ink-3); color:var(--ink-2); }
.answer .h { font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3);
  display:block; margin-bottom:6px; }

/* ---- empty state ------------------------------------------------------------- */
.empty { text-align:center; padding:70px 24px; border:1px dashed var(--line-2); border-radius:var(--r); }
.empty h3 { color:var(--ink); font-weight:600; margin:0 0 8px; font-size:15px; }
.empty p { color:var(--ink-2); font-size:13px; margin:0; }
.empty code { background:var(--panel-2); padding:2px 7px; border-radius:5px; font-family:var(--mono); font-size:12px;
  color:var(--accent-ink); }

@media (max-width:820px) { .kpis { grid-template-columns:repeat(2,1fr) !important; } }
"""


def inject() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def _esc(v) -> str:
    return _html.escape(str(v))


# --- string helpers (return markup) ------------------------------------------------
def pill(text: str, tone: str = "") -> str:
    cls = f"pill pill--{tone}" if tone else "pill"
    return f'<span class="{cls}">{_esc(text)}</span>'


def delta(text: str, direction: str) -> str:
    cls = {"improved": "up", "regressed": "down", "neutral": "flat"}.get(direction, "flat")
    arrow = {"up": "▲", "down": "▼", "flat": "–"}[cls]
    return f'<span class="delta delta--{cls}">{arrow}&nbsp;{_esc(text)}</span>'


def bar(frac: Optional[float], tone: str = "") -> str:
    if frac is None:
        return '<span class="mut">—</span>'
    pct = max(0.0, min(1.0, float(frac))) * 100
    cls = f"bar {tone}" if tone else "bar"
    return f'<span class="{cls}"><i style="width:{pct:.1f}%"></i></span>'


def mono(text) -> str:
    return f'<span style="font-family:var(--mono);font-feature-settings:\'tnum\'">{_esc(text)}</span>'


# --- render helpers (write to page) --------------------------------------------------
def topbar(section: str, experiment_id: Optional[str] = None, n_experiments: int = 0) -> None:
    ctx = f'<span class="livedot"></span><span class="mono-chip">{_esc(experiment_id)}</span>' if experiment_id \
        else f'<span class="mono-chip">{n_experiments} experiments</span>'
    st.markdown(
        f'<div class="topbar"><div class="tt"><small>RAG Eval · Observability</small>{_esc(section)}</div>'
        f'<div class="ctx">{ctx}</div></div>',
        unsafe_allow_html=True,
    )


def sec(title: str, eyebrow: str = "", desc: str = "") -> None:
    eb = f'<div class="eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    ds = f"<p>{_esc(desc)}</p>" if desc else ""
    st.markdown(f'<div class="sec">{eb}<h2>{_esc(title)}</h2>{ds}</div>', unsafe_allow_html=True)


def chips(pairs: Iterable[tuple[str, str]]) -> None:
    inner = "".join(
        f'<span class="chip"><b>{_esc(k)}</b><span class="v">{_esc(v)}</span></span>' for k, v in pairs
    )
    st.markdown(f'<div class="chips">{inner}</div>', unsafe_allow_html=True)


def kpis(items: list[dict], cols: Optional[int] = None) -> None:
    n = cols or len(items)
    cells = []
    for it in items:
        val = it.get("value")
        vcls = "v dim" if (val in (None, "—")) else "v"
        parts = [f'<span class="l">{_esc(it["label"])}</span>',
                 f'<span class="{vcls}">{_esc(val if val is not None else "—")}</span>']
        if it.get("delta"):
            parts.append(it["delta"])
        if it.get("bar") is not None:
            parts.append(bar(it["bar"]))
        if it.get("sub"):
            parts.append(f'<span class="s">{_esc(it["sub"])}</span>')
        cells.append(f'<div class="kpi">{"".join(parts)}</div>')
    grid = f'grid-template-columns:repeat({n},minmax(0,1fr))'
    st.markdown(f'<div class="kpis" style="{grid}">{"".join(cells)}</div>', unsafe_allow_html=True)


def table(headers: list[str], rows: list[list[str]], num_cols: Iterable[int] = ()) -> None:
    num = set(num_cols)
    head = "".join(
        f'<th class="num">{_esc(h)}</th>' if i in num else f"<th>{_esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        hl = ' class="hl"' if (r and r[0] == "__HL__") else ""
        cells = r[1:] if hl else r
        tds = "".join(
            f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
            for i, c in enumerate(cells)
        )
        body.append(f"<tr{hl}>{tds}</tr>")
    st.markdown(
        f'<div class="rt-wrap"><table class="rt"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def timeline(segments: list[tuple[str, float]]) -> None:
    """segments: list of (stage_name, milliseconds)."""
    total = sum(ms for _, ms in segments) or 1.0
    bars = "".join(
        f'<span style="width:{(ms / total) * 100:.2f}%;background:{STAGE_COLORS.get(name, "#8A909B")}"></span>'
        for name, ms in segments if ms > 0
    )
    legend = "".join(
        f'<span><i style="background:{STAGE_COLORS.get(name, "#8A909B")}"></i>{_esc(name)}<b>{ms:.0f}ms</b></span>'
        for name, ms in segments if ms > 0
    )
    st.markdown(
        f'<div class="tl-legend">{legend}</div><div class="timeline">{bars}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, lines: list[tuple[str, str]]) -> None:
    """lines: list of (tone, text) where tone in {'', 'pos', 'neg'}."""
    body = "".join(
        f'<div class="li"><span class="dot {t}"></span><span>{_esc(x)}</span></div>' for t, x in lines
    )
    st.markdown(f'<div class="card"><h4>{_esc(title)}</h4>{body}</div>', unsafe_allow_html=True)


def qcard(qid: str, category: str, tone: str, question: str, reason: str) -> None:
    st.markdown(
        f'<div class="card"><h4>{_esc(qid)}&nbsp;&nbsp;{pill(category, tone)}</h4>'
        f'<div class="li"><span class="dot"></span><span>{_esc(question)}</span></div>'
        f'<div class="li"><span class="dot neg"></span><span>{_esc(reason)}</span></div></div>',
        unsafe_allow_html=True,
    )


def esc(v) -> str:
    return _esc(v)


def answer_block(label: str, text: str, kind: str = "") -> None:
    cls = f"answer {kind}" if kind else "answer"
    st.markdown(
        f'<div class="{cls}"><span class="h">{_esc(label)}</span>{_esc(text)}</div>',
        unsafe_allow_html=True,
    )


def empty(title: str, hint_html: str) -> None:
    st.markdown(f'<div class="empty"><h3>{_esc(title)}</h3><p>{hint_html}</p></div>', unsafe_allow_html=True)

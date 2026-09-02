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

ACCENT = "#8B5CF6"
STAGE_COLORS = {
    "embedding": "#8B5CF6", "retrieval": "#34D399", "reranking": "#64748B",
    "generation": "#FBBF24", "evaluation": "#F472B6", "total": "#64748B",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0B0A16; --bg-1:#0E0C1C; --panel:#1C1942; --panel-2:#252150; --panel-3:#2E2960;
  --line:rgba(255,255,255,.09); --line-2:rgba(255,255,255,.15); --line-3:rgba(255,255,255,.22);
  --toplit:linear-gradient(180deg, rgba(255,255,255,.03), transparent 32%);
  --ink:#F3F1FA; --ink-2:#ADA9C6; --ink-3:#78748F; --ink-4:#565272;
  --accent:#8B5CF6; --accent-2:#A78BFA; --accent-ink:#CDBEFF;
  --grad:linear-gradient(135deg,#9168FF 0%,#6D5CFF 100%);
  --pos:#34D399; --pos-ink:#6EE7B7; --neg:#F87171; --neg-ink:#FCA5A5;
  --warn:#FBBF24; --warn-ink:#FCD34D; --info:#60A5FA; --pink:#F472B6;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --head:'Plus Jakarta Sans','Inter',-apple-system,BlinkMacSystemFont,sans-serif;
  --r:18px; --r-sm:13px; --r-xs:9px;
  --shadow:0 16px 40px -18px rgba(0,0,0,.7);
  --shadow-sm:0 8px 24px -12px rgba(0,0,0,.55);
  --glow:radial-gradient(135% 115% at 0% 0%, rgba(139,92,246,.10), transparent 46%);
}

/* shell ------------------------------------------------------------------------ */
html,body,[data-testid="stApp"]{background:var(--bg);}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(1100px 460px at 80% -10%, rgba(139,92,246,.12), transparent 60%),
    radial-gradient(900px 380px at 0% 0%, rgba(96,165,250,.06), transparent 55%), var(--bg);}
.stApp{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink);
  -webkit-font-smoothing:antialiased;font-feature-settings:'cv05','ss01';}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stDeployButton"],#MainMenu,footer{display:none!important;}
[data-testid="stMainBlockContainer"]{max-width:1260px;padding:2rem 2.6rem 5rem;}
[data-testid="stMain"]{background:transparent;}
[data-testid="stVerticalBlock"]{gap:.85rem;}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]{gap:1rem;}
hr{border-color:var(--line);}

/* typography ----------------------------------------------------------------------- */
[data-testid="stMarkdownContainer"] p{color:var(--ink-2);font-size:13px;line-height:1.55;}
a,a:visited{color:var(--accent-ink);text-decoration:none;} a:hover{text-decoration:underline;}
code,kbd,pre{font-family:var(--mono)!important;}
[data-testid="stCode"]{border:1px solid var(--line);border-radius:var(--r-sm);background:var(--bg-1)!important;}
[data-testid="stCode"] pre{background:transparent!important;font-size:12px!important;line-height:1.6;}

/* shared panel look --------------------------------------------------------------- */
.sl,.hbc,.dn,.kv,.card,.rt-wrap,.cmp,.qlist,.tro,.cols2,.empty{
  background:var(--toplit), var(--glow), var(--panel);
  border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,.06);}
.sl-h,.hbc-h,.kv-h,.dn-h,.cmp-fam{
  font:600 12px/1.2 var(--head);letter-spacing:0;text-transform:none;color:var(--ink-2);}

/* sidebar --------------------------------------------------------------------------- */
/* Pinned open, always visible — Streamlit's own collapse control hides off-screen with
   no reopen affordance on narrow viewports, so we keep the nav permanently docked. */
[data-testid="stSidebar"],[data-testid="stSidebar"][aria-expanded="false"]{
  transform:none!important;visibility:visible!important;margin-left:0!important;
  width:256px!important;min-width:256px!important;max-width:256px!important;
  background:
    radial-gradient(420px 200px at 12% -4%, rgba(139,92,246,.22), transparent 70%),
    linear-gradient(180deg, #131029, var(--bg-1) 42%);
  border-right:1px solid var(--line-2);
  box-shadow:1px 0 0 rgba(255,255,255,.03), 24px 0 60px -30px rgba(0,0,0,.7);}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:1.5rem .95rem 1rem;}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{display:none!important;}
[data-testid="stAppViewContainer"]>section:first-child{min-width:256px!important;}

.brand{display:flex;align-items:center;gap:12px;padding:2px 4px 16px;margin-bottom:14px;
  border-bottom:1px solid var(--line);position:relative;}
.brand .mk{width:36px;height:36px;border-radius:11px;flex-shrink:0;color:#fff;
  background:var(--grad);display:grid;place-items:center;
  box-shadow:0 0 0 1px rgba(139,92,246,.4),0 14px 28px -8px rgba(124,92,255,.7),inset 0 1px 0 rgba(255,255,255,.35);}
.brand .mk svg{width:21px;height:21px;display:block;}
.brand .nm{font:800 16.5px/1.1 var(--head);letter-spacing:-.01em;color:var(--ink);}
.brand .nm span{display:block;font:500 9.5px var(--head);color:var(--accent-ink);opacity:.75;letter-spacing:.04em;margin-top:3px;text-transform:uppercase;}
.s-lbl{font:700 9px var(--head);letter-spacing:.16em;text-transform:uppercase;color:var(--ink-4);
  margin:20px 8px 9px;display:flex;align-items:center;gap:8px;}
.s-lbl::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line-2),transparent);}
.s-foot{margin:18px 8px 2px;font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;color:var(--ink-4);
  display:flex;align-items:center;gap:6px;}
.s-foot::before{content:"";width:5px;height:5px;border-radius:99px;background:var(--accent);
  box-shadow:0 0 8px 1px rgba(139,92,246,.7);}

[data-testid="stSidebar"] [role="radiogroup"]{gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"]>label{display:flex;align-items:center;gap:12px;
  padding:10px 13px;border-radius:12px;margin:0;border:1px solid transparent;cursor:pointer;
  transition:transform .13s, background .13s, box-shadow .13s;}
[data-testid="stSidebar"] [role="radiogroup"]>label>div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"]>label::before{content:"";width:18px;height:18px;flex-shrink:0;
  background:var(--ink-3);transition:background .13s;
  -webkit-mask:var(--ic) center/contain no-repeat;mask:var(--ic) center/contain no-repeat;}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover{background:rgba(255,255,255,.045);transform:translateX(3px);}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover::before{background:var(--ink);}
[data-testid="stSidebar"] [role="radiogroup"] div[data-testid="stMarkdownContainer"] p{
  font:500 13px var(--head)!important;color:var(--ink-2)!important;text-transform:none!important;letter-spacing:0!important;}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked){
  background:var(--grad);border-color:transparent;transform:translateX(3px);
  box-shadow:0 12px 26px -10px rgba(124,92,255,.8),inset 0 1px 0 rgba(255,255,255,.22);}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked)::before{background:#fff;}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked) div[data-testid="stMarkdownContainer"] p{color:#fff!important;font-weight:700!important;}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(1){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1.5'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1.5'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(2){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3 21 8l-9 5-9-5 9-5Z'/%3E%3Cpath d='M3 13l9 5 9-5'/%3E%3Cpath d='M3 17.5l9 5 9-5'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(3){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 20h18'/%3E%3Cpath d='M7 20V11'/%3E%3Cpath d='M12 20V4'/%3E%3Cpath d='M17 20v-6'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(4){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z'/%3E%3Cpath d='M12 9v4'/%3E%3Cpath d='M12 17h.01'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(5){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12h4l3 8 4-16 3 8h4'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(6){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 12V3a9 9 0 0 1 7.8 4.5Z'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type(7){--ic:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 8v8'/%3E%3Cpath d='M8 12h8'/%3E%3C/svg%3E");}
[data-testid="stSidebar"] .stButton,[data-testid="stSidebar"] [data-testid="stButton"]{margin-top:14px;}
[data-testid="stSidebar"] .stButton>button{position:relative;z-index:2;width:100%;}
.mini{display:flex;flex-direction:column;gap:7px;margin:10px 4px 6px;padding:14px;border-radius:14px;
  background:linear-gradient(180deg,rgba(139,92,246,.18),rgba(139,92,246,.03)),var(--panel);
  border:1px solid rgba(139,92,246,.3);box-shadow:0 14px 30px -14px rgba(124,92,255,.5);}
.mini .mh{font:700 8.5px var(--head);letter-spacing:.13em;text-transform:uppercase;color:var(--accent-ink);margin-bottom:1px;}
.mini .mrow{display:flex;justify-content:space-between;align-items:center;font-size:11.5px;color:var(--ink-3);}
.mini .mrow b{font:600 12px var(--head);color:var(--ink);}

/* form controls ---------------------------------------------------------------------- */
label p,[data-testid="stWidgetLabel"] p{color:var(--ink-3)!important;font-size:10px!important;font-weight:700!important;letter-spacing:.08em;text-transform:uppercase;}
[data-baseweb="select"]>div,[data-baseweb="base-input"]{background:var(--panel)!important;border-color:var(--line-2)!important;border-radius:11px!important;font-size:12.5px!important;}
[data-baseweb="select"]>div:hover{border-color:var(--accent)!important;}
[data-baseweb="popover"] [role="listbox"]{background:var(--panel-2)!important;border:1px solid var(--line-2)!important;border-radius:12px!important;}
[data-baseweb="tag"]{background:rgba(139,92,246,.2)!important;border-radius:7px!important;}
.stButton>button,[data-testid="stFormSubmitButton"] button{background:var(--grad);border:none;color:#fff;border-radius:11px;font:600 12px var(--head);padding:.5rem .95rem;transition:all .12s;box-shadow:0 10px 22px -10px rgba(124,92,255,.55);}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.1);color:#fff;transform:translateY(-1px);}
[data-testid="stForm"]{border:1px solid var(--line)!important;border-radius:var(--r)!important;
  background:var(--toplit), var(--glow), var(--panel)!important;box-shadow:var(--shadow)!important;padding:18px 20px!important;}
[data-testid="stNumberInput"] button{background:var(--panel-2)!important;border-color:var(--line-2)!important;}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{background:var(--accent)!important;border-color:var(--accent)!important;box-shadow:0 0 0 4px rgba(139,92,246,.2)!important;}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"]>div{background:var(--grad)!important;}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:var(--r-sm)!important;background:var(--panel)!important;box-shadow:var(--shadow-sm);}
[data-testid="stExpander"] summary{font:500 12px var(--head);color:var(--ink-2);}
[data-testid="stAlert"]{border-radius:var(--r-sm);border:1px solid var(--line-2);background:var(--panel)!important;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{color:var(--ink)!important;}
[data-testid="stDataFrame"]{border-radius:var(--r-sm);overflow:hidden;}

/* topbar --------------------------------------------------------------------------- */
.tb{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-bottom:18px;border-bottom:1px solid var(--line);margin-bottom:24px;}
.tb .h{font:600 25px var(--head);letter-spacing:-.02em;color:var(--ink);}
.tb .h small{font:700 10px var(--head);letter-spacing:.16em;text-transform:uppercase;color:var(--accent-ink);opacity:.8;margin-right:11px;}
.tb .rt{display:flex;align-items:center;gap:9px;font-size:11px;color:var(--ink-3);flex-shrink:0;}
.dot{width:7px;height:7px;border-radius:99px;background:var(--pos);box-shadow:0 0 0 4px rgba(52,211,153,.18);}
.mc{font-family:var(--mono);font-size:11px;color:var(--ink-2);background:var(--panel);border:1px solid var(--line-2);padding:5px 11px;border-radius:99px;}

/* vitals ------------------------------------------------------------------------- */
.vitals{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin-bottom:24px;}
.vt{background:var(--toplit), var(--glow), var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:14px 16px 15px;display:flex;flex-direction:column;gap:8px;min-height:78px;
  box-shadow:var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,.06);}
.vt .l{font:700 10px var(--head);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.vt .v{font:700 25px/1 var(--head);color:var(--ink);letter-spacing:-.025em;font-feature-settings:'tnum';}
.vt .v.dim{color:var(--ink-4);}
.vt .mtr{margin-top:3px;}

/* section header ----------------------------------------------------------------- */
.sec{display:flex;align-items:baseline;gap:12px;margin:28px 0 14px;flex-wrap:wrap;}
.sec:first-of-type{margin-top:4px;}
.sec h2{font:600 16px var(--head);letter-spacing:-.01em;margin:0;color:var(--ink);}
.sec .eb{font:700 9px var(--head);letter-spacing:.12em;text-transform:uppercase;color:var(--accent-ink);
  background:rgba(139,92,246,.14);padding:4px 9px;border-radius:99px;}
.sec .ds{font-size:12px;color:var(--ink-3);}

/* stat list --------------------------------------------------------------------------- */
.sl{overflow:hidden;}
.sl-h{padding:14px 18px;border-bottom:1px solid var(--line);}
.sl-row{display:grid;grid-template-columns:1fr auto 72px auto;align-items:center;gap:16px;padding:11px 18px;border-bottom:1px solid var(--line);}
.sl-row:last-child{border-bottom:0;}
.sl-row:hover{background:rgba(139,92,246,.045);}
.sl-l{font-size:12.5px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sl-v{font:600 13px var(--head);color:var(--ink);text-align:right;font-feature-settings:'tnum';}
.sl-v.dim{color:var(--ink-4);}
.sl-d{font:600 11px var(--head);text-align:right;}

/* meter -------------------------------------------------------------------------------- */
.mtr{position:relative;height:7px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;display:block;}
.mtr>i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:var(--grad);}
.mtr>i::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.25),transparent 65%);}
.mtr.good>i{background:linear-gradient(90deg,#34D399,#10B981);}
.mtr.warn>i{background:linear-gradient(90deg,#FBBF24,#F59E0B);}
.mtr.bad>i{background:linear-gradient(90deg,#F87171,#EF4444);}
.mtr.t2>i{background:linear-gradient(90deg,#60A5FA,#3B82F6);}
.mtr-i{display:inline-block;width:64px;vertical-align:middle;}

/* hbar chart ------------------------------------------------------------------------ */
.hbc{padding:17px 19px;}
.hbc-h{margin-bottom:14px;}
.hbc-row{display:grid;grid-template-columns:90px 1fr 58px;align-items:center;gap:12px;padding:5px 0;}
.hbc-l{font-size:12px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hbc-trk{height:22px;border-radius:8px;background:rgba(255,255,255,.05);position:relative;overflow:hidden;}
.hbc-trk>i{position:absolute;left:0;top:0;bottom:0;border-radius:8px;min-width:3px;}
.hbc-trk>i::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.22),transparent 60%);}
.hbc-v{font:600 12px var(--head);color:var(--ink);text-align:right;font-feature-settings:'tnum';}

/* donut ----------------------------------------------------------------------------- */
.dn{padding:19px;display:flex;gap:22px;align-items:center;}
.dn .ring{position:relative;flex-shrink:0;width:132px;height:132px;}
.dn .ring svg{width:100%;height:100%;transform:rotate(-90deg);}
.dn .ring .ctr{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;}
.dn .ring .ctr b{font:700 15px var(--head);color:var(--ink);font-feature-settings:'tnum';}
.dn .ring .ctr span{font:700 8px var(--head);letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);}
.dn .lg{display:flex;flex-direction:column;gap:9px;font-size:12.5px;color:var(--ink-2);flex:1;min-width:0;}
.dn .lg .lgr{display:flex;align-items:center;gap:9px;}
.dn .lg i{width:10px;height:10px;border-radius:3px;flex-shrink:0;}
.dn .lg b{font:600 12px var(--head);color:var(--ink);margin-left:auto;padding-left:14px;font-feature-settings:'tnum';}

/* kv --------------------------------------------------------------------------------- */
.kv{overflow:hidden;}
.kv-h{padding:14px 18px;border-bottom:1px solid var(--line);}
.kv-row{display:flex;justify-content:space-between;gap:12px;padding:9px 18px;border-bottom:1px solid var(--line);font-size:12.5px;}
.kv-row:last-child{border-bottom:0;}
.kv-row .k{color:var(--ink-3);} .kv-row .v{font-family:var(--mono);color:var(--ink);font-size:11.5px;}

/* table ---------------------------------------------------------------------------- */
.rt-wrap{overflow:auto;}
table.rt{width:100%;border-collapse:separate;border-spacing:0;font-size:12.5px;}
table.rt th{text-align:left;font:700 9.5px var(--head);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);background:rgba(255,255,255,.022);padding:12px 16px;border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0;}
table.rt td{padding:11px 16px;border-bottom:1px solid var(--line);color:var(--ink);vertical-align:middle;}
table.rt tr:last-child td{border-bottom:0;}
table.rt tbody tr:hover td{background:rgba(139,92,246,.05);}
table.rt td.num,table.rt th.num{text-align:right;font-family:var(--mono);font-feature-settings:'tnum';}
table.rt td.mut{color:var(--ink-3);}
table.rt tr.hl td{background:rgba(139,92,246,.10);}
table.rt tr.hl td:first-child{box-shadow:inset 3px 0 0 var(--accent);font-weight:600;}

/* compare grid --------------------------------------------------------------------- */
.cmp{overflow:hidden;width:fit-content;max-width:100%;}
.cmp-fam{padding:11px 16px 8px;background:rgba(255,255,255,.022);border-bottom:1px solid var(--line);
  font:700 9px var(--head)!important;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);}
.cmp-row{display:grid;align-items:center;gap:14px;padding:9px 16px;border-bottom:1px solid var(--line);font-size:12.5px;}
.cmp-row:last-child{border-bottom:0;}
.cmp-row:hover{background:rgba(139,92,246,.045);}
.cmp-m{color:var(--ink-2);}
.cmp-c{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-feature-settings:'tnum';color:var(--ink);}
.cmp-c .mtr-i{width:54px;}
.cmp-hd{color:var(--ink-3)!important;background:rgba(255,255,255,.022);}
.cmp-hd .cmp-c{color:var(--ink-3);font:700 9px var(--head);letter-spacing:.06em;text-transform:uppercase;}

/* pills / deltas ------------------------------------------------------------------- */
.pill{display:inline-flex;align-items:center;gap:5px;font:700 9px var(--head);letter-spacing:.05em;padding:3px 9px;border-radius:99px;border:1px solid var(--line-2);color:var(--ink-2);background:var(--panel-2);white-space:nowrap;text-transform:uppercase;}
.pill--pos{color:var(--pos-ink);background:rgba(52,211,153,.13);border-color:rgba(52,211,153,.32);}
.pill--neg{color:var(--neg-ink);background:rgba(248,113,113,.13);border-color:rgba(248,113,113,.32);}
.pill--warn{color:var(--warn-ink);background:rgba(251,191,36,.13);border-color:rgba(251,191,36,.32);}
.pill--accent{color:var(--accent-ink);background:rgba(139,92,246,.16);border-color:rgba(139,92,246,.36);}
.dl{font:600 11px var(--head);white-space:nowrap;}
.dl--up{color:var(--pos-ink);} .dl--down{color:var(--neg-ink);} .dl--flat{color:var(--ink-4);}

/* cards ----------------------------------------------------------------------------- */
.card{padding:16px 18px;}
.card h4{margin:0 0 9px;font:600 13px var(--head);display:flex;align-items:center;gap:8px;}
.card .li{display:flex;gap:9px;align-items:flex-start;font-size:12.5px;color:var(--ink-2);margin-top:6px;line-height:1.5;}
.card .li .b{width:6px;height:6px;border-radius:99px;margin-top:6px;flex-shrink:0;background:var(--ink-4);}
.card .li .b.pos{background:var(--pos);} .card .li .b.neg{background:var(--neg);}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);overflow:hidden;}
.cols2>div{background:var(--panel);padding:15px 17px;}
.cols2 .ch{font:700 9.5px var(--head);letter-spacing:.09em;text-transform:uppercase;margin-bottom:9px;}
.cols2 .ch.pos{color:var(--pos-ink);} .cols2 .ch.neg{color:var(--neg-ink);}
.tro{padding:14px 17px;margin-bottom:10px;}
.tro .t{font:600 13px var(--head);margin-bottom:3px;}
.tro .s{font-size:11.5px;color:var(--ink-3);}

.ans{background:var(--glow), var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:var(--r-sm);padding:14px 16px;font-size:13.5px;line-height:1.6;color:var(--ink);box-shadow:var(--shadow-sm);}
.ans.exp{border-left-color:var(--ink-4);color:var(--ink-2);}
.ans .h{font:700 8.5px var(--head);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-4);display:block;margin-bottom:6px;}

.qlist{overflow:hidden;}
.qrow{display:grid;grid-template-columns:60px 156px 1fr;gap:14px;align-items:baseline;padding:11px 18px;border-bottom:1px solid var(--line);font-size:12.5px;}
.qrow:last-child{border-bottom:0;} .qrow:hover{background:rgba(139,92,246,.045);}
.qrow .qi{font-family:var(--mono);font-size:11px;color:var(--ink-3);}
.qrow .qw{color:var(--ink-2);}

.empty{text-align:center;padding:64px 24px;border-style:dashed;border-color:var(--line-2);}
.empty h3{color:var(--ink);font:600 16px var(--head);margin:0 0 8px;}
.empty p{color:var(--ink-3);font-size:12.5px;margin:0;}
.empty code{background:var(--panel-2);padding:2px 7px;border-radius:6px;font-family:var(--mono);font-size:11.5px;color:var(--accent-ink);}

@media (max-width:1100px){.vitals{grid-template-columns:repeat(4,1fr);}}
@media (max-width:640px){.vitals{grid-template-columns:repeat(2,1fr);}}
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
        frac = it.get("frac")
        m = (f'<span class="mtr {_tone_for(frac)}" style="width:100%">'
             f'<i style="width:{max(0, min(1, frac)) * 100:.0f}%"></i></span>') if frac is not None else ""
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


def donut_html(header: str, segments: list[tuple], center: Optional[tuple] = None) -> str:
    """segments: (label, value, color). center: (big, small) shown in the ring."""
    total = sum(v for _, v, _ in segments) or 1.0
    circ, off, arcs = 100.0, 25.0, []
    for _, v, c in segments:
        frac = v / total
        seg = max(frac * circ - 1.4, 0.4)  # small gap between arcs
        arcs.append(f'<circle cx="18" cy="18" r="15.915" fill="none" stroke="{c}" stroke-width="5" '
                    f'stroke-dasharray="{seg:.2f} {circ - seg:.2f}" stroke-dashoffset="{off:.2f}"/>')
        off -= frac * circ
    big, small = center or (f"${total:.4f}", "total")
    legend = "".join(f'<div class="lgr"><i style="background:{c}"></i>{_e(l)}<b>{_e(f"${v:.6f}")}</b></div>'
                     for l, v, c in segments)
    return (
        f'<div class="dn"><div class="ring"><svg viewBox="0 0 36 36">'
        f'<circle cx="18" cy="18" r="15.915" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="5"/>'
        f'{"".join(arcs)}</svg><div class="ctr"><b>{_e(big)}</b><span>{_e(small)}</span></div></div>'
        f'<div class="lg"><div class="dn-h">{_e(header)}</div>{legend}</div></div>')


def donut(header: str, segments: list[tuple]) -> None:
    st.markdown(donut_html(header, segments), unsafe_allow_html=True)


def kv_html(header: str, pairs: list[tuple]) -> str:
    body = "".join(f'<div class="kv-row"><span class="k">{_e(k)}</span><span class="v">{_e(v)}</span></div>'
                   for k, v in pairs)
    return f'<div class="kv"><div class="kv-h">{_e(header)}</div>{body}</div>'


def kv(header: str, pairs: list[tuple]) -> None:
    st.markdown(kv_html(header, pairs), unsafe_allow_html=True)


def grid(cells: list[str], cols: str, gap: str = "16px") -> None:
    """One CSS-grid row from pre-rendered HTML strings — avoids st.columns quirks."""
    inner = "".join(f"<div>{c}</div>" for c in cells if c)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:{cols};gap:{gap};'
        f'align-items:start;margin:4px 0 18px">{inner}</div>', unsafe_allow_html=True)


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

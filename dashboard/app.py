"""RAG Eval — evaluation & observability dashboard (Phase 15).

    streamlit run dashboard/app.py

Reads only from the experiment store. Pure presentation over the CLI's own
functions. The look lives in dashboard/theme.py + .streamlit/config.toml.
"""

from __future__ import annotations

import os
import sys
from html import escape
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.experiment.comparison import compare_experiments
from app.experiment.failure_analysis import FailureThresholds, analyze_failures
from app.experiment.results import ExperimentResult
from app.experiment.slicing import slice_report, underperforming_slices
from app.experiment.store import ExperimentStore
from app.observability.trace import Trace
from dashboard import theme

st.set_page_config(page_title="Assay — RAG eval", page_icon="◎", layout="wide", initial_sidebar_state="expanded")
theme.inject()

SECTIONS = ["Overview", "Experiments", "Comparison", "Failures", "Traces", "Slices"]
CAT_TONE = {"OK": "pos", "INSUFFICIENT_CONTEXT": "accent", "RETRIEVAL_FAILURE": "neg",
            "GENERATION_FAILURE": "neg", "HALLUCINATION": "neg", "CITATION_FAILURE": "warn", "ERROR": "neg"}
CAT_COLOR = {"OK": "#3FBE6B", "INSUFFICIENT_CONTEXT": "#7C8CF8", "RETRIEVAL_FAILURE": "#E5544B",
             "GENERATION_FAILURE": "#E5544B", "HALLUCINATION": "#C77DE0", "CITATION_FAILURE": "#D9A441",
             "ERROR": "#E5544B"}
STAGE_C = ["#7C8CF8", "#4CC9C0", "#8A909B", "#E0A458", "#C77DE0"]
CMP_FAMILIES = [
    ("Retrieval", {"hit_rate", "precision", "recall", "mrr", "ndcg"}),
    ("Generation", {"correctness", "relevance", "exact_match", "token_f1"}),
    ("Faithfulness", {"faithfulness"}),
    ("Citations", {"citation_precision", "citation_completeness", "citation_correctness", "citation_hallucination_rate"}),
    ("Performance", {"latency_total_ms", "latency_total_p95_ms", "latency_generation_ms",
                     "latency_evaluation_ms", "cost_per_query_usd", "cost_total_usd"}),
]


# --- data --------------------------------------------------------------------------------
def _db_path() -> str:
    """Which experiment DB to read.

    `EXPERIMENTS_DB` env var wins (any host can override); then a real local DB;
    then the small committed seed DB so a fresh deploy has something to show.
    """
    env = os.environ.get("EXPERIMENTS_DB")
    if env:
        return env
    local = _ROOT / "data" / "experiments.db"
    return str(local if local.exists() else _ROOT / "data" / "demo.db")


DB_PATH = _db_path()


@st.cache_data(show_spinner=False)
def _summaries() -> list[dict]:
    with ExperimentStore(DB_PATH) as store:
        return [s.model_dump() for s in store.list(limit=200)]


@st.cache_data(show_spinner=False)
def _result_json(eid: str) -> Optional[str]:
    with ExperimentStore(DB_PATH) as store:
        r = store.get(eid)
        return r.model_dump_json() if r else None


@st.cache_data(show_spinner=False)
def _traces_json(eid: str) -> list[str]:
    with ExperimentStore(DB_PATH) as store:
        return [t.model_dump_json() for t in store.get_traces(eid)]


def result_of(eid: str) -> Optional[ExperimentResult]:
    raw = _result_json(eid)
    return ExperimentResult.model_validate_json(raw) if raw else None


# --- formatting -------------------------------------------------------------------------
def f3(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def short(eid: str) -> str:
    return eid.rsplit("_", 1)[0]


def fmtval(v, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "$":
        return f"${v:.5f}"
    if unit == "ms":
        return f"{v:.0f}ms"
    return f"{v:.3f}"


def cmp_delta(d, unit: str) -> str:
    if d is None:
        return '<span class="dl dl--flat">–</span>'
    if unit == "ms":
        body = f"{d.absolute:+.0f}ms"
    elif unit == "$":
        body = f"{d.absolute:+.5f}"
    elif d.percent is not None:
        body = f"{d.percent:+.1f}%"
    else:
        body = f"{d.absolute:+.3f}"
    return theme.delta(body, d.direction)


def tint(v, ov) -> str:
    if ov is None:
        return ""
    d = v - ov
    if abs(d) < 1e-9:
        return ""
    if d > 0:
        return f"background:rgba(63,190,107,{min(0.28, 0.07 + d * 1.4):.3f})"
    return f"background:rgba(229,84,75,{min(0.34, 0.09 + abs(d) * 1.8):.3f})"


def _g(agg, attr):
    return getattr(agg, attr, None) if agg is not None else None


def vitals_for(r: ExperimentResult) -> None:
    theme.vitals([
        {"label": "Recall@K", "value": f3(_g(r.retrieval, "recall")), "frac": _g(r.retrieval, "recall")},
        {"label": "Precision@K", "value": f3(_g(r.retrieval, "precision")), "frac": _g(r.retrieval, "precision")},
        {"label": "Correctness", "value": f3(_g(r.generation, "judge_correctness")), "frac": _g(r.generation, "judge_correctness")},
        {"label": "Faithfulness", "value": f3(_g(r.faithfulness, "faithfulness")), "frac": _g(r.faithfulness, "faithfulness")},
        {"label": "Cite prec", "value": f3(_g(r.citation, "citation_precision")), "frac": _g(r.citation, "citation_precision")},
        {"label": "Latency p95", "value": f"{r.latency_report.p95('total'):.0f}ms" if r.latency_report else "—"},
        {"label": "$ / query", "value": f"{r.cost.cost_per_query_usd:.5f}"},
    ])


# --- sections ----------------------------------------------------------------------------
def overview(r: ExperimentResult) -> None:
    panels = []
    if r.retrieval:
        m = r.retrieval
        panels.append(theme.statlist_html("Retrieval quality", [
            {"label": "Hit@K", "value": f3(m.hit_rate), "frac": m.hit_rate},
            {"label": "Precision@K", "value": f3(m.precision), "frac": m.precision},
            {"label": "Recall@K", "value": f3(m.recall), "frac": m.recall},
            {"label": "MRR", "value": f3(m.mrr), "frac": m.mrr},
            {"label": "NDCG@K", "value": f3(m.ndcg), "frac": m.ndcg},
        ]))
    if r.generation:
        g = r.generation
        panels.append(theme.statlist_html("Generation quality", [
            {"label": "Correctness", "value": f3(g.judge_correctness), "frac": g.judge_correctness},
            {"label": "Relevance", "value": f3(g.judge_relevance), "frac": g.judge_relevance},
            {"label": "Exact match", "value": f3(g.exact_match), "frac": g.exact_match},
            {"label": "Token F1", "value": f3(g.token_f1), "frac": g.token_f1},
            {"label": "Abstention", "value": f3(g.abstention_accuracy), "frac": g.abstention_accuracy},
        ]))
    rows = []
    if r.faithfulness:
        fa = r.faithfulness
        rows += [{"label": "Faithful · macro", "value": f3(fa.faithfulness), "frac": fa.faithfulness},
                 {"label": "Faithful · micro", "value": f3(fa.claim_support_rate), "frac": fa.claim_support_rate}]
    if r.citation:
        ct = r.citation
        rows += [{"label": "Cite complete", "value": f3(ct.citation_completeness), "frac": ct.citation_completeness},
                 {"label": "Cite precision", "value": f3(ct.citation_precision), "frac": ct.citation_precision},
                 {"label": "Cite halluc", "value": f3(ct.citation_hallucination_rate),
                  "frac": ct.citation_hallucination_rate, "tone": "bad" if ct.citation_hallucination_rate else "good"}]
    if rows:
        panels.append(theme.statlist_html("Faithfulness · Citations", rows))
    if panels:
        theme.grid(panels, cols=f"repeat({len(panels)}, 1fr)")

    cc = r.cost
    segs = [("generation", cc.generation_usd, "#E0A458"), ("evaluation", cc.evaluation_usd, "#C77DE0"),
            ("query embed", cc.query_embedding_usd, "#7C8CF8")]
    segs = [s for s in segs if s[1] > 0] or [("generation", max(cc.total_usd, 1e-9), "#E0A458")]
    c = r.config
    lat = theme.hbar_html("Latency by stage · p95 ms",
                          [(k, s.p95_ms) for k, s in r.latency_report.stages.items() if k != "total"],
                          fmt=lambda v: f"{v:.0f}", colors=STAGE_C) if r.latency_report else ""
    theme.grid([
        lat,
        theme.donut_html("Cost split · this run", segs),
        theme.kv_html("Run", [
            ("model", c.generation_model), ("chunk / overlap", f"{c.chunk_size} / {c.chunk_overlap}"),
            ("top_k", str(c.top_k)), ("questions", f"{r.num_questions}  ({r.num_errors} err)"),
            ("corpus", f"{r.document_count} docs · {r.chunk_count} chunks"),
            ("tokens", f"{r.total_token_usage.total_tokens:,}"),
            ("total cost", f"${cc.total_usd:.5f}"), ("date", r.started_at[:16].replace("T", " ")),
        ]),
    ], cols="minmax(0,1.2fr) minmax(0,.95fr) minmax(0,1.05fr)")


def experiments(summaries: list[dict]) -> None:
    models = sorted({s["generation_model"].replace("claude-", "") for s in summaries})
    spend = sum(s.get("estimated_cost_usd") or 0 for s in summaries)
    theme.sec("Tracked experiments", "Registry",
              f"{len(summaries)} runs · {', '.join(models)} · ${spend:.4f} total")
    rows = []
    for s in summaries:
        rc, co = s.get("retrieval_recall"), s.get("judge_correctness")
        rows.append([
            theme.mono(s["experiment_id"]),
            theme.pill(s["generation_model"].replace("claude-", ""), "accent"),
            theme.mono(s["chunk_size"]), theme.mono(s["top_k"]), theme.mono(s["num_questions"]),
            f'{theme.mono(f3(rc))} {theme.meter(rc)}' if rc is not None else "—",
            f'{theme.mono(f3(co))} {theme.meter(co)}' if co is not None else "—",
            theme.mono(f'${s.get("estimated_cost_usd") or 0:.4f}'),
        ])
    theme.table(["Experiment", "Model", "Chunk", "K", "N", "Recall", "Correctness", "Cost"],
                rows, num_cols=[2, 3, 4, 7])
    with st.expander("Raw table · sortable"):
        st.dataframe(summaries, width="stretch", hide_index=True)


def comparison(ids: list[str]) -> None:
    theme.sec("Compare experiments", "Analysis",
              "First selection is the baseline. Deltas are direction-aware — lower latency / cost is better.")
    picks = st.multiselect("experiments", ids, default=ids[: min(2, len(ids))], label_visibility="collapsed")
    if len(picks) < 2:
        theme.empty("Select at least two experiments",
                    "Pick a baseline and one or more variants to compare metrics and tradeoffs.")
        return

    report = compare_experiments([result_of(i) for i in picks])
    base, others = report.baseline_id, [i for i in picks if i != report.baseline_id]

    if report.config_diff:
        theme.sec("Configuration", "Diff")
        theme.table(["Field"] + [escape(short(i)) for i in picks],
                    [[escape(cd.field)] + [theme.mono(cd.values.get(i)) for i in picks]
                     for cd in report.config_diff],
                    max_w=200 + 190 * len(picks))

    theme.sec("Metrics", "Baseline vs variants")
    blocks = []
    for fam_name, keys in CMP_FAMILIES:
        fam = [mc for mc in report.metrics if mc.key in keys and mc.values.get(base) is not None]
        if not fam:
            continue
        rows = []
        for mc in fam:
            bf = mc.values.get(base) if mc.unit == "" else None
            cols = []
            for oid in others:
                ov = mc.values.get(oid)
                of = ov if mc.unit == "" else None
                cols.append((fmtval(ov, mc.unit), of, cmp_delta(mc.deltas.get(oid), mc.unit)))
            rows.append({"label": mc.label, "base": (fmtval(mc.values.get(base), mc.unit), bf), "cols": cols})
        blocks.append((fam_name, rows))
    theme.compare(blocks, short(base), [short(o) for o in others])

    theme.sec("Tradeoffs", "Summary", "No winner is declared — quality / cost / latency is your call.")
    for t in report.tradeoffs:
        theme.tradeoff(t.experiment_id, t.summary, t.gains, t.losses)


def failures(r: ExperimentResult) -> None:
    c1, c2, _ = st.columns([1, 1, 2])
    corr = c1.slider("correctness min", 0.0, 1.0, 0.5, 0.05)
    faith = c2.slider("faithfulness min", 0.0, 1.0, 0.7, 0.05)
    fa = analyze_failures(r, thresholds=FailureThresholds(correctness_min=corr, faithfulness_min=faith))

    theme.sec("Outcome distribution", "Triage")
    total = sum(fa.category_counts.values()) or 1
    order = sorted(fa.category_counts.items(), key=lambda kv: -kv[1])
    segs = "".join(f'<span style="width:{v/total*100:.2f}%;background:{CAT_COLOR.get(k, "#8A909B")}"></span>'
                   for k, v in order)
    pills = "&nbsp;&nbsp;".join(theme.pill(f"{k} {v}", CAT_TONE.get(k, "")) for k, v in order)
    st.markdown(f'<div style="display:flex;height:10px;border-radius:99px;overflow:hidden;border:1px solid var(--line);'
                f'background:var(--bg-1);margin:2px 0 10px">{segs}</div><div>{pills}</div>', unsafe_allow_html=True)

    boards = [
        ("Lowest recall", fa.lowest_recall, lambda v: f3(v)),
        ("Lowest faithfulness", fa.lowest_faithfulness, lambda v: f3(v)),
        ("Lowest correctness", fa.lowest_correctness, lambda v: f3(v)),
        ("Highest latency", fa.highest_latency, lambda v: f"{v:.0f}ms"),
        ("Highest cost", fa.highest_cost, lambda v: f"${v:.5f}"),
    ]
    boards = [b for b in boards if b[1]]
    for i in range(0, len(boards), 2):
        cols = st.columns(2, gap="medium")
        for col, (title, brd, fmt) in zip(cols, boards[i:i + 2]):
            with col:
                theme.sec(title, "Worst 5")
                theme.table(["Q", "Question", "Value"],
                            [[theme.mono(x.question_id), escape(x.question[:46]), theme.mono(fmt(x.value))]
                             for x in brd], num_cols=[2])

    theme.sec(f"Failing questions · {len(fa.failures)}", "Detail")
    if not fa.failures:
        theme.empty("No failures at these thresholds",
                    "Every answerable question passed. Loosen the sliders to probe marginal cases.")
        return
    theme.qrows([(theme.esc(d.question_id), theme.pill(d.category.value, CAT_TONE.get(d.category.value, "")),
                  f"{d.question}  —  {d.reason}") for d in fa.failures])


def traces(r: ExperimentResult) -> None:
    all_traces = [Trace.model_validate_json(t) for t in _traces_json(r.experiment_id)]
    if not all_traces:
        theme.empty("No traces stored",
                    "This run had <code>tracing_enabled = false</code>. Re-run with tracing on.")
        return
    labels = [t.question_id or t.trace_id for t in all_traces]
    pick = st.selectbox("question", labels, label_visibility="collapsed")
    tr = next(t for t in all_traces if (t.question_id or t.trace_id) == pick)
    expected = next((q.generation.expected_answer for q in r.per_question
                     if q.question_id == tr.question_id and q.generation is not None), None)

    st.markdown(f'<div style="font-size:14px;font-weight:500;margin:6px 0 12px;color:var(--ink)">{escape(tr.question)}</div>',
                unsafe_allow_html=True)
    for err in tr.errors:
        st.error(err)

    p = tr.performance
    left, right = st.columns([2, 3], gap="medium")
    with left:
        theme.hbar("Stage timeline · ms",
                   [("embedding", p.embedding_ms), ("retrieval", p.retrieval_ms), ("reranking", p.reranking_ms),
                    ("generation", p.generation_ms), ("evaluation", p.evaluation_ms)],
                   fmt=lambda v: f"{v:.0f}", colors=STAGE_C)
        theme.kv("Performance", [
            ("total", f"{p.total_ms:.0f} ms"),
            ("tokens", f"{p.token_usage.total_tokens:,}"),
            ("prompt / completion", f"{p.token_usage.prompt_tokens} / {p.token_usage.completion_tokens}"),
            ("cost", f"${p.estimated_cost_usd:.6f}"),
        ])
    with right:
        if tr.generation:
            gt = tr.generation
            theme.answer_block("Answer", gt.answer)
            if expected:
                theme.answer_block("Expected", expected, "exp")
        if tr.evaluation:
            et = tr.evaluation
            theme.statlist("Evaluation", [
                {"label": "Correctness", "value": f3(et.correctness), "frac": et.correctness},
                {"label": "Relevance", "value": f3(et.relevance), "frac": et.relevance},
                {"label": "Faithfulness", "value": f3(et.faithfulness.score) if et.faithfulness else "—",
                 "frac": et.faithfulness.score if et.faithfulness else None},
            ])

    if tr.retrieval:
        rt = tr.retrieval
        theme.sec("Retrieved context", "Stage",
                  f"{rt.embedding_model} · dim {rt.embedding_dim} · top_k {rt.top_k}")
        peak = max((c.score for c in rt.chunks), default=1.0) or 1.0
        theme.table(["#", "Score", "Document", "Chunk", "Preview"],
                    [[theme.mono(c.rank),
                      f"{theme.mono(f'{c.score:.4f}')} {theme.meter(c.score / peak, 't2')}",
                      escape(c.document_id), theme.mono(c.chunk_id), escape(c.text_preview[:78])]
                     for c in rt.chunks], num_cols=[0])

    if tr.generation and tr.generation.citations:
        theme.sec("Citations", "Stage")
        theme.table(["Marker", "Status", "Chunk", "Document"],
                    [[theme.mono(f"[{c.marker}]"),
                      theme.pill("resolved" if c.exists else "hallucinated", "pos" if c.exists else "neg"),
                      theme.mono(c.chunk_id or "—"), escape(c.document_id or "—")]
                     for c in tr.generation.citations])

    if tr.evaluation and tr.evaluation.faithfulness and tr.evaluation.faithfulness.claims:
        theme.sec("Claim verdicts", "Stage")
        theme.table(["Claim", "Verdict", "Reason"],
                    [[escape(cl.text[:70]),
                      theme.pill("supported" if cl.supported else "unsupported", "pos" if cl.supported else "neg"),
                      escape(cl.reason[:60])]
                     for cl in tr.evaluation.faithfulness.claims])
    if tr.generation:
        with st.expander("Prompt sent to the model"):
            st.code(tr.generation.prompt or "")


def slices(r: ExperimentResult) -> None:
    rep = slice_report(r)
    c1, c2, _ = st.columns([1, 1, 2])
    metric = c1.selectbox("flag metric",
                          ["recall", "precision", "correctness", "faithfulness", "citation_precision"])
    gap = c2.slider("gap vs overall", 0.0, 0.5, 0.05, 0.01)

    theme.sec("Performance by slice", "Segments",
              "Cells are tinted by distance from the overall value — greener above, redder below.")
    mcols = ["recall", "precision", "correctness", "faithfulness", "citation_precision"]
    ov = {m: rep.overall.metric(m) for m in mcols}
    rows = []
    for sm in [rep.overall, *rep.slices]:
        is_o = sm.label == "overall"
        line = (["__HL__"] if is_o else []) + [
            f"<b>{escape(sm.label)}</b>" if is_o else escape(sm.label), theme.mono(sm.num_questions)]
        for m in mcols:
            v = sm.metric(m)
            if v is None:
                line.append('<span style="color:var(--ink-4)">—</span>')
            else:
                stl = "" if is_o else tint(v, ov[m])
                line.append(f'<span style="{stl};padding:1px 6px;border-radius:4px;'
                            f'font-family:var(--mono);font-feature-settings:\'tnum\'">{v:.3f}</span>')
        line += [theme.mono(f"{sm.latency_total_ms:.0f}"), theme.mono(f"${sm.cost_per_query_usd:.5f}")]
        rows.append(line)
    theme.table(["Slice", "N", "Recall", "Precision", "Correctness", "Faithful", "Cite prec", "Latency", "Cost"],
                rows, num_cols=[1, 7, 8])

    under = underperforming_slices(rep, metric=metric, min_gap=gap)
    theme.sec("Underperformance", "Flags", f"slices > {gap:.2f} below overall on '{metric}'")
    if not under:
        theme.empty("No weak slices", f"Every slice is within {gap:.2f} of the overall <code>{metric}</code>.")
    else:
        theme.table(["Slice", metric, "Overall", "Gap"],
                    [[escape(u.label), theme.mono(f"{u.slice_value:.3f}"), theme.mono(f"{u.overall_value:.3f}"),
                      theme.delta(f"{u.gap:+.3f}", "regressed")] for u in under], num_cols=[1, 2])


# --- shell --------------------------------------------------------------------------------
_MARK = (
    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M12 2.2 21.8 12 12 21.8 2.2 12Z" stroke="currentColor" stroke-width="2" '
    'stroke-linejoin="round" opacity=".4"/>'
    '<path d="M12 7 17 12 12 17 7 12Z" stroke="currentColor" stroke-width="2" '
    'stroke-linejoin="round" opacity=".85"/>'
    '<circle cx="12" cy="12" r="1.9" fill="currentColor"/></svg>')


def _brand() -> None:
    st.markdown(f'<div class="brand"><div class="mk">{_MARK}</div>'
                '<div class="nm">Assay<span>evaluation&nbsp;·&nbsp;observability</span></div></div>',
                unsafe_allow_html=True)


def main() -> None:
    summaries = _summaries()
    if not summaries:
        with st.sidebar:
            _brand()
        theme.topbar("Overview", right="0 experiments")
        theme.empty("No experiments yet",
                    "Run one:&nbsp; <code>python -m scripts.run_experiment --name demo --limit 5</code>")
        return

    ids = [s["experiment_id"] for s in summaries]
    with st.sidebar:
        _brand()
        st.markdown('<div class="s-lbl">Workspace</div>', unsafe_allow_html=True)
        section = st.radio("nav", SECTIONS, label_visibility="collapsed")
        st.markdown('<div class="s-lbl">Experiment</div>', unsafe_allow_html=True)
        selected = st.selectbox("experiment", ids, label_visibility="collapsed")
        sm = next(s for s in summaries if s["experiment_id"] == selected)
        st.markdown(
            '<div class="mini">'
            f'<div class="mrow">recall<b>{f3(sm.get("retrieval_recall"))}</b></div>'
            f'<div class="mrow">correctness<b>{f3(sm.get("judge_correctness"))}</b></div>'
            f'<div class="mrow">cost<b>${sm.get("estimated_cost_usd") or 0:.4f}</b></div>'
            '</div>', unsafe_allow_html=True)
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(f'<div class="s-foot">assay · {len(summaries)} runs tracked</div>', unsafe_allow_html=True)

    theme.topbar(section, selected)

    if section == "Experiments":
        experiments(summaries)
        return
    if section == "Comparison":
        comparison(ids)
        return

    result = result_of(selected)
    if result is None:
        theme.empty("Could not load experiment", f"<code>{escape(selected)}</code> is missing from the store.")
        return

    vitals_for(result)
    {"Overview": overview, "Failures": failures, "Traces": traces, "Slices": slices}[section](result)


main()

"""RAG Eval — evaluation & observability dashboard (Phase 15).

    streamlit run dashboard/app.py

Reads only from the experiment store (data/experiments.db). Every number here comes
from the same functions the CLI uses; this is a presentation layer. The look lives
in dashboard/theme.py and .streamlit/config.toml.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

from app.experiment.comparison import compare_experiments
from app.experiment.failure_analysis import FailureThresholds, analyze_failures
from app.experiment.results import ExperimentResult
from app.experiment.slicing import slice_report, underperforming_slices
from app.experiment.store import ExperimentStore
from app.observability.trace import Trace
from dashboard import theme

st.set_page_config(page_title="RAG Eval", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
theme.inject()

SECTIONS = ["Overview", "Experiments", "Comparison", "Failures", "Traces", "Slices"]
CAT_TONE = {
    "OK": "pos", "INSUFFICIENT_CONTEXT": "accent", "RETRIEVAL_FAILURE": "neg",
    "GENERATION_FAILURE": "neg", "HALLUCINATION": "neg", "CITATION_FAILURE": "warn", "ERROR": "neg",
}
CMP_FAMILIES = [
    ("Retrieval", {"hit_rate", "precision", "recall", "mrr", "ndcg"}),
    ("Generation", {"correctness", "relevance", "exact_match", "token_f1"}),
    ("Faithfulness", {"faithfulness"}),
    ("Citations", {"citation_precision", "citation_completeness", "citation_correctness", "citation_hallucination_rate"}),
    ("Performance", {"latency_total_ms", "latency_total_p95_ms", "latency_generation_ms",
                     "latency_evaluation_ms", "cost_per_query_usd", "cost_total_usd"}),
]


# --- data (cached JSON, re-parsed to models) ---------------------------------------
@st.cache_data(show_spinner=False)
def _summaries() -> list[dict]:
    with ExperimentStore() as store:
        return [s.model_dump() for s in store.list(limit=200)]


@st.cache_data(show_spinner=False)
def _result_json(experiment_id: str) -> Optional[str]:
    with ExperimentStore() as store:
        r = store.get(experiment_id)
        return r.model_dump_json() if r else None


@st.cache_data(show_spinner=False)
def _traces_json(experiment_id: str) -> list[str]:
    with ExperimentStore() as store:
        return [t.model_dump_json() for t in store.get_traces(experiment_id)]


def result_of(experiment_id: str) -> Optional[ExperimentResult]:
    raw = _result_json(experiment_id)
    return ExperimentResult.model_validate_json(raw) if raw else None


# --- formatting -------------------------------------------------------------------------
def f3(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def short(experiment_id: str) -> str:
    return experiment_id.rsplit("_", 1)[0]


def fmtval(v, unit: str) -> str:
    if v is None:
        return theme.mono("—")
    if unit == "$":
        return theme.mono(f"${v:.6f}")
    if unit == "ms":
        return theme.mono(f"{v:.0f} ms")
    return theme.mono(f"{v:.3f}")


def cmp_delta(d, unit: str) -> str:
    if d is None:
        return '<span class="delta delta--flat">–</span>'
    if unit == "ms":
        body = f"{d.absolute:+.0f} ms"
    elif unit == "$":
        body = f"{d.absolute:+.6f}"
    elif d.percent is not None:
        body = f"{d.percent:+.1f}%"
    else:
        body = f"{d.absolute:+.3f}"
    return theme.delta(body, d.direction)


def bar_cell(v) -> str:
    return "—" if v is None else f"{theme.mono(f3(v))}&nbsp;&nbsp;{theme.bar(v)}"


def tint(v, overall) -> str:
    if overall is None:
        return ""
    d = v - overall
    if d >= -1e-9:
        return f"background:rgba(70,178,107,{min(0.16, 0.04 + abs(d) * 0.5):.3f})"
    return f"background:rgba(229,84,75,{min(0.22, 0.06 + abs(d) * 0.7):.3f})"


# --- sections ----------------------------------------------------------------------------
def overview(r: ExperimentResult) -> None:
    c = r.config
    theme.chips([
        ("model", c.generation_model),
        ("chunk", f"{c.chunk_size}/{c.chunk_overlap}"),
        ("top_k", str(c.top_k)),
        ("questions", f"{r.num_questions}" + (f" · {r.num_errors} err" if r.num_errors else "")),
        ("corpus", f"{r.document_count}d / {r.chunk_count}c"),
        ("$/query", f"{r.cost.cost_per_query_usd:.6f}"),
        ("tokens", f"{r.total_token_usage.total_tokens:,}"),
        ("date", r.started_at[:10]),
    ])

    if r.retrieval:
        m = r.retrieval
        theme.sec("Retrieval quality", "Evaluation")
        theme.kpis([
            {"label": "Hit@K", "value": f3(m.hit_rate), "bar": m.hit_rate},
            {"label": "Precision@K", "value": f3(m.precision), "bar": m.precision},
            {"label": "Recall@K", "value": f3(m.recall), "bar": m.recall},
            {"label": "MRR", "value": f3(m.mrr), "bar": m.mrr},
            {"label": "NDCG@K", "value": f3(m.ndcg), "bar": m.ndcg},
        ], cols=5)

    if r.generation:
        g = r.generation
        theme.sec("Generation quality", "Evaluation")
        theme.kpis([
            {"label": "Correctness", "value": f3(g.judge_correctness), "bar": g.judge_correctness},
            {"label": "Relevance", "value": f3(g.judge_relevance), "bar": g.judge_relevance},
            {"label": "Exact match", "value": f3(g.exact_match), "bar": g.exact_match},
            {"label": "Token F1", "value": f3(g.token_f1), "bar": g.token_f1},
            {"label": "Abstention acc", "value": f3(g.abstention_accuracy), "bar": g.abstention_accuracy},
        ], cols=5)

    if r.faithfulness:
        fa = r.faithfulness
        theme.sec("Faithfulness", "Evaluation")
        theme.kpis([
            {"label": "Macro", "value": f3(fa.faithfulness), "bar": fa.faithfulness,
             "sub": f"{fa.num_scored} scored"},
            {"label": "Micro · claim rate", "value": f3(fa.claim_support_rate), "bar": fa.claim_support_rate,
             "sub": f"{fa.total_supported}/{fa.total_claims} claims"},
        ], cols=2)

    if r.citation:
        ct = r.citation
        theme.sec("Citations", "Evaluation")
        theme.kpis([
            {"label": "Completeness", "value": f3(ct.citation_completeness), "bar": ct.citation_completeness},
            {"label": "Precision", "value": f3(ct.citation_precision), "bar": ct.citation_precision},
            {"label": "Correctness", "value": f3(ct.citation_correctness), "bar": ct.citation_correctness},
            {"label": "Hallucination", "value": f3(ct.citation_hallucination_rate),
             "bar": ct.citation_hallucination_rate, "sub": f"{ct.total_hallucinated_links} links"},
        ], cols=4)

    theme.sec("Latency", "Performance", "milliseconds per stage, aggregated across questions")
    if r.latency_report:
        peak = max((s.p95_ms for s in r.latency_report.stages.values()), default=1.0) or 1.0
        rows = [
            [escape(name), theme.mono(f"{s.mean_ms:.0f}"), theme.mono(f"{s.median_ms:.0f}"),
             theme.mono(f"{s.p95_ms:.0f}"), theme.mono(f"{s.max_ms:.0f}"), theme.bar(s.p95_ms / peak)]
            for name, s in r.latency_report.stages.items()
        ]
        theme.table(["Stage", "Mean", "Median", "P95", "Max", "P95 →"], rows, num_cols=[1, 2, 3, 4])

    theme.sec("Cost", "Performance", "USD, this run")
    cc = r.cost
    theme.table(
        ["Component", "USD"],
        [
            ["Ingestion embedding · one-time", theme.mono(f"${cc.ingestion_embedding_usd:.6f}")],
            ["Query embedding", theme.mono(f"${cc.query_embedding_usd:.6f}")],
            ["Generation", theme.mono(f"${cc.generation_usd:.6f}")],
            ["Evaluation", theme.mono(f"${cc.evaluation_usd:.6f}")],
            ["__HL__", "Total", theme.mono(f"${cc.total_usd:.6f}")],
            ["Per query · marginal", theme.mono(f"${cc.cost_per_query_usd:.6f}")],
        ],
        num_cols=[1],
    )


def experiments(summaries: list[dict]) -> None:
    theme.sec("Tracked experiments", "Registry", f"{len(summaries)} runs, newest first")
    rows = [
        [
            theme.mono(s["experiment_id"]),
            theme.pill(s["generation_model"].replace("claude-", ""), "accent"),
            theme.mono(s["chunk_size"]),
            theme.mono(s["top_k"]),
            theme.mono(s["num_questions"]),
            bar_cell(s.get("retrieval_recall")),
            bar_cell(s.get("judge_correctness")),
            theme.mono(f'${s.get("estimated_cost_usd") or 0:.4f}'),
        ]
        for s in summaries
    ]
    theme.table(["Experiment", "Model", "Chunk", "K", "N", "Recall", "Correctness", "Cost"],
                rows, num_cols=[2, 3, 4, 7])
    with st.expander("Raw table · sortable"):
        st.dataframe(summaries, width="stretch", hide_index=True)


def comparison(ids: list[str]) -> None:
    theme.sec("Compare experiments", "Analysis",
              "First selection is the baseline. Deltas are direction-aware — lower latency / cost is 'better'.")
    picks = st.multiselect("experiments", ids, default=ids[: min(2, len(ids))], label_visibility="collapsed")
    if len(picks) < 2:
        theme.empty("Select at least two experiments",
                    "Pick a baseline and one or more variants to compare metrics and tradeoffs.")
        return

    report = compare_experiments([result_of(i) for i in picks])
    base = report.baseline_id
    others = [i for i in picks if i != base]

    if report.config_diff:
        theme.sec("Configuration", "Diff")
        rows = [[escape(cd.field)] + [theme.mono(cd.values.get(i)) for i in picks] for cd in report.config_diff]
        theme.table(["Field"] + [escape(short(i)) for i in picks], rows)

    for fam_name, keys in CMP_FAMILIES:
        fam = [mc for mc in report.metrics if mc.key in keys]
        if not any(mc.values.get(base) is not None for mc in fam):
            continue
        theme.sec(fam_name, "Metrics")
        rows = []
        for mc in fam:
            row = [escape(mc.label), fmtval(mc.values.get(base), mc.unit)]
            for oid in others:
                row.append(f"{fmtval(mc.values.get(oid), mc.unit)}&nbsp;&nbsp;{cmp_delta(mc.deltas.get(oid), mc.unit)}")
            rows.append(row)
        theme.table(
            ["Metric", f"{short(base)} · baseline"] + [escape(short(o)) for o in others],
            rows, num_cols=list(range(1, len(others) + 2)),
        )

    theme.sec("Tradeoffs", "Summary",
              "What each variant gains and loses vs the baseline. No winner is declared.")
    for t in report.tradeoffs:
        lines = [("pos", g) for g in t.gains] + [("neg", loss) for loss in t.losses]
        theme.card(f"{t.experiment_id} — {t.summary}", lines or [("", "no material differences")])


def failures(r: ExperimentResult) -> None:
    theme.sec("Failure analysis", "Diagnostics",
              "Each question gets one primary category. Tune the thresholds below.")
    c1, c2, _ = st.columns([1, 1, 2])
    corr = c1.slider("correctness_min", 0.0, 1.0, 0.5, 0.05)
    faith = c2.slider("faithfulness_min", 0.0, 1.0, 0.7, 0.05)
    fa = analyze_failures(r, thresholds=FailureThresholds(correctness_min=corr, faithfulness_min=faith))

    pills = "".join(
        theme.pill(f"{k} · {v}", CAT_TONE.get(k, ""))
        for k, v in sorted(fa.category_counts.items(), key=lambda kv: -kv[1])
    )
    st.markdown(f'<div class="chips">{pills}</div>', unsafe_allow_html=True)

    boards = [
        ("Lowest recall", fa.lowest_recall, False),
        ("Lowest faithfulness", fa.lowest_faithfulness, False),
        ("Lowest correctness", fa.lowest_correctness, False),
        ("Highest latency", fa.highest_latency, True),
        ("Highest cost", fa.highest_cost, True),
    ]
    boards = [b for b in boards if b[1]]
    for i in range(0, len(boards), 2):
        cols = st.columns(2)
        for col, (title, brd, is_ms) in zip(cols, boards[i:i + 2]):
            with col:
                theme.sec(title, "Worst 5")
                rows = [
                    [theme.mono(rq.question_id), escape(rq.question[:42]),
                     theme.mono(f"{rq.value:.0f}" if is_ms else f"{rq.value:.3f}")]
                    for rq in brd
                ]
                theme.table(["Q", "Question", "Value"], rows, num_cols=[2])

    theme.sec(f"Failing questions · {len(fa.failures)}", "Detail")
    if not fa.failures:
        theme.empty("No failures at these thresholds",
                    "Every answerable question passed. Loosen the sliders to probe marginal cases.")
        return
    for d in fa.failures:
        theme.qcard(d.question_id, d.category.value, CAT_TONE.get(d.category.value, ""),
                    d.question, d.reason)


def traces(r: ExperimentResult) -> None:
    theme.sec("Execution trace", "Observability", "Full detail of one question's run.")
    all_traces = [Trace.model_validate_json(t) for t in _traces_json(r.experiment_id)]
    if not all_traces:
        theme.empty("No traces stored",
                    "This run had <code>tracing_enabled = false</code>. Re-run with tracing on.")
        return

    labels = [t.question_id or t.trace_id for t in all_traces]
    pick = st.selectbox("question", labels, label_visibility="collapsed")
    tr = next(t for t in all_traces if (t.question_id or t.trace_id) == pick)
    expected = next(
        (q.generation.expected_answer for q in r.per_question
         if q.question_id == tr.question_id and q.generation is not None),
        None,
    )

    st.markdown(
        f'<div style="font-size:15px;font-weight:500;margin:8px 0 2px;color:var(--ink)">{escape(tr.question)}</div>',
        unsafe_allow_html=True,
    )
    for err in tr.errors:
        st.error(err)

    p = tr.performance
    theme.sec("Stage timeline", "Performance",
              f"total {p.total_ms:.0f} ms · {p.token_usage.total_tokens:,} tokens · ${p.estimated_cost_usd:.6f}")
    theme.timeline([
        ("embedding", p.embedding_ms), ("retrieval", p.retrieval_ms), ("reranking", p.reranking_ms),
        ("generation", p.generation_ms), ("evaluation", p.evaluation_ms),
    ])

    if tr.retrieval:
        rt = tr.retrieval
        theme.sec("Retrieval", "Stage", f"{rt.embedding_model} · dim {rt.embedding_dim} · top_k {rt.top_k}")
        peak = max((c.score for c in rt.chunks), default=1.0) or 1.0
        rows = [
            [theme.mono(c.rank),
             f"{theme.mono(f'{c.score:.4f}')}&nbsp;&nbsp;{theme.bar(c.score / peak)}",
             escape(c.document_id), theme.mono(c.chunk_id), escape(c.text_preview[:72])]
            for c in rt.chunks
        ]
        theme.table(["#", "Score", "Document", "Chunk", "Preview"], rows, num_cols=[0])

    if tr.generation:
        gt = tr.generation
        theme.sec("Generation", "Stage", gt.model)
        theme.answer_block("Answer", gt.answer)
        if expected:
            theme.answer_block("Expected", expected, "exp")
        if gt.token_usage:
            theme.chips([("prompt tok", str(gt.token_usage.prompt_tokens)),
                         ("completion tok", str(gt.token_usage.completion_tokens))])
        if gt.citations:
            rows = [
                [theme.mono(f"[{c.marker}]"),
                 theme.pill("resolved" if c.exists else "hallucinated", "pos" if c.exists else "neg"),
                 theme.mono(c.chunk_id or "—"), escape(c.document_id or "—")]
                for c in gt.citations
            ]
            theme.table(["Marker", "Status", "Chunk", "Document"], rows)
        with st.expander("Prompt sent to the model"):
            st.code(gt.prompt or "")

    if tr.evaluation:
        et = tr.evaluation
        theme.sec("Evaluation", "Stage")
        theme.kpis([
            {"label": "Correctness", "value": f3(et.correctness), "bar": et.correctness},
            {"label": "Relevance", "value": f3(et.relevance), "bar": et.relevance},
            {"label": "Faithfulness", "value": f3(et.faithfulness.score) if et.faithfulness else "—",
             "bar": et.faithfulness.score if et.faithfulness else None},
        ], cols=3)
        if et.faithfulness and et.faithfulness.claims:
            rows = [
                [escape(cl.text[:66]),
                 theme.pill("supported" if cl.supported else "unsupported", "pos" if cl.supported else "neg"),
                 escape(cl.reason[:56])]
                for cl in et.faithfulness.claims
            ]
            theme.table(["Claim", "Verdict", "Reason"], rows)
        if et.citation and et.citation.links:
            rows = []
            for lk in et.citation.links:
                if lk.supports_claim:
                    verdict, tone = "supports", "pos"
                elif not lk.exists:
                    verdict, tone = "hallucinated", "neg"
                else:
                    verdict, tone = "unsupported", "warn"
                rows.append([theme.mono(f"[{lk.marker}]"), theme.pill(verdict, tone), escape(lk.claim_text[:56])])
            theme.table(["Marker", "Verdict", "Claim"], rows)


def slices(r: ExperimentResult) -> None:
    theme.sec("Performance by slice", "Segments",
              "Every metric per dataset-slice label. Cells are tinted by distance from the overall value.")
    rep = slice_report(r)
    c1, c2, _ = st.columns([1, 1, 2])
    metric = c1.selectbox("flag metric",
                          ["recall", "precision", "correctness", "faithfulness", "citation_precision"])
    gap = c2.slider("gap vs overall", 0.0, 0.5, 0.05, 0.01)

    metric_cols = ["recall", "precision", "correctness", "faithfulness", "citation_precision"]
    overall_vals = {m: rep.overall.metric(m) for m in metric_cols}

    rows = []
    for sm in [rep.overall, *rep.slices]:
        is_overall = sm.label == "overall"
        line = ["__HL__"] if is_overall else []
        line.append(f"<b>{escape(sm.label)}</b>" if is_overall else escape(sm.label))
        line.append(theme.mono(sm.num_questions))
        for m in metric_cols:
            v = sm.metric(m)
            if v is None:
                line.append('<span style="color:var(--ink-3)">—</span>')
                continue
            style = "" if is_overall else tint(v, overall_vals[m])
            line.append(
                f'<span style="{style};padding:2px 7px;border-radius:5px;'
                f'font-family:var(--mono);font-feature-settings:\'tnum\'">{v:.3f}</span>'
            )
        line.append(theme.mono(f"{sm.latency_total_ms:.0f}"))
        line.append(theme.mono(f"${sm.cost_per_query_usd:.5f}"))
        rows.append(line)
    theme.table(
        ["Slice", "N", "Recall", "Precision", "Correctness", "Faithfulness", "Cite prec", "Latency", "Cost"],
        rows, num_cols=[1, 7, 8],
    )

    under = underperforming_slices(rep, metric=metric, min_gap=gap)
    theme.sec("Underperformance", "Flags", f"slices more than {gap:.2f} below overall on '{metric}'")
    if not under:
        theme.empty("No weak slices", f"Every slice is within {gap:.2f} of the overall <code>{metric}</code>.")
    else:
        rows = [
            [escape(u.label), theme.mono(f"{u.slice_value:.3f}"), theme.mono(f"{u.overall_value:.3f}"),
             theme.delta(f"{u.gap:+.3f}", "regressed")]
            for u in under
        ]
        theme.table(["Slice", metric, "Overall", "Gap"], rows, num_cols=[1, 2])


# --- shell --------------------------------------------------------------------------------
def _brand() -> None:
    st.markdown(
        '<div class="brand"><div class="mark">R</div>'
        '<div class="name">RAG&nbsp;Eval<span>evaluation &amp; observability</span></div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    summaries = _summaries()

    if not summaries:
        with st.sidebar:
            _brand()
        theme.topbar("Overview", None, 0)
        theme.empty(
            "No experiments yet",
            "Run one from the CLI:&nbsp; <code>python -m scripts.run_experiment --name demo --limit 5</code>",
        )
        return

    ids = [s["experiment_id"] for s in summaries]
    with st.sidebar:
        _brand()
        st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
        section = st.radio("nav", SECTIONS, label_visibility="collapsed")
        st.markdown('<div class="side-label">Experiment</div>', unsafe_allow_html=True)
        selected = st.selectbox("experiment", ids, label_visibility="collapsed")
        st.caption(f"{len(summaries)} tracked")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    theme.topbar(section, selected, len(summaries))

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

    {"Overview": overview, "Failures": failures, "Traces": traces, "Slices": slices}[section](result)


main()

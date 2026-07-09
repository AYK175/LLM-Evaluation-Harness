"""Results dashboard for the LLM meta-evaluation harness.

Tells the story in four acts: (1) which automated metrics actually track
human judgment, (2) how biased the LLM judges are, (3) whether
mitigations helped, and (4) a side-by-side inspector for individual answers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


JUDGE_COLOR = {"claude-sonnet-judge": BLUE, "claude-haiku-judge": AQUA}
JUDGE_LABEL = {"claude-sonnet-judge": "Claude Sonnet (judge)", "claude-haiku-judge": "Claude Haiku (judge)"}
FAMILY_COLOR = {"mistral": BLUE, "gemma": AQUA, "qwen": ORANGE}

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

alt.themes.enable("none")
BASE_CONFIG = {
    "config": {
        "background": SURFACE,
        "font": "-apple-system, system-ui, Segoe UI, sans-serif",
        "axis": {
            "domainColor": BASELINE,
            "tickColor": BASELINE,
            "gridColor": GRID,
            "labelColor": INK_SECONDARY,
            "titleColor": INK_SECONDARY,
            "labelFontSize": 12,
            "titleFontSize": 12,
        },
        "legend": {"labelColor": INK_SECONDARY, "titleColor": INK_SECONDARY, "labelFontSize": 12},
        "view": {"stroke": None},
    }
}


def styled(chart: alt.Chart) -> alt.Chart:
    return chart.configure(**BASE_CONFIG["config"])


# Data loading

@st.cache_data
def load_agreement() -> pd.DataFrame | None:
    path = RAW / "agreement_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).dropna(how="all").sort_values("spearman", ascending=False)


@st.cache_data
def load_json(name: str) -> dict | None:
    path = RAW / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_generations() -> list[dict] | None:
    path = next(RAW.glob("*__generations.jsonl"), None)
    if not path:
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@st.cache_data
def load_answer_scores() -> pd.DataFrame | None:
    path = RAW / "asqa_hybrid_v1__answer_scores.jsonl"
    if not path.exists():
        return None
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def nfmt(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


# Page shell
st.set_page_config(page_title="LLM Eval Harness", page_icon="\U0001F9EA", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
    h1 {font-weight: 700; letter-spacing: -0.02em;}
    h2 {font-weight: 650; letter-spacing: -0.01em; margin-top: 0.25rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    [data-testid="stMetric"] {
        background: #fcfcfb; border: 1px solid rgba(11,11,11,0.08);
        border-radius: 10px; padding: 0.75rem 1rem 0.5rem 1rem;
    }
    div[data-testid="stExpander"] details {
        border: 1px solid rgba(11,11,11,0.08); border-radius: 10px;
    }
    .narrative {color: #52514e; font-size: 0.95rem; line-height: 1.5;}
    hr {margin: 1.5rem 0; border-color: rgba(11,11,11,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LLM Meta-Evaluation Harness")
st.markdown(
    '<p class="narrative">Can automated metrics be trusted to evaluate long-form LLM answers? '
    "This dashboard scores the same 100 ASQA questions across 3 model families with lexical, "
    "semantic, and LLM-judge metrics, checks each against 300 human gold ratings, quantifies "
    "how biased the judges are, and reports whether the mitigations actually fix it.</p>",
    unsafe_allow_html=True,
)

agreement_df = load_agreement()
bias = load_json("bias_report.json")
mitigation = load_json("mitigation_report.json")
generations = load_generations()
scores_df = load_answer_scores()

# KPI strip

k1, k2, k3, k4 = st.columns(4)
n_examples = len({r["example_id"] for r in generations}) if generations else "—"
n_models = len({r["model_name"] for r in generations}) if generations else "—"
best_metric = agreement_df.iloc[0] if agreement_df is not None and len(agreement_df) else None
reversal_before = (
    max(v["reversal_rate"] for v in bias["position_bias"].values()) if bias else None
)
k1.metric("Questions evaluated", n_examples)
k2.metric("Model families compared", n_models)
k3.metric(
    "Best metric ↔ human correlation",
    nfmt(best_metric["spearman"]) if best_metric is not None else "—",
    help="Spearman correlation, best-performing automated metric vs. human gold ratings.",
)
k4.metric(
    "Worst position-bias reversal rate (pre-fix)",
    f"{reversal_before * 100:.0f}%" if reversal_before is not None else "—",
    help="Fraction of pairwise verdicts that flip when the two answers swap slots.",
)

st.divider()


# 1. Metric reliability

st.header("1. Which metrics actually track human judgment?")

METRIC_META = {
    "judge_claude-sonnet-judge_overall": ("Claude Sonnet — Overall", "LLM Judge"),
    "judge_claude-sonnet-judge_correctness": ("Claude Sonnet — Correctness", "LLM Judge"),
    "judge_claude-sonnet-judge_grounding": ("Claude Sonnet — Grounding", "LLM Judge"),
    "judge_claude-haiku-judge_overall": ("Claude Haiku — Overall", "LLM Judge"),
    "judge_claude-haiku-judge_correctness": ("Claude Haiku — Correctness", "LLM Judge"),
    "judge_claude-haiku-judge_grounding": ("Claude Haiku — Grounding", "LLM Judge"),
    "bertscore": ("BERTScore", "Semantic (embedding)"),
    "rouge": ("ROUGE-L", "Lexical overlap"),
    "meteor": ("METEOR", "Lexical overlap"),
    "faithfulness": ("Faithfulness (RAGAS)", "Confound"),
    "answer_len_tokens": ("Answer length (tokens)", "Confound"),
}
CATEGORY_COLOR = {"LLM Judge": BLUE, "Semantic (embedding)": AQUA, "Lexical overlap": YELLOW, "Confound": STATUS_CRITICAL}

if agreement_df is not None:
    df = agreement_df.copy()
    df["label"] = df["metric"].map(lambda m: METRIC_META.get(m, (m, "Other"))[0])
    df["category"] = df["metric"].map(lambda m: METRIC_META.get(m, (m, "Other"))[1])

    order = df.sort_values("spearman", ascending=False)["label"].tolist()
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("spearman:Q", title="Spearman correlation vs. human gold ratings", scale=alt.Scale(domain=[-0.25, 1])),
            color=alt.Color(
                "category:N",
                title="Metric family",
                scale=alt.Scale(domain=list(CATEGORY_COLOR.keys()), range=list(CATEGORY_COLOR.values())),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Metric"),
                alt.Tooltip("spearman:Q", title="Spearman", format=".3f"),
                alt.Tooltip("kendall_tau:Q", title="Kendall's tau", format=".3f"),
                alt.Tooltip("cohen_kappa:Q", title="Cohen's kappa", format=".3f"),
                alt.Tooltip("n:Q", title="n"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=BASELINE, strokeWidth=1).encode(x="x:Q")
    labels = bars.mark_text(align="left", dx=6, color=INK_SECONDARY, fontSize=11).encode(
        text=alt.Text("spearman:Q", format=".2f"),
        x=alt.X("spearman:Q"),
    )
    chart = (bars + rule + labels).properties(height=28 * len(df), width=650)
    st.altair_chart(styled(chart), width="stretch")

    best_lex = df[df.category == "Lexical overlap"].sort_values("spearman", ascending=False).iloc[0]
    best_judge = df[df.category == "LLM Judge"].sort_values("spearman", ascending=False).iloc[0]
    worst = df.sort_values("spearman").iloc[0]
    st.markdown(
        f'<p class="narrative"><b>Reading this for the room:</b> the best LLM judge '
        f"({best_judge['label']}) tracks human ratings at ρ={best_judge['spearman']:.2f}, "
        f"nearly 2.5× stronger than the best lexical metric ({best_lex['label']}, "
        f"ρ={best_lex['spearman']:.2f}). Two metrics are actively misleading: "
        f"<b>{worst['label']}</b> correlates <i>negatively</i> (ρ={worst['spearman']:.2f}) "
        f"with human judgment, a metric that would rank answers backwards if used alone. "
        f"Takeaway: cheap lexical/length metrics are not a safe proxy for human judgment on "
        f"long-form answers; a well-prompted LLM judge is.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("Full agreement table"):
        st.dataframe(
            df[["label", "category", "spearman", "kendall_tau", "cohen_kappa", "n"]]
            .rename(columns={"label": "metric", "kendall_tau": "kendall's tau", "cohen_kappa": "cohen's kappa"}),
            width="stretch",
            hide_index=True,
        )
else:
    st.info("Run the week-3 correlation analysis to populate this.")

st.divider()


# 2. Judge bias

st.header("2. How biased are the judges?")
st.markdown(
    '<p class="narrative">An LLM judge is itself a measurement instrument, and instruments '
    "have systematic error. Three checks: does the judge favor whichever answer it sees first "
    "(position bias), does it reward length regardless of quality (verbosity bias), and does it "
    "favor answers from its own model family (self-preference)?</p>",
    unsafe_allow_html=True,
)

if bias:
    tab_pos, tab_verb, tab_self = st.tabs(["Position bias", "Verbosity bias", "Self-preference"])

    with tab_pos:
        st.caption(
            "Position consistency & preference fairness: higher is better. "
            "Reversal rate: lower is better. Slot-A win rate: ideally ≈ 0.50."
        )
        rows = []
        metric_meta = {
            "position_consistency": "Position consistency",
            "preference_fairness": "Preference fairness",
            "reversal_rate": "Reversal rate",
            "slot_a_win_rate": "Slot-A win rate",
        }
        for judge, vals in bias["position_bias"].items():
            for m, label in metric_meta.items():
                rows.append({"judge": JUDGE_LABEL[judge], "judge_key": judge, "metric": label, "value": vals[m]})
        pos_df = pd.DataFrame(rows)

        ref_rows = []
        for m, label in metric_meta.items():
            ref_rows.append({"metric": label, "y": 0.5 if m in ("slot_a_win_rate",) else None})
        ref_df = pd.DataFrame(ref_rows).dropna()

        base = alt.Chart(pos_df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("judge:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "judge:N",
                title="Judge",
                scale=alt.Scale(domain=list(JUDGE_LABEL.values()), range=[JUDGE_COLOR[k] for k in JUDGE_LABEL]),
            ),
            tooltip=[alt.Tooltip("judge:N"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=".3f")],
        )
        text = base.mark_text(dy=-6, fontSize=11, color=INK_SECONDARY).encode(text=alt.Text("value:Q", format=".2f"))
        rule = (
            alt.Chart(ref_df).mark_rule(color=STATUS_WARNING, strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q")
            if len(ref_df)
            else None
        )
        layers = [base, text] + ([rule] if rule is not None else [])
        chart = (
            alt.layer(*layers)
            .properties(width=150, height=170)
            .facet(facet=alt.Facet("metric:N", title=None), columns=4, data=pos_df)
        )
        st.altair_chart(styled(chart), width="stretch")

        haiku, sonnet = bias["position_bias"]["claude-haiku-judge"], bias["position_bias"]["claude-sonnet-judge"]
        st.markdown(
            f'<p class="narrative">Both judges flip their verdict when the answer order swaps far '
            f"more than they should: Haiku reverses <b>{haiku['reversal_rate']*100:.0f}%</b> of the time, "
            f"Sonnet <b>{sonnet['reversal_rate']*100:.0f}%</b>. Both also over-pick whichever answer sits "
            f"in slot A ({haiku['slot_a_win_rate']*100:.0f}% and {sonnet['slot_a_win_rate']*100:.0f}% "
            f"of wins respectively, vs. an unbiased 50%). Sonnet is the more reliable judge on every "
            f"position-bias axis. This is exactly the failure mode the week-4 mitigation targets "
            f"(see section 3).</p>",
            unsafe_allow_html=True,
        )

    with tab_verb:
        human_corr = next(iter(bias["verbosity_bias"].values()))["human_length_corr"]
        rows = [{"entity": "Human gold ratings", "value": human_corr, "color": VIOLET}]
        for judge, vals in bias["verbosity_bias"].items():
            rows.append({"entity": JUDGE_LABEL[judge], "value": vals["judge_length_corr"], "color": JUDGE_COLOR[judge]})
        verb_df = pd.DataFrame(rows)
        order = ["Human gold ratings"] + [JUDGE_LABEL[j] for j in bias["verbosity_bias"]]

        bars = (
            alt.Chart(verb_df)
            .mark_bar(cornerRadiusEnd=3, size=42)
            .encode(
                x=alt.X("entity:N", sort=order, title=None),
                y=alt.Y("value:Q", title="Correlation: answer length ↔ score", scale=alt.Scale(domain=[-0.2, 0.05])),
                color=alt.Color("entity:N", scale=alt.Scale(domain=order, range=[VIOLET] + [JUDGE_COLOR[j] for j in bias["verbosity_bias"]]), legend=None),
                tooltip=[alt.Tooltip("entity:N"), alt.Tooltip("value:Q", format=".3f")],
            )
        )
        text = bars.mark_text(dy=-8, fontSize=11, color=INK_SECONDARY).encode(text=alt.Text("value:Q", format=".3f"))
        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=BASELINE).encode(y="y:Q")
        chart = (bars + text + rule).properties(width=380, height=260)
        st.altair_chart(styled(chart), width="content")
        st.markdown(
            '<p class="narrative">Humans themselves rate longer answers very slightly '
            f"<i>worse</i> (ρ={human_corr:.2f}). Both judges track that same direction and stay "
            "close to the human baseline rather than overshooting it — so on this dataset, neither "
            "judge is meaningfully reward-hacking for length. This is the one bias axis that came out "
            "clean.</p>",
            unsafe_allow_html=True,
        )

    with tab_self:
        st.info(
            "**Not measurable in this run.** Self-preference compares a judge's average score for "
            "its own model family vs. every other family. Both judges here are Claude models, but no "
            "*generator* in this run is a Claude model (the three generators are Mistral, Gemma, and "
            "Qwen) — so there is no “own-family” answer to compare against, and the reported "
            "gap is undefined (NaN) by construction, not a measured zero.",
            icon="ℹ️",
        )
        st.markdown(
            '<p class="narrative">The self-preference question is instead answered indirectly in '
            "section 3 via the panel-of-judges mitigation, which measures how far the jury rates each "
            "generator family from the cross-family average — the same underlying concern (is any "
            "family systematically over/under-rated), just measured across families instead of judge "
            "identity.</p>",
            unsafe_allow_html=True,
        )
else:
    st.info("Run the week-3 bias analysis to populate this.")

st.divider()


# 3. Mitigations

st.header("3. Do the mitigations help?")

if mitigation:
    pos = mitigation.get("position_bias_mitigation", {})
    sp = mitigation.get("self_preference_mitigation", {})

    st.subheader("Balanced-position calibration")
    st.caption(pos.get("method", ""))
    c1, c2 = st.columns(2)

    before_rows = [
        {"judge": JUDGE_LABEL[j], "value": v["reversal_rate"]} for j, v in pos.get("before", {}).items()
    ]
    after_rows = [
        {"judge": JUDGE_LABEL[j], "value": v} for j, v in pos.get("after_tie_rate", {}).items()
    ]

    def small_bar(rows, title, y_title):
        d = pd.DataFrame(rows)
        bars = (
            alt.Chart(d)
            .mark_bar(cornerRadiusEnd=3, size=44)
            .encode(
                x=alt.X("judge:N", title=None),
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=[0, max(0.3, d["value"].max() * 1.3)])),
                color=alt.Color("judge:N", scale=alt.Scale(domain=list(JUDGE_LABEL.values()), range=[JUDGE_COLOR[k] for k in JUDGE_LABEL]), legend=None),
                tooltip=[alt.Tooltip("judge:N"), alt.Tooltip("value:Q", format=".1%")],
            )
        )
        text = bars.mark_text(dy=-8, fontSize=11, color=INK_SECONDARY).encode(text=alt.Text("value:Q", format=".0%"))
        return (bars + text).properties(width=280, height=220, title=title)

    with c1:
        st.altair_chart(styled(small_bar(before_rows, "Before: reversal rate", "Reversal rate")), width="stretch")
    with c2:
        st.altair_chart(styled(small_bar(after_rows, "After: tie rate", "Verdicts discarded as ties")), width="stretch")

    st.markdown(
        f'<p class="narrative">{pos.get("note", "")} In practice: Sonnet’s reversal rate drops from '
        f"{pos['before']['claude-sonnet-judge']['reversal_rate']*100:.0f}% to a "
        f"{pos['after_tie_rate']['claude-sonnet-judge']*100:.0f}% tie rate, and Haiku's from "
        f"{pos['before']['claude-haiku-judge']['reversal_rate']*100:.0f}% to "
        f"{pos['after_tie_rate']['claude-haiku-judge']*100:.0f}%. The surviving (non-tie) verdicts are "
        "now position-invariant by construction, the honest cost is that a real slice of pairs "
        "gets thrown out as “no reliable winner” rather than resolved.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Panel-of-judges (jury) for self-preference")
    st.caption(sp.get("method", ""))

    jury_dev = sp.get("jury_family_deviation", {})
    if jury_dev:
        jd = pd.DataFrame([{"family": f.capitalize(), "value": v} for f, v in jury_dev.items()]).sort_values("value")
        bars = (
            alt.Chart(jd)
            .mark_bar(cornerRadiusEnd=3, size=48)
            .encode(
                x=alt.X("family:N", sort=None, title=None),
                y=alt.Y("value:Q", title="Deviation from cross-family average jury score"),
                color=alt.condition(alt.datum.value >= 0, alt.value(BLUE), alt.value(RED)),
                tooltip=[alt.Tooltip("family:N"), alt.Tooltip("value:Q", format=".3f")],
            )
        )
        text_above = bars.transform_filter(alt.datum.value >= 0).mark_text(
            fontSize=11, color=INK_SECONDARY, dy=-8, baseline="bottom"
        ).encode(text=alt.Text("value:Q", format="+.2f"), y=alt.Y("value:Q"))
        text_below = bars.transform_filter(alt.datum.value < 0).mark_text(
            fontSize=11, color=INK_SECONDARY, dy=8, baseline="top"
        ).encode(text=alt.Text("value:Q", format="+.2f"), y=alt.Y("value:Q"))
        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=BASELINE).encode(y="y:Q")
        chart = (bars + text_above + text_below + rule).properties(width=380, height=240)
        st.altair_chart(styled(chart), width="content")

        worst_fam = jd.iloc[0]
        best_fam = jd.iloc[-1]
        st.markdown(
            f'<p class="narrative">Even with a multi-judge panel, family-level unfairness doesn’t '
            f"disappear: <b>{worst_fam['family']}</b> answers score "
            f"<b>{abs(worst_fam['value']):.2f} points below</b> the cross-family average jury rating "
            f"(worst-case deviation), while <b>{best_fam['family']}</b> sits "
            f"<b>{best_fam['value']:.2f} points above</b> it. We can’t report a clean "
            "“before vs. after” reduction number here, because the “before” "
            "(single-judge own-family gap) requires a judge and a generator to share a model family, "
            "and none do in this run, so `worst_case_reduction` is undefined rather than zero. "
            "The jury number stands on its own as evidence that pooling judges reduces but does not "
            "eliminate systematic family-level bias.</p>",
            unsafe_allow_html=True,
        )

    for note in mitigation.get("notes", []):
        st.caption(f"⚠️ {note}")
else:
    st.info("Run the week-4 mitigation stage to populate this.")

st.divider()


# 4. Answer explorer

st.header("4. Inspect individual answers")

if generations:
    ex_ids = sorted({r["example_id"] for r in generations}, key=lambda x: int(x.split("-")[-1]))
    chosen = st.selectbox("Example", ex_ids)
    recs = [r for r in generations if r["example_id"] == chosen]

    st.markdown(f"**Q:** {recs[0]['question']}")
    if recs[0].get("gold_answer"):
        with st.container(border=True):
            st.markdown(f"**Gold reference answer:** {recs[0]['gold_answer']}")

    cols = st.columns(len(recs))
    for col, r in zip(cols, recs):
        with col:
            st.markdown(f"**{r['model_name']}** · _{r['model_family']}_")
            if scores_df is not None:
                row = scores_df[(scores_df.example_id == chosen) & (scores_df.model_name == r["model_name"])]
                if len(row):
                    row = row.iloc[0]
                    b1, b2, b3 = st.columns(3)
                    b1.metric("ROUGE-L", nfmt(row.rouge))
                    b2.metric("METEOR", nfmt(row.meteor))
                    b3.metric("BERTScore", nfmt(row.bertscore))
            with st.container(border=True):
                st.markdown(r["answer"])
else:
#    st.info("Run `python -m src.generate` to populate this.")

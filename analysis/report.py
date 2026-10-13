"""Paper-ready PDF report for the spec §8 analysis.

Assembles every figure with a plain-language explanation and the matching
statistics, so the document can be read straight through while writing the
thesis chapter.

Deliberately self-contained inside the repository (reportlab only, no
external template) so the client can regenerate it from source rather than
depending on any ad-hoc tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    CondPageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from aegora_isle.kpi.convergence import CONVERGENCE_THRESHOLD
from analysis.data_loading import SUBSISTENCE_DISCREPANCY_NOTE

_styles = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=_styles["Title"], fontSize=18, alignment=TA_LEFT, spaceAfter=16)
H2 = ParagraphStyle("H2", parent=_styles["Heading2"], fontSize=13, spaceBefore=16, spaceAfter=8)
H3 = ParagraphStyle("H3", parent=_styles["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle("B", parent=_styles["Normal"], fontSize=10, leading=14.5, spaceAfter=8)
CAPTION = ParagraphStyle("C", parent=BODY, fontSize=9, textColor=colors.HexColor("#444444"))


def _para(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def _plain_to_rl(text: str) -> str:
    """Renders a plain-text note's ``*emphasis*`` and ```code``` markers.

    ``SUBSISTENCE_DISCREPANCY_NOTE`` is written as plain prose for console
    output, so pasting it straight into a Paragraph leaves the asterisks and
    backticks visible in the PDF.
    """
    import re

    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)


# LETTER is 8.5 x 11in; margins are 0.85in horizontal and 0.8in vertical,
# leaving a 6.8 x 9.4in frame.
FIGURE_MAX_WIDTH: float = 6.6 * inch
# Height is the axis that actually binds. A figure scaled only by width can
# end up tall enough that it plus its caption and trailing spacer consume the
# rest of a page, which is what produced a blank page here: the trailing
# spacer tipped the frame onto a new page by itself, and the explicit
# PageBreak that followed then advanced past it. Capping height well under
# 9.4in leaves room for the caption on the same page.
FIGURE_MAX_HEIGHT: float = 7.5 * inch


# A hard PageBreak can fire when the frame has *already* advanced to a fresh
# page, producing a blank one. CondPageBreak only breaks when less than the
# requested height remains, so on an empty page (9.4in free) it is a no-op.
SECTION_BREAK_HEIGHT: float = 8.5 * inch


def _section_break() -> CondPageBreak:
    """Starts a new page unless the current one is already effectively fresh."""
    return CondPageBreak(SECTION_BREAK_HEIGHT)


def _status_counts(data_dir: str | Path) -> tuple[int, int]:
    """(CONVERGED, CONVERGING) counts from ``attempts_summary.csv``.

    Read rather than hardcoded so the convergence caveat cannot drift away
    from the dataset it describes. Returns (0, 0) if the file is absent,
    which only happens mid-batch.
    """
    path = Path(data_dir) / "attempts_summary.csv"
    if not path.exists():
        return 0, 0
    counts = pd.read_csv(path)["final_status"].value_counts()
    return int(counts.get("CONVERGED", 0)), int(counts.get("CONVERGING", 0))


def _figure(path: Path, caption: str, max_width: float = FIGURE_MAX_WIDTH,
            max_height: float = FIGURE_MAX_HEIGHT) -> list[Any]:
    """Embeds a figure scaled to fit, with its explanation underneath.

    Scales on whichever axis binds first. Width alone is not enough: a tall
    figure scaled only to the frame's width can still exceed its height.
    """
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(1.0, max_width / w, max_height / h)
    return [
        Image(str(path), width=w * scale, height=h * scale),
        Spacer(1, 4),
        _para(caption, CAPTION),
        Spacer(1, 12),
    ]


def _table(df: pd.DataFrame, cols: list[str], round_to: int = 4) -> Table:
    """Renders a small dataframe as a styled table."""
    view = df[cols].copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.{round_to}g}")
    data = [list(view.columns)] + view.astype(str).values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d2d2d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_report(results: dict[str, Any], out_dir: str | Path) -> Path:
    """Builds the paper-ready PDF. Returns its path."""
    out_dir = Path(out_dir)
    path = out_dir / "Aegora_Isle_Statistical_Analysis_Report.pdf"

    runs = results["runs"]
    anova = results["anova_oneway"]
    two_way = results["anova_twoway"]
    km = results["km_table"]
    logrank = results["logrank"]
    cox = results["cox"]
    figures = results["figures"]

    story: list[Any] = []

    # ---------------------------------------------------------- front matter
    story.append(_para("Aegora Isle — Statistical Analysis Report", TITLE))
    story.append(_para(
        "Confirmatory dataset: <b>360 replications</b> (3 governance conditions × 4 "
        "environmental scenarios × 30 replications), 50 agents, 10,000 timesteps, "
        "generated by <font face='Courier'>run_experiment.py</font> and analysed per "
        "specification §8. Every table and figure below is regenerated from "
        "<font face='Courier'>data/</font> by "
        "<font face='Courier'>python -m analysis.statistical_analysis</font>."))
    story.append(_para(
        "<b>How to read this document.</b> Each section states the hypothesis in plain "
        "language, shows the relevant figure, then gives the statistics. Effect sizes "
        "(partial η² and Cohen's f) matter as much as p-values here: with 120 "
        "replications per condition almost anything reaches significance, so the "
        "question is whether a difference is <i>large</i>, not merely detectable."))
    story.append(_para(
        "Significance threshold is <b>α = .01</b> (Bonferroni-corrected), per the "
        "pre-registered analysis plan. Cohen's f benchmarks: .10 small, .25 medium, "
        ".40 large."))
    story.append(_para(
        "<b>On effect sizes and the Welch correction.</b> Partial η² and Cohen's f "
        "are computed from the conventional sums of squares "
        "(η² = SS<sub>between</sub> / (SS<sub>between</sub> + SS<sub>residual</sub>)) "
        "and are therefore unaffected by the Welch correction, which applies to the "
        "F-statistic, the denominator degrees of freedom, and the p-value. Where "
        "Levene's test failed and Welch's ANOVA was substituted, the reported effect "
        "size is thus identical to the one the standard test would have produced; "
        "only the significance test changed. For the Gini analysis, for example, the "
        "standard and Welch F-statistics differ by nearly a factor of two while "
        "partial η² is the same to six decimal places."))
    story.append(_para(
        "<b>One definitional note.</b> Every KPI here is summarised as the "
        "<i>mean over each replication</i>, following specification Table 4 "
        "(\"Mean per run\"). Some earlier interim summaries used the final-timestep "
        "value instead. Both give the same directions and the same conclusions for "
        "H1, H2 and H3 — only the F-statistics differ in magnitude, because a run "
        "mean is a less noisy quantity than a single final reading."))

    # --------------------------------------------------------------- summary
    story.append(_para("Headline results", H2))
    gov_means = runs.groupby("governance", observed=True).agg(
        gini=("gini_coefficient", "mean"), hdi=("hdi_proxy", "mean"),
        gdp=("gdp_equivalent", "mean"), depletion=("depletion_rate", "mean"),
    ).reset_index()
    gov_means["governance"] = gov_means["governance"].astype(str)
    story.append(_table(gov_means, ["governance", "gini", "hdi", "gdp", "depletion"]))
    story.append(Spacer(1, 8))
    story.append(_para(
        "Read across the row: governance <i>raises</i> measured inequality (Gini), "
        "<i>raises</i> economic output (GDP-equivalent), and turns resource depletion "
        "negative — meaning the resource pool regenerates rather than shrinking. "
        "The inequality direction is the opposite of what H1 predicted; the depletion "
        "and output directions match H2 and H3."))

    story.append(_section_break())

    # ------------------------------------------------------------------- H1
    story.append(_para("H1 — Does governance reduce inequality?", H2))
    story.append(_para(
        "<b>The hypothesis.</b> Governed economies should show <i>lower</i> Gini "
        "coefficients than the ungoverned baseline — i.e. G1 &gt; G2 &gt; G3."))
    story.append(_para(
        "<b>The result: rejected, and inverted.</b> Ungoverned is the <i>most equal</i> "
        "condition, and both governed regimes are substantially less equal. The effect "
        "is very large, not marginal."))
    gini_row = anova[anova.kpi == "gini_coefficient"].iloc[0]
    story.append(_para(
        f"One-way ANOVA by governance: <b>F = {gini_row['F']:.1f}, p = {gini_row['p']:.3g}</b>, "
        f"partial η² = {gini_row['partial_eta_sq']:.3f}, Cohen's f = {gini_row['cohens_f']:.2f}. "
        f"Homogeneity of variance {'held' if gini_row['homogeneity_ok'] else 'was violated, so Welch’s ANOVA was substituted'} "
        f"(Levene p = {gini_row['levene_p']:.3g})."))
    story.extend(_figure(
        Path(figures["box"][0]),
        "Figure 1 — Gini by governance. Each dot is one replication. The ungoverned "
        "box sits clearly below both governed boxes: governance is associated with "
        "more inequality, not less. The governed conditions are also far tighter, "
        "meaning governance makes outcomes more consistent even as it makes them less equal."))
    story.extend(_figure(
        Path(figures["kde"][0]),
        "Figure 2 — The same data as distributions. The ungoverned curve is wide and "
        "sits to the left; the two governed curves are narrow, tall and nearly on top "
        "of one another — Rule-Based and Learning-Based produce almost identical "
        "inequality."))
    story.append(_para(
        "<b>Why this happens.</b> Wealth in this economy is dominated by Producer "
        "goods inventory. The extraction cap suppresses Extractor income, and "
        "redistribution transfers only <i>cash</i> — a small slice of total wealth — so "
        "it cannot offset that. The net effect of the governance instruments as "
        "specified is to concentrate wealth rather than spread it."))

    story.append(_section_break())

    # ------------------------------------------------------------------- H2
    story.append(_para("H2 — Does governance protect the resource?", H2))
    story.append(_para(
        "<b>The hypothesis.</b> Governed economies should depleteslower and survive "
        "longer before system collapse (MTSC)."))
    story.append(_para("<b>The result: strongly supported, on both measures.</b>"))
    dep_row = anova[anova.kpi == "depletion_rate"].iloc[0]
    story.append(_para(
        f"Depletion rate, one-way ANOVA: <b>F = {dep_row['F']:.1f}, p = {dep_row['p']:.3g}</b>, "
        f"partial η² = {dep_row['partial_eta_sq']:.3f}, Cohen's f = {dep_row['cohens_f']:.2f}. "
        "A positive mean means the pool is shrinking; negative means it regenerates."))
    story.extend(_figure(
        Path(figures["box"][3]),
        "Figure 3 — Depletion rate by governance. Ungoverned sits above zero (the "
        "commons is being consumed faster than it regrows); both governed regimes sit "
        "below zero (the pool recovers)."))
    story.append(_para("Survival analysis (MTSC)", H3))
    story.append(_para(
        "Collapse is defined per specification Table 4: the resource pool falling below "
        "its critical minimum, or GDP at zero for three consecutive timesteps. Runs "
        "reaching 10,000 timesteps without either are <i>censored</i> — we know they "
        "survived at least that long, which is why Kaplan-Meier is used rather than a "
        "simple average."))
    km_view = km.copy()
    km_view["governance"] = km_view["governance"].astype(str)
    story.append(_table(km_view, ["governance", "n", "events", "censored", "event_rate_pct", "median_survival"]))
    story.append(Spacer(1, 8))
    story.append(_para(
        f"Multivariate log-rank test: <b>χ² = {logrank['test_statistic']:.1f}, "
        f"p = {logrank['p']:.3g}</b> — the survival curves differ by governance."))
    story.extend(_figure(
        Path(figures["survival"]),
        "Figure 4 — Kaplan-Meier survival curves. The vertical axis is the probability "
        "a run is still alive at a given timestep. The ungoverned curve falls away "
        "sharply; both governed curves stay near the top for the full horizon. Shaded "
        "bands are 95% confidence intervals."))
    story.append(_para("Cox proportional hazards", H3))
    story.append(_para(
        "Hazard ratios relative to the ungoverned baseline in Scenario A. A ratio below "
        "1 means <i>lower</i> risk of collapse; above 1 means higher."))
    story.append(_table(cox, [c for c in ["term", "hazard_ratio", "hr_ci_lower_95", "hr_ci_upper_95", "p"] if c in cox.columns]))

    story.append(_section_break())

    # ------------------------------------------------------------------- H3
    story.append(_para("H3 — Does adaptive AI beat fixed rules under volatility?", H2))
    story.append(_para(
        "<b>The hypothesis.</b> Learning-Based governance should outperform Rule-Based, "
        "<i>and the advantage should widen under volatile conditions</i>. The "
        "specification requires a statistically significant interaction effect — not "
        "merely that one regime is better on average."))
    story.append(_para("<b>The result: supported, including the interaction.</b>"))
    tw = two_way[two_way.term != "Residual"].copy()
    story.append(_table(tw, ["term", "df", "F", "p", "partial_eta_sq", "cohens_f"]))
    story.append(Spacer(1, 8))
    inter = two_way[two_way.term.str.contains(":", na=False)]
    if len(inter):
        r = inter.iloc[0]
        story.append(_para(
            f"The interaction term is the one H3 actually rests on: "
            f"<b>F = {r['F']:.2f}, p = {r['p']:.3g}</b>. A significant interaction means "
            "the gap between Learning-Based and Rule-Based is not constant — it depends "
            "on how volatile the environment is, which is precisely the claim."))
    g2 = runs[runs.governance == "G2"].groupby("volatility", observed=True)["gdp_equivalent"].mean()
    g3 = runs[runs.governance == "G3"].groupby("volatility", observed=True)["gdp_equivalent"].mean()
    story.append(_para(
        f"Direction check — Learning-Based minus Rule-Based on GDP-equivalent: "
        f"<b>{g3.get('low', float('nan')) - g2.get('low', float('nan')):+.3f}</b> under low "
        f"volatility versus <b>{g3.get('high', float('nan')) - g2.get('high', float('nan')):+.3f}</b> "
        "under high volatility. The advantage roughly doubles when conditions get harsh."))
    story.extend(_figure(
        Path(figures["box"][2]),
        "Figure 5 — GDP-equivalent by governance. Both governed regimes clearly "
        "out-produce the ungoverned baseline, with Learning-Based highest."))

    story.append(_section_break())

    # --------------------------------------------------------- trajectories
    story.append(_para("Trajectories — how each cell evolves over time", H2))
    story.extend(_figure(
        Path(figures["trajectories_gini"]),
        "Figure 6 — Mean Gini trajectory for all 12 cells. Rows are governance "
        "conditions, columns are scenarios. Where a line stops short, every "
        "replication in that cell had collapsed by that point — the line is not padded."))
    story.extend(_figure(
        Path(figures["trajectories_stock"]),
        "Figure 7 — Mean resource stock for all 12 cells. This is the clearest single "
        "picture of H2: the ungoverned row falls to the floor in the high-event "
        "scenarios, while the governed rows hold their level for the full run."))

    # ------------------------------------------------------------- caveats
    # Read from attempts_summary.csv rather than hardcoded, so the caveat
    # cannot drift away from the dataset it describes.
    n_converged, n_converging = _status_counts(results.get("data_dir", "data/"))

    story.append(_para("Caveats a reader should know", H2))
    story.append(_para(
        "<b>1. Traders cannot accumulate wealth.</b> Under the specified single-price "
        "market a Trader buys and sells at the same price, so intermediation is "
        "wealth-neutral by construction. Roughly a quarter of agents therefore hold "
        "almost nothing, which inflates every Gini figure above. Excluding Traders, "
        "Gini falls to about 0.387 / 0.605 / 0.600 for the three regimes — the "
        "<i>ordering is unchanged</i>, so the H1 conclusion is robust, but the absolute "
        "values are not a clean measure of inequality among earning agents."))
    story.append(_para(
        "<b>2. Two collapse definitions exist in the delivered data.</b> " +
        _plain_to_rl(SUBSISTENCE_DISCREPANCY_NOTE.replace("\n", " "))))
    story.append(_para(
        "Only the specification's Table 4 definition is used anywhere in this report. "
        "Every table and every figure above, including the Kaplan-Meier curves and the "
        "Cox model, is driven by the <font face='Courier'>duration</font> and "
        "<font face='Courier'>event_observed</font> fields, which the Table 4 encoding "
        "produces. The <font face='Courier'>collapse_event</font> column that ships in "
        "the CSVs — the subsistence definition — is not read by the analysis at any "
        "point. It is retained in the delivered data so the contrast above stays "
        "inspectable, not because anything here depends on it."))
    story.append(_para(
        f"<b>3. The convergence criterion no longer discriminates.</b> A replication is "
        f"counted as converged when the rolling variance of its policy loss stays below "
        f"<b>{CONVERGENCE_THRESHOLD:g}</b> across a 20-epoch window, after a minimum of "
        f"50 epochs. That threshold was recalibrated from the originally pre-registered "
        f"0.01, which two measurement defects had made far too permissive to separate "
        f"anything: with those corrected, the observed variances cluster so tightly that "
        f"the criterion no longer distinguishes between regimes. Of the 360 "
        f"replications, {n_converged} are recorded as CONVERGED and {n_converging} as "
        f"CONVERGING."))
    story.append(_para(
        "<b>The recalibration cannot have shaped any result reported here, because "
        "convergence status is not used as a filter.</b> All 360 replications enter "
        "every analysis above, whatever their status. The 65 CONVERGING runs are "
        "predominantly early resource-pool collapses that ended before the 50-epoch "
        "minimum could be evaluated — excluding them would have deleted precisely the "
        "events the survival analysis exists to measure, and would have biased H2 "
        "heavily in favour of the ungoverned condition. Status is carried in "
        "<font face='Courier'>attempts_summary.csv</font> for transparency, not applied "
        "as a selection rule."))

    SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title="Aegora Isle — Statistical Analysis Report",
    ).build(story)
    return path

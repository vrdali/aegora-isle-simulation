"""Spec §8 statistical analysis for the Aegora Isle confirmatory dataset.

Produces every output required by spec §8.1, writing tables to
``analysis/outputs/tables/`` and figures to ``analysis/outputs/figures/``
per §8.2, plus a console summary and a paper-ready PDF.

Run:

    uv run python -m analysis.statistical_analysis --data_dir data/

Statistical conventions, per the client-confirmed plan recorded in
CLAUDE.md:

* H1/H2 -- one-way ANOVA by governance with Tukey HSD post-hoc,
  **Bonferroni-corrected to alpha = .01**.
* H3 -- two-way ANOVA (governance x volatility) with the interaction term.
* MTSC -- Kaplan-Meier plus Cox PH, censored at 10,000 steps, using the
  spec Table 4 event definition (see ``analysis.data_loading``).

Levene's test is run for every one-way ANOVA and Welch's variant is
substituted automatically when homogeneity of variance is violated, as
spec §8.1 requires.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats

from analysis.data_loading import (
    GOVERNANCE_ORDER,
    KPI_COLUMNS,
    SUBSISTENCE_DISCREPANCY_NOTE,
    load_run_index,
    write_epoch_summaries,
)

# Client-confirmed (CLAUDE.md "Resolved questions"): H1/H2 use Bonferroni
# correction to alpha = .01, not the conventional .05.
ALPHA: float = 0.01

GOVERNANCE_LABELS: dict[str, str] = {
    "G1": "No-Governance",
    "G2": "Rule-Based",
    "G3": "Learning-Based",
}


# ------------------------------------------------------------ effect sizes


def cohens_f(partial_eta_sq: float) -> float:
    """Cohen's f from partial eta-squared: f = sqrt(eta2 / (1 - eta2)).

    Spec §8.1 requires both. Conventional benchmarks: .10 small, .25 medium,
    .40 large.
    """
    eta = float(partial_eta_sq)
    if not 0.0 <= eta < 1.0:
        return float("nan")
    return float(np.sqrt(eta / (1.0 - eta)))


# --------------------------------------------------------------- 8.1 (1)


def descriptive_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, min, max, skewness, kurtosis for every KPI x cell."""
    records = []
    for (gov, scen), grp in runs.groupby(["governance", "scenario"], observed=True):
        for kpi in KPI_COLUMNS:
            v = grp[kpi].dropna().to_numpy()
            records.append({
                "governance": gov, "scenario": scen, "kpi": kpi, "n": len(v),
                "mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "max": v.max(),
                "skewness": stats.skew(v), "kurtosis": stats.kurtosis(v),
            })
        d = grp["duration"].to_numpy()
        records.append({
            "governance": gov, "scenario": scen, "kpi": "mtsc_duration", "n": len(d),
            "mean": d.mean(), "sd": d.std(ddof=1), "min": d.min(), "max": d.max(),
            "skewness": stats.skew(d), "kurtosis": stats.kurtosis(d),
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------- 8.1 (2,4,5)


def one_way_anova_by_governance(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-way ANOVA per KPI by governance, with Levene -> Welch fallback.

    Returns ``(anova_table, tukey_table)``. The ANOVA table carries F, p,
    partial eta-squared, Cohen's f, Levene's p, and which test was actually
    used -- so a reader can see where homogeneity failed and Welch was
    substituted rather than having to infer it.
    """
    anova_rows: list[dict[str, Any]] = []
    tukey_frames: list[pd.DataFrame] = []

    for kpi in KPI_COLUMNS:
        sub = runs[["governance", kpi]].dropna().copy()
        sub["governance"] = sub["governance"].astype(str)

        lev = pg.homoscedasticity(data=sub, dv=kpi, group="governance", method="levene")
        equal_var = bool(lev["equal_var"].iloc[0])
        levene_p = float(lev["pval"].iloc[0])

        if equal_var:
            aov = pg.anova(data=sub, dv=kpi, between="governance", detailed=True)
            F = float(aov.loc[0, "F"])
            p = float(aov.loc[0, "p_unc"])
            eta = float(aov.loc[0, "np2"])
            test_used = "one-way ANOVA"
        else:
            aov = pg.welch_anova(data=sub, dv=kpi, between="governance")
            F = float(aov.loc[0, "F"])
            p = float(aov.loc[0, "p_unc"])
            eta = float(aov.loc[0, "np2"])
            test_used = "Welch ANOVA (Levene violated)"

        anova_rows.append({
            "kpi": kpi, "test": test_used, "F": F, "p": p,
            "partial_eta_sq": eta, "cohens_f": cohens_f(eta),
            "levene_p": levene_p, "homogeneity_ok": equal_var,
            "alpha": ALPHA, "significant": p < ALPHA,
        })

        tk = pg.pairwise_tukey(data=sub, dv=kpi, between="governance")
        tk.insert(0, "kpi", kpi)
        tukey_frames.append(tk)

    return pd.DataFrame(anova_rows), pd.concat(tukey_frames, ignore_index=True)


# --------------------------------------------------------------- 8.1 (3)


def two_way_anova(runs: pd.DataFrame, dv: str = "gdp_equivalent") -> pd.DataFrame:
    """Two-way ANOVA: governance x volatility, **with the interaction term**.

    H3 as specified requires "a statistically significant interaction
    effect", so the interaction row is the load-bearing one here, not the
    main effects.
    """
    sub = runs[["governance", "volatility", dv]].dropna().copy()
    sub["governance"] = sub["governance"].astype(str)
    model = smf.ols(f"{dv} ~ C(governance) * C(volatility)", data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2)

    ss_total = table["sum_sq"].sum()
    ss_resid = table.loc["Residual", "sum_sq"]
    out = table.reset_index().rename(
        columns={"index": "term", "PR(>F)": "p", "sum_sq": "sum_sq", "df": "df"}
    )
    out["eta_sq"] = out["sum_sq"] / ss_total
    out["partial_eta_sq"] = out.apply(
        lambda r: np.nan if r["term"] == "Residual"
        else r["sum_sq"] / (r["sum_sq"] + ss_resid), axis=1
    )
    out["cohens_f"] = out["partial_eta_sq"].apply(
        lambda e: cohens_f(e) if pd.notna(e) else np.nan
    )
    out["dv"] = dv
    return out


# --------------------------------------------------------------- 8.1 (6)


def kaplan_meier(runs: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Kaplan-Meier per governance condition, plus multivariate log-rank.

    Returns ``(fitters_by_governance, summary_table, logrank_result)``.
    """
    fitters: dict[str, Any] = {}
    rows = []
    for gov in GOVERNANCE_ORDER:
        grp = runs[runs["governance"] == gov]
        if grp.empty:
            continue
        kmf = KaplanMeierFitter(label=f"{gov} ({GOVERNANCE_LABELS[gov]})")
        kmf.fit(grp["duration"], event_observed=grp["event_observed"])
        fitters[gov] = kmf
        median = kmf.median_survival_time_
        rows.append({
            "governance": gov,
            "n": len(grp),
            "events": int(grp["event_observed"].sum()),
            "censored": int((1 - grp["event_observed"]).sum()),
            "event_rate_pct": 100.0 * grp["event_observed"].mean(),
            "median_survival": median,
            "mean_duration": float(grp["duration"].mean()),
        })

    logrank = multivariate_logrank_test(
        runs["duration"], runs["governance"].astype(str), runs["event_observed"]
    )
    return fitters, pd.DataFrame(rows), {
        "test_statistic": float(logrank.test_statistic), "p": float(logrank.p_value),
    }


# --------------------------------------------------------------- 8.1 (7)


def cox_hazard_ratios(runs: pd.DataFrame) -> pd.DataFrame:
    """Cox proportional hazards: governance + scenario as covariates.

    Dummy-coded with G1 and Scenario A as references, so hazard ratios read
    as "relative to ungoverned, in the stable scenario".
    """
    df = runs[["duration", "event_observed", "governance", "scenario"]].copy()
    df["governance"] = df["governance"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    design = pd.get_dummies(df, columns=["governance", "scenario"], drop_first=True, dtype=float)

    cph = CoxPHFitter()
    cph.fit(design, duration_col="duration", event_col="event_observed")
    out = cph.summary.reset_index().rename(columns={"covariate": "term"})
    keep = ["term", "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"]
    out = out[[c for c in keep if c in out.columns]]
    return out.rename(columns={
        "exp(coef)": "hazard_ratio",
        "exp(coef) lower 95%": "hr_ci_lower_95",
        "exp(coef) upper 95%": "hr_ci_upper_95",
    })


# --------------------------------------------------------------- 8.1 (8)


def ols_regressions(runs: pd.DataFrame) -> pd.DataFrame:
    """OLS per KPI: governance (G1 reference) + scenario + n_agents.

    ``n_agents`` is constant at 50 across the confirmatory batch, so it is
    included for spec compliance and reported as dropped rather than
    silently omitted.
    """
    rows = []
    n_agents_varies = runs["n_agents"].nunique() > 1
    formula_tail = " + n_agents" if n_agents_varies else ""

    for kpi in KPI_COLUMNS:
        sub = runs[["governance", "scenario", "n_agents", kpi]].dropna().copy()
        sub["governance"] = sub["governance"].astype(str)
        sub["scenario"] = sub["scenario"].astype(str)
        model = smf.ols(
            f'{kpi} ~ C(governance, Treatment("G1")) + C(scenario, Treatment("A")){formula_tail}',
            data=sub,
        ).fit()
        for term in model.params.index:
            rows.append({
                "kpi": kpi, "term": term,
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "t": float(model.tvalues[term]),
                "p": float(model.pvalues[term]),
                "ci_lower": float(model.conf_int().loc[term, 0]),
                "ci_upper": float(model.conf_int().loc[term, 1]),
                "r_squared": float(model.rsquared),
                "n_agents_in_model": n_agents_varies,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver


def run_analysis(data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Executes the full spec §8.1 battery and writes every artefact."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading confirmatory dataset from {data_dir} ...")
    runs = load_run_index(data_dir)
    print(f"  {len(runs)} replications across {runs.groupby(['governance','scenario'], observed=True).ngroups} cells")

    print("Writing spec Table 7 epoch summaries ...")
    write_epoch_summaries(data_dir)

    results: dict[str, Any] = {"runs": runs, "data_dir": data_dir}

    print("[1/8] Descriptive statistics ...")
    desc = descriptive_statistics(runs)
    desc.to_csv(tables_dir / "descriptive_statistics.csv", index=False)
    results["descriptives"] = desc

    print("[2/8] One-way ANOVA + Tukey HSD (Levene -> Welch where needed) ...")
    anova, tukey = one_way_anova_by_governance(runs)
    anova.to_csv(tables_dir / "anova_oneway_by_governance.csv", index=False)
    tukey.to_csv(tables_dir / "tukey_hsd_pairwise.csv", index=False)
    results["anova_oneway"], results["tukey"] = anova, tukey

    print("[3/8] Two-way ANOVA (governance x volatility) ...")
    two_way = two_way_anova(runs, dv="gdp_equivalent")
    two_way.to_csv(tables_dir / "anova_twoway_gdp.csv", index=False)
    results["anova_twoway"] = two_way

    print("[4/8] Kaplan-Meier + log-rank ...")
    fitters, km_table, logrank = kaplan_meier(runs)
    km_table.to_csv(tables_dir / "survival_summary.csv", index=False)
    pd.DataFrame([logrank]).to_csv(tables_dir / "logrank_test.csv", index=False)
    results["km_fitters"], results["km_table"], results["logrank"] = fitters, km_table, logrank

    print("[5/8] Cox proportional hazards ...")
    cox = cox_hazard_ratios(runs)
    cox.to_csv(tables_dir / "cox_hazard_ratios.csv", index=False)
    results["cox"] = cox

    print("[6/8] OLS regressions ...")
    ols = ols_regressions(runs)
    ols.to_csv(tables_dir / "ols_regression.csv", index=False)
    results["ols"] = ols

    print("[7/8] Figures ...")
    from analysis import plots
    results["figures"] = plots.render_all(runs, fitters, data_dir, figures_dir)

    print("[8/8] Paper-ready PDF ...")
    from analysis import report
    results["pdf"] = report.build_report(results, out_dir)

    _print_console_summary(results)
    return results


def _print_console_summary(results: dict[str, Any]) -> None:
    """Spec §8.2: print key results (ANOVA F/p, effect sizes) on completion."""
    print("\n" + "=" * 68)
    print("CONSOLE SUMMARY — key results")
    print("=" * 68)

    print("\nH1/H2 — one-way ANOVA by governance (alpha = .01, Bonferroni):")
    for _, r in results["anova_oneway"].iterrows():
        flag = "significant" if r["significant"] else "n.s."
        print(f"  {r['kpi']:<20} F={r['F']:>10.2f}  p={r['p']:<10.3g} "
              f"np2={r['partial_eta_sq']:.3f}  f={r['cohens_f']:.2f}  [{flag}]")
        if not r["homogeneity_ok"]:
            print(f"       Levene p={r['levene_p']:.3g} -> {r['test']}")

    print("\nH3 — two-way ANOVA, governance x volatility on GDP-equivalent:")
    for _, r in results["anova_twoway"].iterrows():
        if r["term"] == "Residual":
            continue
        print(f"  {r['term']:<28} F={r['F']:>8.2f}  p={r['p']:<10.3g} np2={r['partial_eta_sq']:.4f}")

    print("\nMTSC — survival (spec Table 4 event definition):")
    for _, r in results["km_table"].iterrows():
        med = "not reached" if pd.isna(r["median_survival"]) else f"{r['median_survival']:.0f}"
        print(f"  {r['governance']:<4} n={r['n']:<4} events={r['events']:<4} "
              f"({r['event_rate_pct']:.1f}%)  median survival={med}")
    lr = results["logrank"]
    print(f"  multivariate log-rank: chi2={lr['test_statistic']:.1f}  p={lr['p']:.3g}")
    print("=" * 68)
    print(f"\nTables  -> {Path(results['pdf']).parent / 'tables'}")
    print(f"Figures -> {Path(results['pdf']).parent / 'figures'}")
    print(f"Report  -> {results['pdf']}")
    print("\nNOTE ON COLLAPSE DEFINITIONS")
    print(SUBSISTENCE_DISCREPANCY_NOTE)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aegora Isle statistical analysis (spec §8)")
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--output_dir", type=str, default="analysis/outputs/")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_analysis(args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

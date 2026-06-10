"""Turn experiment records into the paper's quantitative results.

Four questions, all answered with measured numbers (no fabrication):

  1. **Marginal coverage.** Coverage by method x DGP with experiment-level
     CIs. Does split conformal stay near nominal under stationary-but-non-
     exchangeable DGPs (AR(1), GARCH)? Quantify the gap.
  2. **Conditional coverage.** Coverage stratified by TRUE-vol tercile on the
     GARCH DGP: how badly does the unnormalized score under-cover high-vol
     regimes, how much of the spread does the normalized score close (paired
     per-experiment comparison), and where does CQR land?
  3. **Regime breaks.** Coverage-hole depth and duration after an abrupt
     shift; ACI's repair speed vs gamma and its width cost.
  4. **Width at matched coverage.** Among methods whose marginal coverage is
     within a band around nominal, the average width relative to the ORACLE
     interval -- including the parametric baseline, split by whether its
     Gaussian innovation assumption is right (gaussian) or wrong (student-t).

Everything is stratified; aggregates alone are never reported without the
stratified breakdown. All outputs are JSON-able.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NOMINAL = 0.90
COVERAGE_BAND = (0.885, 0.915)  # 'matched coverage' band for width comparison


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _mean_ci(values: pd.Series) -> dict:
    """Mean with a normal-approx 95% CI over experiments."""
    v = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    m = float(v.mean())
    half = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size) if v.size > 1 else float("nan")
    return {"mean": m, "ci_lo": m - half, "ci_hi": m + half, "n": int(v.size)}


# --------------------------------------------------------------------------- #
# 1. marginal coverage by method x DGP
# --------------------------------------------------------------------------- #
def marginal_coverage(df: pd.DataFrame) -> dict:
    out: dict = {}
    for kind, gk in df.groupby("cfg_kind"):
        out[kind] = {}
        for method, gm in gk.groupby("method"):
            cov = _mean_ci(gm["coverage"])
            out[kind][method] = {
                "coverage": cov["mean"],
                "coverage_ci": [cov["ci_lo"], cov["ci_hi"]],
                "gap_vs_nominal": cov["mean"] - NOMINAL,
                "mean_width_vs_oracle": _mean_ci(gm["width_vs_oracle"])["mean"],
                "frac_unbounded": float(gm["frac_unbounded"].mean()),
                "n_experiments": cov["n"],
            }
    return out


def exchangeability_gap(df: pd.DataFrame) -> dict:
    """Coverage gap of plain split conformal vs nominal, per DGP, with CI.
    The headline 'does the exchangeability violation actually bite marginally?'
    number, also split by innovation family."""
    out: dict = {}
    sub = df[df["method"] == "split_abs"]
    for kind, g in sub.groupby("cfg_kind"):
        c = _mean_ci(g["coverage"])
        entry = {"coverage": c["mean"], "ci": [c["ci_lo"], c["ci_hi"]],
                 "gap": c["mean"] - NOMINAL, "n": c["n"]}
        by_innov = {}
        for innov, gi in g.groupby("cfg_innov"):
            ci = _mean_ci(gi["coverage"])
            by_innov[innov] = {"coverage": ci["mean"], "ci": [ci["ci_lo"], ci["ci_hi"]],
                               "n": ci["n"]}
        entry["by_innovation"] = by_innov
        out[kind] = entry
    return out


# --------------------------------------------------------------------------- #
# 2. conditional coverage by true-vol tercile
# --------------------------------------------------------------------------- #
def conditional_coverage(df: pd.DataFrame, kind: str = "garch") -> dict:
    """Tercile-stratified coverage on a vol-clustered DGP + paired spread
    comparison between the normalized and unnormalized scores."""
    sub = df[df["cfg_kind"] == kind]
    out: dict = {"dgp_kind": kind, "methods": {}}
    for method, g in sub.groupby("method"):
        out["methods"][method] = {
            "cov_vol_low": _mean_ci(g["cov_vol_low"])["mean"],
            "cov_vol_mid": _mean_ci(g["cov_vol_mid"])["mean"],
            "cov_vol_high": _mean_ci(g["cov_vol_high"]),
            "vol_cov_spread": _mean_ci(g["vol_cov_spread"]),
            "width_ratio_vol_low": _mean_ci(g["width_ratio_vol_low"])["mean"],
            "width_ratio_vol_high": _mean_ci(g["width_ratio_vol_high"])["mean"],
        }

    # paired per-experiment spread reduction: split_abs vs split_norm vs cqr
    piv = sub.pivot_table(index="exp_id", columns="method", values="vol_cov_spread")
    out["paired_spread_reduction"] = {}
    for other in ("split_norm", "cqr"):
        if other in piv.columns and "split_abs" in piv.columns:
            d = (piv["split_abs"] - piv[other]).dropna()
            out["paired_spread_reduction"][f"split_abs_minus_{other}"] = _mean_ci(d)
    # high-vol under-coverage closure (paired): cov_high(other) - cov_high(abs)
    piv_hi = sub.pivot_table(index="exp_id", columns="method", values="cov_vol_high")
    out["paired_high_vol_gain"] = {}
    for other in ("split_norm", "cqr", "param_gauss"):
        if other in piv_hi.columns:
            d = (piv_hi[other] - piv_hi["split_abs"]).dropna()
            out["paired_high_vol_gain"][f"{other}_minus_split_abs"] = _mean_ci(d)
    return out


# --------------------------------------------------------------------------- #
# 3. regime breaks: hole depth/duration, ACI repair vs gamma, width cost
# --------------------------------------------------------------------------- #
def break_analysis(df: pd.DataFrame) -> dict:
    sub = df[df["cfg_kind"] == "break"]
    out: dict = {"methods": {}, "aci_gamma_sweep": []}
    for method, g in sub.groupby("method"):
        rec = {
            "cov_pre_break": _mean_ci(g["cov_pre_break"])["mean"],
            "cov_post_break": _mean_ci(g["cov_post_break"])["mean"],
            "hole_depth": _mean_ci(g["hole_depth"]),
            "frac_recovered": float(g["recovered"].mean()),
            "recovery_steps_median": float(g["recovery_steps"].median()),
            "mean_width_vs_oracle": _mean_ci(g["width_vs_oracle"])["mean"],
            "frac_unbounded": float(g["frac_unbounded"].mean()),
        }
        for col in [c for c in g.columns if c.startswith("cov_post_") and c != "cov_post_break"]:
            rec[col] = _mean_ci(g[col])["mean"]
        out["methods"][method] = rec

    aci = sub[sub["family"] == "aci"].copy()
    aci["score_type"] = aci["method"].str.contains("_norm_").map({True: "norm", False: "abs"})
    for (stype, gamma), g in aci.groupby(["score_type", "aci_gamma"]):
        out["aci_gamma_sweep"].append({
            "score_type": stype,
            "gamma": float(gamma),
            "hole_depth": _mean_ci(g["hole_depth"])["mean"],
            "cov_post_0_60": _mean_ci(g["cov_post_0_60"])["mean"],
            "frac_recovered": float(g["recovered"].mean()),
            "recovery_steps_median": float(g["recovery_steps"].median()),
            "mean_width_vs_oracle": _mean_ci(g["width_vs_oracle"])["mean"],
            "frac_unbounded": float(g["frac_unbounded"].mean()),
        })
    out["aci_gamma_sweep"].sort(key=lambda r: (r["score_type"], r["gamma"]))
    return out


# --------------------------------------------------------------------------- #
# 4. width at matched coverage (efficiency, incl. the parametric baseline)
# --------------------------------------------------------------------------- #
def width_at_matched_coverage(df: pd.DataFrame) -> dict:
    """Per DGP kind and innovation family: methods whose mean coverage falls in
    COVERAGE_BAND, ranked by mean width relative to the oracle interval. This
    answers 'is conformal buying anything over the parametric vol model when
    that model is right (gaussian) -- and when it is misspecified (student-t)?'
    """
    out: dict = {"band": list(COVERAGE_BAND)}
    for (kind, innov), g in df.groupby(["cfg_kind", "cfg_innov"]):
        entry = {}
        for method, gm in g.groupby("method"):
            cov = float(gm["coverage"].mean())
            entry[method] = {
                "coverage": cov,
                "in_band": bool(COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1]),
                "width_vs_oracle": _mean_ci(gm["width_vs_oracle"])["mean"],
                "n": int(len(gm)),
            }
        out[f"{kind}/{innov}"] = entry
    return out


# --------------------------------------------------------------------------- #
# top-level summary
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> dict:
    n_exp = df["exp_id"].nunique()
    return {
        "n_experiments": int(n_exp),
        "n_records": int(len(df)),
        "experiments_per_kind": {k: int(g["exp_id"].nunique())
                                 for k, g in df.groupby("cfg_kind")},
        "marginal_coverage": marginal_coverage(df),
        "exchangeability_gap_split_abs": exchangeability_gap(df),
        "conditional_coverage_garch": conditional_coverage(df, "garch"),
        # NOTE: on the break DGP sigma_t takes <=2 values, so "terciles" degenerate
        # to a pre/post-break split and the low tercile is often empty (NaN). Kept for
        # schema stability; the paper never quotes this block.
        "conditional_coverage_break": conditional_coverage(df, "break"),
        "break_analysis": break_analysis(df),
        "width_at_matched_coverage": width_at_matched_coverage(df),
    }

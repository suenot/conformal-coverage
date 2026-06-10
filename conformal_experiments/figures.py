"""Generate the paper's figures (vector PDF) from the saved results.

    python -m conformal_experiments.figures      # writes paper/figures/*.pdf

Reads results/results.json and results/records.csv; the small illustrative
setup panels are recomputed deterministically (fixed seeds, illustrative-only
parameters that feed no quantitative result).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import DGPConfig, simulate_path
from .simulate import Protocol, run_experiment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {
    "oracle": "#777777",
    "split_abs": "#c0392b",
    "split_norm": "#1f3b73",
    "cqr": "#2e8b57",
    "raw_qr": "#b48ead",
    "param_gauss": "#e0a458",
}
LBL = {
    "oracle": "oracle (true quantiles)",
    "split_abs": "split conformal (abs. residual)",
    "split_norm": "split conformal (normalized)",
    "cqr": "CQR",
    "raw_qr": "raw quantile regression",
    "param_gauss": "parametric Gaussian-EWMA",
}
SHORT = {
    "oracle": "oracle",
    "split_abs": "split conf.\n(abs.)",
    "split_norm": "split conf.\n(normalized)",
    "cqr": "CQR",
    "raw_qr": "raw quantile\nregression",
    "param_gauss": "parametric\nGauss-EWMA",
}
NOMINAL = 0.90


# --------------------------------------------------------------------------- #
# Fig 1: setup -- DGPs with known truth + interval traces around a break
# --------------------------------------------------------------------------- #
def fig_setup(path: Path) -> None:
    proto = Protocol(train_window=300, cal_window=150, test_steps=500,
                     refit_every=250)
    t0 = proto.test_start()

    # (a) GARCH path with the true conditional band (illustrative params only)
    cfg_g = DGPConfig(kind="garch", n_steps=proto.total_steps(), sigma=0.01,
                      garch_alpha=0.12, garch_beta=0.83, innov="gaussian",
                      label="fig")
    path_g = simulate_path(cfg_g, np.random.default_rng(np.random.SeedSequence(11)))
    lo_t, hi_t = path_g.true_interval(proto.alpha)

    # (b) break path: oracle band vs split conformal vs ACI around the break
    cfg_b = DGPConfig(kind="break", n_steps=proto.total_steps(), sigma=0.01,
                      break_step=t0 + 250, vol_mult=4.0, innov="gaussian",
                      label="fig")
    rngb = np.random.default_rng(np.random.SeedSequence(13))
    _, _, steps_b = run_experiment(cfg_b, proto, rngb, ewma_lambda=0.94,
                                   return_steps=True)
    path_b = simulate_path(cfg_b, np.random.default_rng(np.random.SeedSequence(13)))
    lo_b, hi_b = path_b.true_interval(proto.alpha)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0))

    ax = axes[0]
    sl = slice(t0, t0 + 500)
    tt = np.arange(500)
    ax.plot(tt, path_g.returns[sl], lw=0.5, color="#444444", alpha=0.8,
            label="returns $r_t$")
    ax.fill_between(tt, lo_t[sl], hi_t[sl], color="#1f3b73", alpha=0.18,
                    label="true conditional 90% band")
    ax.set_title("(a) GARCH(1,1) DGP: the truth is known")
    ax.set_xlabel("test step")
    ax.set_ylabel("return")
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    rel0, rel1 = 150, 450   # window around the break (break at test step 250)
    tt = np.arange(rel0, rel1) - 250
    sl = slice(t0 + rel0, t0 + rel1)
    isl = slice(rel0, rel1)
    ax.plot(tt, path_b.returns[sl], lw=0.5, color="#444444", alpha=0.7)
    ax.fill_between(tt, lo_b[sl], hi_b[sl], color="#777777", alpha=0.25,
                    label="oracle 90% band")
    ax.plot(tt, steps_b["split_abs"][0][isl], color=C["split_abs"], lw=1.0,
            label=LBL["split_abs"])
    ax.plot(tt, steps_b["split_abs"][1][isl], color=C["split_abs"], lw=1.0)
    ax.plot(tt, steps_b["aci_abs_g0.05"][0][isl], color="#2e8b57", lw=1.0,
            ls="--", label=r"ACI on abs. score ($\gamma=0.05$)")
    ax.plot(tt, steps_b["aci_abs_g0.05"][1][isl], color="#2e8b57", lw=1.0, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title(r"(b) abrupt vol break ($\times 4$): the coverage hole")
    ax.set_xlabel("steps since break")
    ax.set_ylabel("return")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: marginal coverage by method x DGP
# --------------------------------------------------------------------------- #
def fig_marginal(path: Path, df: pd.DataFrame) -> None:
    methods = ["oracle", "split_abs", "split_norm", "cqr", "raw_qr", "param_gauss"]
    kinds = ["iid", "ar1", "garch", "break"]
    kind_colors = {"iid": "#9aa6b2", "ar1": "#e0a458", "garch": "#1f3b73",
                   "break": "#c0392b"}
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    xbase = np.arange(len(methods), dtype=float)
    for ki, kind in enumerate(kinds):
        xs, ys, los, his = [], [], [], []
        for mi, m in enumerate(methods):
            g = df[(df["cfg_kind"] == kind) & (df["method"] == m)]["coverage"]
            if g.empty:
                continue
            mean = g.mean()
            half = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
            xs.append(xbase[mi] + (ki - 1.5) * 0.16)
            ys.append(mean)
            los.append(mean - half)
            his.append(mean + half)
        ys, los, his = np.array(ys), np.array(los), np.array(his)
        ax.errorbar(xs, ys, yerr=[ys - los, his - ys], fmt="o", ms=3.5,
                    color=kind_colors[kind], lw=1.0, capsize=2, label=kind)
    ax.axhline(NOMINAL, color="k", lw=1.0, ls="--", label="nominal 0.90")
    ax.set_xticks(xbase, [SHORT[m] for m in methods], fontsize=7.5)
    ax.set_ylabel("marginal coverage")
    ax.set_title("marginal coverage by method and DGP (95% CIs over experiments)")
    ax.legend(fontsize=7, ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3 (headline): conditional coverage by true-vol tercile on GARCH
# --------------------------------------------------------------------------- #
def fig_conditional(path: Path, df: pd.DataFrame) -> None:
    methods = ["oracle", "split_abs", "split_norm", "cqr", "raw_qr", "param_gauss"]
    sub = df[df["cfg_kind"] == "garch"]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(methods), dtype=float)
    w = 0.26
    terc_cols = {"cov_vol_low": ("#9ecae1", "low-vol tercile"),
                 "cov_vol_mid": ("#4292c6", "mid-vol tercile"),
                 "cov_vol_high": ("#084594", "high-vol tercile")}
    for j, (col, (color, lbl)) in enumerate(terc_cols.items()):
        means, halfs = [], []
        for m in methods:
            g = sub[sub["method"] == m][col].dropna()
            means.append(g.mean())
            halfs.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
        ax.bar(x + (j - 1) * w, means, w, color=color, label=lbl,
               yerr=halfs, error_kw={"lw": 0.8}, capsize=2)
    ax.axhline(NOMINAL, color="k", lw=1.0, ls="--", label="nominal 0.90")
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(x, [SHORT[m] for m in methods], fontsize=7.5)
    ax.set_ylabel("coverage within tercile")
    ax.set_title("conditional coverage by TRUE-vol tercile, GARCH DGP")
    ax.legend(fontsize=7, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: post-break coverage trajectories + ACI gamma sweep
# --------------------------------------------------------------------------- #
def _smooth(vals: list, w: int = 60) -> np.ndarray:
    s = pd.Series([np.nan if v is None else v for v in vals], dtype=float)
    return s.rolling(w, min_periods=w // 2).mean().to_numpy()


def fig_break(path: Path, results: dict, df: pd.DataFrame) -> None:
    traj = results["break_trajectory"]
    rel = np.array(traj["rel_time"], dtype=float)
    cov = traj["coverage"]
    gammas = results["meta"]["protocol"]["aci_gammas"]

    def width_cost(method: str) -> float:
        g = df[(df["cfg_kind"] == "break") & (df["method"] == method)]
        return float(g["width_vs_oracle"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)

    ax = axes[0]
    for m in ["split_abs", "split_norm", "cqr", "param_gauss"]:
        ax.plot(rel, _smooth(cov[m]), lw=1.3, color=C[m],
                label=f"{LBL[m]} (w/oracle {width_cost(m):.2f})")
    ax.axhline(NOMINAL, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title("(a) coverage around the break (rolling 60 steps)")
    ax.set_xlabel("steps since break")
    ax.set_ylabel("coverage")
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    cmap = plt.cm.viridis
    ax.plot(rel, _smooth(cov["split_abs"]), lw=1.3, color=C["split_abs"],
            label=f"no ACI (w/oracle {width_cost('split_abs'):.2f})")
    for gi, g in enumerate(gammas):
        m = f"aci_abs_g{g:g}"
        ax.plot(rel, _smooth(cov[m]), lw=1.2, color=cmap(0.15 + 0.7 * gi / max(len(gammas) - 1, 1)),
                label=rf"ACI $\gamma={g:g}$ (w/oracle {width_cost(m):.2f})")
    ax.axhline(NOMINAL, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title("(b) ACI repair speed vs $\\gamma$ (abs. score)")
    ax.set_xlabel("steps since break")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "records.csv")
    results = json.loads((RESULTS / "results.json").read_text())

    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_marginal(FIGDIR / "fig_marginal_coverage.pdf", df)
    fig_conditional(FIGDIR / "fig_conditional_coverage.pdf", df)
    fig_break(FIGDIR / "fig_break_aci.pdf", results, df)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()

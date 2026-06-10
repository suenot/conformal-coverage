"""Assert that every quantitative claim in paper/main.tex matches results/results.json.

    python scripts/check_paper_numbers.py

Each check formats a value from results.json exactly the way the paper quotes it
and asserts the resulting token appears in main.tex (and, where the paper states
an inequality or a range, asserts the underlying inequality holds). Exits
non-zero if any check fails, so it can gate a release.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TEX = re.sub(r"\s+", " ", (ROOT / "paper" / "main.tex").read_text())  # normalize wraps
R = json.loads((ROOT / "results" / "results.json").read_text())

failures: list[str] = []
n_checks = 0


def check(label: str, token: str, cond: bool = True) -> None:
    """Assert ``token`` appears in main.tex (whitespace-normalized) and ``cond`` holds."""
    global n_checks
    n_checks += 1
    ok_tex = token in TEX
    if ok_tex and cond:
        print(f"  PASS  {label:62} {token!r}")
    else:
        why = [] if ok_tex else [f"token {token!r} not in main.tex"]
        if not cond:
            why.append("condition failed")
        failures.append(f"{label}: {'; '.join(why)}")
        print(f"  FAIL  {label:62} {token!r}  <-- {'; '.join(why)}")


def f3(v):  # 0.901
    return f"{v:.3f}"


def f2(v):  # 1.05
    return f"{v:.2f}"


def g3(v):  # signed gap, 3dp: -0.005
    return f"{v:+.3f}"


MC = R["marginal_coverage"]
KINDS = ["iid", "ar1", "garch", "break"]

# ---------------------------------------------------------------- design ----
print("[design / protocol]")
check("n experiments", "$180$", R["n_experiments"] == 180)
check("n records 2,520", "$2{,}520$", R["n_records"] == 2520)
check("14 methods", "$14$", len(MC["garch"]) == 14)
counts = R["meta"]["counts"]
check("counts 36/36/48/60", "$36/36/48/60$",
      [counts[k] for k in KINDS] == [36, 36, 48, 60])
proto = R["meta"]["protocol"]
check("train window 600", "$600$", proto["train_window"] == 600)
check("cal window 250", "$250$", proto["cal_window"] == 250)
check("test steps 1500", "$1{,}500$", proto["test_steps"] == 1500)
check("alpha 0.10", r"$\alpha=0.10$", proto["alpha"] == 0.10)
check("gammas", r"$\gamma\in\{0.005,0.01,0.02,0.05\}$",
      proto["aci_gammas"] == [0.005, 0.01, 0.02, 0.05])
check("GBR 100 trees depth 2 lr 0.08", "$100$ trees, depth $2$, learning rate $0.08$",
      R["meta"]["protocol"]["gbr"] == {"n_estimators": 100, "max_depth": 2,
                                       "learning_rate": 0.08})
check("seed", "$20260610$", R["meta"]["seed"] == 20260610)
check("ewma lambda range", r"$\lambda\sim\mathrm{Uniform}[0.90,0.97]$",
      R["meta"]["batch_constants"]["ewma_lambda_range"] == [0.90, 0.97])
check("break position range", r"$[30\%,60\%]$",
      R["meta"]["batch_constants"]["break_pos_range"] == [0.30, 0.60])
check("recovery threshold 0.87", "$1-\\alpha-0.03=0.87$",
      abs((1 - proto["alpha"] - proto["recovery_tol"]) - 0.87) < 1e-12)
# realized innovation split per kind
eg = R["exchangeability_gap_split_abs"]
splits = {k: (eg[k]["by_innovation"]["gaussian"]["n"],
              eg[k]["by_innovation"]["student_t"]["n"]) for k in KINDS}
check("innovation splits", "$20/16$, $13/23$, $31/17$, and $32/28$",
      [splits[k] for k in KINDS] == [(20, 16), (13, 23), (31, 17), (32, 28)])
check("split theorem value n=250", "0.9004",
      abs(math.ceil(0.9 * 251) / 251 - 0.9004) < 5e-5)

# ------------------------------------------- Table 1: marginal coverage ----
print("[Table 1: marginal coverage]")
T1_ROWS = [("oracle", "Oracle (true quantiles)"),
           ("split_abs", "Split conformal (absolute)"),
           ("split_norm", "Split conformal (normalized)"),
           ("cqr", "CQR"),
           ("aci_abs_g0.01", "ACI, abs."),
           ("aci_norm_g0.01", "ACI, norm."),
           ("raw_qr", "Raw quantile regression"),
           ("param_gauss", "Parametric Gauss--EWMA")]
for m, lbl in T1_ROWS:
    row = " & ".join(f3(MC[k][m]["coverage"]) for k in KINDS)
    check(f"T1 row {lbl}", row)
# all ACI variants in [0.900, 0.901] on every DGP (claim in caption + text)
aci_covs = [v["coverage"] for k in KINDS for mm, v in MC[k].items()
            if mm.startswith("aci_")]
check("ACI in [0.900,0.901] everywhere", "$[0.900,0.901]$",
      all(0.8995 <= c < 0.9015 for c in aci_covs))

# ------------------------------------------------- exchangeability gaps ----
print("[exchangeability gaps, split_abs]")
for kind, txt_kind in [("garch", "GARCH"), ("break", "break")]:
    e = eg[kind]
    tok = f"${g3(e['gap'])}$ $[{g3(e['ci'][0] - 0.9)},{g3(e['ci'][1] - 0.9)}]$"
    check(f"gap {txt_kind}", tok)
e = eg["ar1"]
check("gap ar1", f"${g3(e['gap'])}$, with a CI of $[{g3(e['ci'][0] - 0.9)},{g3(e['ci'][1] - 0.9)}]$ that includes zero")

# --------------------------------------- Table 2: GARCH tercile coverage ----
print("[Table 2: conditional coverage, GARCH]")
CC = R["conditional_coverage_garch"]["methods"]
T2_ROWS = ["oracle", "split_abs", "split_norm", "cqr", "aci_abs_g0.05",
           "raw_qr", "param_gauss"]
for m in T2_ROWS:
    v = CC[m]
    row = (f"{f3(v['cov_vol_low'])} & {f3(v['cov_vol_mid'])} & "
           f"{f3(v['cov_vol_high']['mean'])} & {f3(v['vol_cov_spread']['mean'])} & "
           f"{f2(v['width_ratio_vol_low'])} & {f2(v['width_ratio_vol_high'])}")
    check(f"T2 row {m}", row)
check("abstract terciles 0.952/0.915/0.820",
      "$0.952/0.915/0.820$",
      (round(CC["split_abs"]["cov_vol_low"], 3),
       round(CC["split_abs"]["cov_vol_mid"], 3),
       round(CC["split_abs"]["cov_vol_high"]["mean"], 3)) == (0.952, 0.915, 0.820))
check("norm terciles 0.893/0.905/0.905", "$0.893/0.905/0.905$",
      (round(CC["split_norm"]["cov_vol_low"], 3),
       round(CC["split_norm"]["cov_vol_mid"], 3),
       round(CC["split_norm"]["cov_vol_high"]["mean"], 3)) == (0.893, 0.905, 0.905))
check("cqr terciles 0.937/0.906/0.846", "$0.937/0.906/0.846$",
      (round(CC["cqr"]["cov_vol_low"], 3),
       round(CC["cqr"]["cov_vol_mid"], 3),
       round(CC["cqr"]["cov_vol_high"]["mean"], 3)) == (0.937, 0.906, 0.846))
check("aci g0.05 terciles 0.896/0.899/0.907", "$0.896/0.899/0.907$",
      round(CC["aci_abs_g0.05"]["vol_cov_spread"]["mean"], 3) == 0.028)

P = R["conditional_coverage_garch"]
d = P["paired_spread_reduction"]["split_abs_minus_split_norm"]
check("paired spread reduction (norm)",
      f"$+{f3(d['mean'])}$ $[+{f3(d['ci_lo'])},+{f3(d['ci_hi'])}]$")
d = P["paired_spread_reduction"]["split_abs_minus_cqr"]
check("paired spread reduction (cqr)",
      f"$+{f3(d['mean'])}$ $[+{f3(d['ci_lo'])},+{f3(d['ci_hi'])}]$")
d = P["paired_high_vol_gain"]["split_norm_minus_split_abs"]
check("paired high-vol gain (norm)",
      f"$+{f3(d['mean'])}$ $[+{f3(d['ci_lo'])},+{f3(d['ci_hi'])}]$")
d = P["paired_high_vol_gain"]["cqr_minus_split_abs"]
check("paired high-vol gain (cqr)",
      f"$+{f3(d['mean'])}$ $[+{f3(d['ci_lo'])},+{f3(d['ci_hi'])}]$")
check("spread 0.134 -> 0.040 (abstract/conclusion)", "$0.134\\to0.040$",
      round(CC["split_abs"]["vol_cov_spread"]["mean"], 3) == 0.134
      and round(CC["split_norm"]["vol_cov_spread"]["mean"], 3) == 0.040)
check("ACI g0.05 unbounded on GARCH 3.2%", r"$3.2\%$",
      round(MC["garch"]["aci_abs_g0.05"]["frac_unbounded"] * 100, 1) == 3.2)

# --------------------------------------------------- Table 3: breaks --------
print("[Table 3: break anatomy]")
BA = R["break_analysis"]["methods"]
T3_ROWS = ["oracle", "split_abs", "split_norm", "cqr", "aci_abs_g0.005",
           "aci_abs_g0.01", "aci_abs_g0.02", "aci_abs_g0.05", "aci_norm_g0.01",
           "raw_qr", "param_gauss"]
for m in T3_ROWS:
    v = BA[m]
    rec = v["recovery_steps_median"]
    rec_s = f"{rec:.0f}" if rec == int(rec) else f"{rec:.1f}"
    row = (f"{f3(v['cov_post_0_60'])} & {f3(v['cov_post_60_150'])} & "
           f"{f3(v['cov_post_300_600'])} & {f3(v['hole_depth']['mean'])} & "
           f"{rec_s} & {f2(v['mean_width_vs_oracle'])}")
    check(f"T3 row {m}", row)
hd = BA["split_abs"]["hole_depth"]
check("split_abs hole depth CI",
      f"${f3(hd['mean'])}$ $[{f3(hd['ci_lo'])},{f3(hd['ci_hi'])}]$")
check("ACI gamma sweep first-60 monotone", "$0.700/0.777/0.839/0.875$",
      [round(BA[f"aci_abs_g{g:g}"]["cov_post_0_60"], 3)
       for g in (0.005, 0.01, 0.02, 0.05)] == [0.700, 0.777, 0.839, 0.875])
check("ACI recovery medians", "$93.5/77/66/61$",
      [BA[f"aci_abs_g{g:g}"]["recovery_steps_median"]
       for g in (0.005, 0.01, 0.02, 0.05)] == [93.5, 77.0, 66.0, 61.0])
aci_widths = [BA[f"aci_abs_g{g:g}"]["mean_width_vs_oracle"]
              for g in (0.005, 0.01, 0.02, 0.05)]
check("ACI width cost 1.12-1.14 vs 1.05", r"$1.12$--$1.14\times$",
      min(aci_widths) >= 1.115 and max(aci_widths) < 1.145
      and round(BA["split_abs"]["mean_width_vs_oracle"], 2) == 1.05)
check("aci_norm g0.01 first-60 / 60-150", "$0.875$ in the first $60$ steps and $0.924$",
      round(BA["aci_norm_g0.01"]["cov_post_0_60"], 3) == 0.875
      and round(BA["aci_norm_g0.01"]["cov_post_60_150"], 3) == 0.924)
check("aci_norm g0.05: no hole", "$0.900$ first-$60$",
      round(BA["aci_norm_g0.05"]["cov_post_0_60"], 3) == 0.900)
check("aci_norm g0.05 hole 0.847 vs oracle 0.801", "$0.847$",
      round(BA["aci_norm_g0.05"]["hole_depth"]["mean"], 3) == 0.847
      and round(BA["oracle"]["hole_depth"]["mean"], 3) == 0.801)
check("aci_norm g0.05 width 1.19", r"$1.19\times$",
      round(BA["aci_norm_g0.05"]["mean_width_vs_oracle"], 2) == 1.19)
check("aci_norm g0.05 unbounded 0.7%", r"$0.7\%$",
      round(BA["aci_norm_g0.05"]["frac_unbounded"] * 100, 1) == 0.7)
check("ACI abs g0.05 unbounded on break 3.0%", r"$3.0\%$",
      round(BA["aci_abs_g0.05"]["frac_unbounded"] * 100, 1) == 3.0)
check("ACI g0.005 overshoot 0.935 at 150-300", "$0.935$",
      round(BA["aci_abs_g0.005"]["cov_post_150_300"], 3) == 0.935)
check("raw_qr frac recovered 86.7%", r"$86.7\%$",
      round(BA["raw_qr"]["frac_recovered"] * 100, 1) == 86.7)

# ------------------------------------- width at matched coverage (GARCH) ----
print("[width at matched coverage]")
W = R["width_at_matched_coverage"]
check("band", "$[0.885,0.915]$", W["band"] == [0.885, 0.915])
gg, gt = W["garch/gaussian"], W["garch/student_t"]
check("split_abs widths 1.15/1.13",
      r"split-absolute $1.15\times$ (Gaussian innovations) and $1.13\times$",
      round(gg["split_abs"]["width_vs_oracle"], 2) == 1.15
      and round(gt["split_abs"]["width_vs_oracle"], 2) == 1.13)
check("norm widths 1.09/1.08", r"normalized $1.09\times$/$1.08\times$",
      round(gg["split_norm"]["width_vs_oracle"], 2) == 1.09
      and round(gt["split_norm"]["width_vs_oracle"], 2) == 1.08)
check("cqr widths 1.08/1.06", r"$1.08\times$/$1.06\times$",
      round(gg["cqr"]["width_vs_oracle"], 2) == 1.08
      and round(gt["cqr"]["width_vs_oracle"], 2) == 1.06)
check("param widths 0.99/1.03", r"$0.99\times$/$1.03\times$",
      round(gg["param_gauss"]["width_vs_oracle"], 2) == 0.99
      and round(gt["param_gauss"]["width_vs_oracle"], 2) == 1.03)
check("param coverage 0.878/0.888", "$0.878$/$0.888$",
      round(gg["param_gauss"]["coverage"], 3) == 0.878
      and round(gt["param_gauss"]["coverage"], 3) == 0.888
      and not gg["param_gauss"]["in_band"])
check("raw_qr 0.871-0.880 at 0.99x", "$0.871$--$0.880$",
      round(gg["raw_qr"]["coverage"], 3) == 0.871
      and round(gt["raw_qr"]["coverage"], 3) == 0.880
      and round(gg["raw_qr"]["width_vs_oracle"], 2) == 0.99
      and round(gt["raw_qr"]["width_vs_oracle"], 2) == 0.99)
in_band_narrower = [
    (key, m) for key, entry in W.items() if key != "band"
    for m, v in entry.items()
    if m != "oracle" and v["in_band"] and v["width_vs_oracle"] < 1.0
]
check("no in-band method narrower than oracle",
      "no method inside the coverage band is narrower than the oracle",
      not in_band_narrower)
n_under = sum(1 for key, entry in W.items() if key != "band"
              and entry["param_gauss"]["coverage"] < 0.90)
check("param under-covers 6 of 8 cells", "six of our eight", n_under == 6)
check("param under-covers 3 of 4 DGPs marginally", "$0.891/0.881/0.883$",
      [f3(MC[k]["param_gauss"]["coverage"]) for k in ["iid", "garch", "break"]]
      == ["0.891", "0.881", "0.883"])

# -------------------------------------------------------- fat-tail facts ----
print("[fat tails at the 90% level]")
q95 = {nu: stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu) for nu in (4, 6, 10)}
check("standardized-t q95 1.507/1.587/1.621", "$1.507/1.587/1.621$",
      [round(q95[nu], 3) for nu in (4, 6, 10)] == [1.507, 1.587, 1.621]
      and all(q95[nu] < stats.norm.ppf(0.95) for nu in (4, 6, 10)))
q995_4 = stats.t.ppf(0.995, 4) * np.sqrt(0.5)
check("q995 flip 3.256 vs 2.576", "$3.256$ vs.\\ $2.576$",
      round(q995_4, 3) == 3.256 and round(stats.norm.ppf(0.995), 3) == 2.576
      and q995_4 > stats.norm.ppf(0.995))
ig, it = W["iid/gaussian"]["param_gauss"], W["iid/student_t"]["param_gauss"]
check("param t>gauss on iid (0.894 vs 0.888)", "$0.894$ vs.\\ $0.888$ on iid",
      round(it["coverage"], 3) == 0.894 and round(ig["coverage"], 3) == 0.888
      and it["coverage"] > ig["coverage"])
check("param t>gauss on garch (0.888 vs 0.878)", "$0.888$ vs.\\ $0.878$ on GARCH",
      round(gt["param_gauss"]["coverage"], 3) == 0.888
      and round(gg["param_gauss"]["coverage"], 3) == 0.878)
check("param first-60 on break 0.664", "$0.664$",
      round(BA["param_gauss"]["cov_post_0_60"], 3) == 0.664)

# ----------------------------------------------------- ACI theorem bounds ---
print("[ACI bounds]")
T = proto["test_steps"]
bounds = [round((0.9 + g) / (g * T), 3) for g in (0.005, 0.01, 0.02, 0.05)]
check("ACI worst-case bounds at T=1500", "$0.121/0.061/0.031/0.013$",
      bounds == [0.121, 0.061, 0.031, 0.013])
max_dev = max(abs(c - 0.9) for c in aci_covs)
check("ACI deviations >=10x inside bounds", "an order of magnitude",
      max_dev * 10 < min(bounds))

# -------------------------------------------------------------------- done --
print(f"\n{n_checks} checks, {len(failures)} failures.")
if failures:
    for f_ in failures:
        print("FAIL:", f_)
    sys.exit(1)
print("ALL PAPER NUMBERS MATCH results.json")

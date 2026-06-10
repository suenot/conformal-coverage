"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + records.csv
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below. No wall-clock / randomness leaks
into results (timing is reported to stdout only). Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformal_experiments import __version__
from conformal_experiments import analysis as A
from conformal_experiments.simulate import (
    BREAK_POS_RANGE,
    EWMA_LAMBDA_RANGE,
    POST_WINDOWS,
    TRAJ_POST,
    TRAJ_PRE,
    Protocol,
    run_batch,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED = 20260610
FULL_COUNTS = {"iid": 36, "ar1": 36, "garch": 48, "break": 60}
QUICK_COUNTS = {"iid": 5, "ar1": 5, "garch": 6, "break": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        counts = QUICK_COUNTS
        proto = Protocol(train_window=400, cal_window=150, test_steps=600,
                         refit_every=200)
    else:
        counts = FULL_COUNTS
        proto = Protocol()
    RESULTS.mkdir(exist_ok=True)

    t_start = time.time()
    total = sum(counts.values())
    print(f"[1/3] online batch: {counts} (total {total}, "
          f"T_test={proto.test_steps}, cal={proto.cal_window}, "
          f"train={proto.train_window}, refit_every={proto.refit_every}) ...",
          flush=True)
    records, traj = run_batch(counts, proto, seed=SEED,
                              progress_every=max(1, total // 8))
    df = A.to_frame(records)
    df.to_csv(RESULTS / "records.csv", index=False)
    t_batch = time.time() - t_start

    print("[2/3] summaries ...", flush=True)
    summary = A.summarize(df)

    print("[3/3] writing results ...", flush=True)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "seed": SEED,
            "quick": bool(args.quick),
            "counts": counts,
            "protocol": {**asdict(proto), "gbr": asdict(proto.gbr)},
            "batch_constants": {
                "ewma_lambda_range": list(EWMA_LAMBDA_RANGE),
                "break_pos_range": list(BREAK_POS_RANGE),
                "traj_window": [TRAJ_PRE, TRAJ_POST],
                "post_windows": [list(w) for w in POST_WINDOWS],
            },
            "runtime_seconds_batch": round(t_batch, 1),
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        **summary,
        "break_trajectory": traj,
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    t_total = time.time() - t_start
    print(f"\nWrote {RESULTS / 'results.json'} and records.csv "
          f"({t_total:.0f}s total, {t_batch:.0f}s batch).")

    # ---- headline numbers to stdout --------------------------------------
    mc = summary["marginal_coverage"]
    kinds = ["iid", "ar1", "garch", "break"]
    methods = ["oracle", "split_abs", "split_norm", "cqr", "raw_qr", "param_gauss",
               "aci_abs_g0.01", "aci_norm_g0.01"]
    print("\n--- MARGINAL COVERAGE (nominal 0.90), method x DGP ---")
    print(f"{'method':16}" + "".join(f"{k:>9}" for k in kinds))
    for m in methods:
        row = "".join(
            f"{mc[k][m]['coverage']:9.3f}" if m in mc.get(k, {}) else f"{'--':>9}"
            for k in kinds)
        print(f"{m:16}" + row)

    gap = summary["exchangeability_gap_split_abs"]
    print("\nsplit_abs gap vs nominal (95% CI over experiments):")
    for k in kinds:
        g = gap[k]
        print(f"  {k:6} coverage {g['coverage']:.3f} "
              f"[{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  gap {g['gap']:+.3f}")

    cc = summary["conditional_coverage_garch"]["methods"]
    print("\n--- CONDITIONAL COVERAGE by true-vol tercile (GARCH) ---")
    print(f"{'method':16}{'low':>8}{'mid':>8}{'high':>8}{'spread':>9}")
    for m in ["oracle", "split_abs", "split_norm", "cqr", "raw_qr", "param_gauss"]:
        c = cc[m]
        print(f"{m:16}{c['cov_vol_low']:8.3f}{c['cov_vol_mid']:8.3f}"
              f"{c['cov_vol_high']['mean']:8.3f}{c['vol_cov_spread']['mean']:9.3f}")
    pr = summary["conditional_coverage_garch"]["paired_spread_reduction"]
    for k, v in pr.items():
        print(f"  paired spread reduction {k}: {v['mean']:+.3f} "
              f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]")

    ba = summary["break_analysis"]
    print("\n--- REGIME BREAKS: post-break coverage ---")
    print(f"{'method':16}{'pre':>7}{'0-60':>7}{'60-150':>8}{'300-600':>9}"
          f"{'hole':>7}{'recov%':>8}{'w/oracle':>9}")
    for m in ["split_abs", "split_norm", "cqr", "param_gauss",
              "aci_abs_g0.005", "aci_abs_g0.01", "aci_abs_g0.02", "aci_abs_g0.05",
              "aci_norm_g0.01"]:
        b = ba["methods"][m]
        print(f"{m:16}{b['cov_pre_break']:7.3f}{b['cov_post_0_60']:7.3f}"
              f"{b['cov_post_60_150']:8.3f}{b['cov_post_300_600']:9.3f}"
              f"{b['hole_depth']['mean']:7.3f}{100 * b['frac_recovered']:8.0f}"
              f"{b['mean_width_vs_oracle']:9.2f}")

    wm = summary["width_at_matched_coverage"]
    print("\n--- WIDTH vs ORACLE at matched coverage (GARCH) ---")
    for key in ["garch/gaussian", "garch/student_t"]:
        print(f"  {key}:")
        entry = wm[key]
        for m in ["oracle", "split_abs", "split_norm", "cqr", "param_gauss", "raw_qr"]:
            e = entry[m]
            flag = " " if e["in_band"] else "*"
            print(f"    {m:12} cov {e['coverage']:.3f}{flag} width/oracle "
                  f"{e['width_vs_oracle']:.2f}")
    print("  (* = marginal coverage outside the band "
          f"{wm['band']}; width not coverage-matched)")


if __name__ == "__main__":
    main()

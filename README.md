# Conformal Prediction for Financial Returns: Where Coverage Survives and Where It Breaks

A reproducible experiment harness behind a short methods paper that stress-tests
conformal prediction intervals for return forecasting on controlled DGPs whose
**true conditional quantiles are known exactly** -- so marginal coverage,
regime-stratified conditional coverage, and post-break behavior are all
measured against ground truth rather than asserted.

Methods at nominal 90%: split conformal (absolute residual), **normalized**
split conformal (residual / EWMA vol), CQR (Romano-Candes), **Adaptive
Conformal Inference** (Gibbs-Candes, gamma swept) on both scores, plus
non-conformal baselines (Gaussian interval from a QML-fitted EWMA vol model,
raw unconformalized quantile regression) and the oracle.

DGPs (all parameters sampled and recorded; no hidden constants): iid,
AR(1)-mean, GARCH(1,1) vol clustering, and abrupt regime breaks
(vol x {2,4} and/or mean shift at a sampled time), with Gaussian or
standardized Student-t innovations.

This is a de-commercialized, experimentally-validated companion to a
[marketmaker.cc](https://marketmaker.cc) blog draft on conformal prediction
for position sizing.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py              # full run (~6-8 min) -> results/results.json + records.csv
python -m conformal_experiments.figures  # -> paper/figures/*.pdf
```

Deterministic given the seed in `scripts/run_all.py`. `--quick` runs a small
smoke batch (~1 min).

## Layout

```
conformal_experiments/
  model.py      # DGPs with known conditional quantiles + causal features + EWMA vol
  methods.py    # conformal quantile, ACI update, GBR point/quantile learners, scores
  simulate.py   # online protocol: rolling fit/calibrate, one-step predict, record
  analysis.py   # marginal/conditional coverage, break holes, width-at-matched-coverage
  figures.py    # the paper's 4 vector-PDF figures
scripts/run_all.py
tests/          # pytest checks of the DGP truths and the conformal theorems
results/        # results.json + records.csv (generated)
paper/figures/  # generated figures
```

## Tests

```bash
python -m pytest -q     # 17 sanity/theorem tests (run from the project root)
```

## License

Code: [MIT](LICENSE). Paper text and figures: CC BY 4.0.

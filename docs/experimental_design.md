# Current synthetic benchmark protocol

The executable specification is `analysis/run_benchmark.py`. Run
`python -m analysis.run_benchmark --seed 42` for 200 five-year scenarios or add
`--smoke-test` for 20. Operators are sampled with replacement using NumPy's seeded
generator. Base, downside and distressed regimes have probabilities 0.5, 0.3 and
0.2. Growth, margin, sweep and lease flows are drawn and clipped in `_draw_scenario`;
regime adjustments modify growth, margins, debt, leases and cash.

`data/synthetic/operators.csv` supplies operator parameters. The separate
`scenario_parameters.csv` is checksummed metadata; it does not control the current
sampler. No exit-multiple distribution, rate sampling, Sobol experiment, IRR hurdle
or historical transaction success rate is part of this benchmark.

A simulated failure is any nonpositive EBITDA, payment default, reserve funding
deficit, net leverage above 6.0, or interest coverage below 1.8 in any year.
`failure_type` prioritizes negative EBITDA, then payment default, then combined
reserve/covenant failure, then either alone. Separate payment-default, insolvency
and covenant-breach booleans preserve overlapping conditions. `failed_scenario_count`
counts execution exceptions, not financially distressed scenarios. Exceptions and
single-class samples abort reporting rather than being silently dropped.

The analytic score is `1 / (1 + exp(2 * minimum_headroom))`, a ranking transform,
not a calibrated probability. Classification uses 0.5. AUC receives a percentile
95% interval from 500 scenario bootstrap attempts (single-class resamples skipped).
Scenarios share synthetic operator assumptions; this interval does not establish
out-of-sample performance on real borrowers.

The JSON report contains AUC, false-positive/negative rates, leverage/coverage
MAE and RMSE averaged across scenarios, minimum-headroom errors, scenario records,
failure counts, diagnostic envelopes, data checksums and git SHA. No calibration
curve, Brier score, Sobol indices or equity-return results are emitted. Diagnostic
envelopes are assumption-based heuristics, not proven bounds on all simulations.

Timing warms both paths and reports median and IQR over five repetitions of the
same scenarios. Speed ratios can be below one and vary by machine and load.
Deterministic financial results are reproducible for the same code, inputs, seed
and dependency versions; timings are not. The report SHA names HEAD and does not
capture uncommitted edits: retain the patch and environment alongside local runs.
Both modes overwrite `output/benchmark/benchmark_report.json`; archive it before
running another mode. Older manuscripts and bundled archives are historical
snapshots, not current benchmark output.

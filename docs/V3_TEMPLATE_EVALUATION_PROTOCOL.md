# Frozen template evaluation protocol

Specified before computing held-out performance. Input: the unchanged reviewed
`results/paper_v2/scenario_records.csv`, read with round-trip float precision and
string scenario IDs. The five `operator_id` values define groups. Inspection finds
38/5, 32/3, 47/1, 49/11 and 34/0 scenarios/failures for templates 001–005,
respectively; all 200 scenario IDs are unique. No new simulations are needed.

For each group, select a threshold using only the other four groups. Maximize
training balanced accuracy, defined as (sensitivity + specificity)/2. Candidates
are sorted unique training scores, 0, and `nextafter(1, +infinity)` (scores must
be finite in [0,1]). These boundaries permit all-positive and all-negative
predictions. Predictions use score >= threshold. Among candidates within absolute
1e-12 of the maximum, choose the lowest threshold. A single-class training set
is an error: the primary objective is then undefined. No alternative objective,
score transformation or post-result tuning is permitted.

Concatenate held-out rows exactly once; recompute selected-threshold and fixed-0.5
classification metrics on those same rows. ROC-AUC and PR-AUC (defined here as
average precision, not trapezoidal PR area) use the original continuous score.
For a single-class test fold, both ranking metrics and balanced accuracy are
undefined. Each missing-denominator rate is undefined, including precision when
there are no positive predictions; F1 uses 2TP/(2TP+FP+FN). Undefined values are
JSON null / empty CSV cells. Single-class folds remain in aggregate evaluation.

Primary uncertainty reporting is finite fold min/median/max with the number of
defined folds, plus aggregate point estimates. Five synthetic templates, sparse
positives and overlapping training sets do not support a persuasive precise
confidence interval. No bootstrap or significance claim is planned. Dispersion
is descriptive, not a confidence interval. No secondary threshold objective is
planned.

The fixed score is never trained. Thus pooled and aggregate held-out ROC-AUC on
the same records must be identical; this identity cannot independently establish
ranking generalization. Per-template ranking and threshold transfer are the
informative checks. The score is not a calibrated default probability. Financial
failure remains the reviewed union of nonpositive EBITDA, payment default,
insolvency/funding deficit and covenant breach. Preserve subtype fields without
redefining the primary label. No generator, engine or reviewed artifact changes.

Before publishing, require disjoint train/test IDs in each fold, exhaustive unique
held-out coverage, unchanged labels/scores, exact prediction formulas and
deterministic thresholds under row permutation. Adversarial tests must establish
held-out label/score independence, training-only fitting, template integrity,
lowest-threshold tie handling and held-out-only aggregation. Run the new tests,
foundation tests, complete suite/coverage, Ruff lint and formatting checks.

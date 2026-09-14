# V3 template-held-out evaluation

Training-only threshold selection improves aggregate recall from 15% to 75% and
balanced accuracy from 0.575 to 0.769444, while introducing 38 false positives.
Threshold transfer is poor for several templates. These results materially
qualify the strong pooled v2 ranking result; they do not establish real-world
generalization or a deployable common threshold.

## Protocol and provenance

The [frozen protocol](V3_TEMPLATE_EVALUATION_PROTOCOL.md), evaluator and tests were
committed as `75f1b56d7613389012f7cc7e25ae95bcb4ddd004` before computing final
held-out performance. Run `python -m analysis.run_v3_template_evaluation` from the
repository root. Inputs and protocol SHA-256 hashes, source commit, command,
Python and package versions are recorded in
`results/v3/template_evaluation/summary.json`. `verification.json` records the
artifact reproduction check and file hashes. Reproduction on a later commit
updates source provenance; numerical results remain deterministic.

The evaluator reads unchanged reviewed-v2 records with string scenario IDs and
round-trip score precision. Each `operator_id` defines a structural template.
Each fold trains on all other templates and applies its chosen threshold to the
excluded template. No new paths were simulated. The generator, logistic score
coefficient, analytic headroom covenants, financial engine and failure union are
unchanged. The score is a ranking score, not a calibrated default probability.
Original failure subtype/flag fields are retained in held-out predictions;
payment default, funding deficit, covenant breach and nonpositive EBITDA remain
distinct concepts within the primary financial-failure union.

The sole threshold objective is maximum training balanced accuracy,
`0.5 * (sensitivity + specificity)`. Candidates are sorted unique training scores
plus 0 and the next representable float above 1. The latter includes an
all-negative classifier even when a score equals 1. Predictions use `score >=
threshold`. Ties within absolute `1e-12` of the maximum select the lowest
threshold, weakly favoring recall. Test scores and labels never enter candidate
generation or threshold fitting. Full training curves are archived for all five
folds. No secondary objective or post-result tuning was used.

## Data and folds

All 200 scenario IDs are unique, nonempty strings with stable template IDs.
There are 20 financial failures. No template has all failures; template 005 has
none. Every training set contains both classes. IDs below abbreviate
`SYN_HOTEL_` plus the displayed suffix.

| Held-out template | Test scenarios | Test failures | Train scenarios | Train failures | Selected threshold | Train BA |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 38 | 5 | 162 | 15 | 0.04780082990379734 | 0.949660 |
| 002 | 32 | 3 | 168 | 17 | 0.04735235093945553 | 0.869108 |
| 003 | 47 | 1 | 153 | 19 | 0.04735235093945553 | 0.902396 |
| 004 | 49 | 11 | 151 | 9 | 0.00967805410948959 | 0.954225 |
| 005 | 34 | 0 | 166 | 20 | 0.04735235093945553 | 0.882877 |

| Template | TP | FP | TN | FN | Sensitivity | Specificity | FNR | FPR | BA | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 1 | 0 | 33 | 4 | 0.200000 | 1.000000 | 0.800000 | 0.000000 | 0.600000 | 0.969697 | 0.871111 |
| 002 | 3 | 1 | 28 | 0 | 1.000000 | 0.965517 | 0.000000 | 0.034483 | 0.982759 | 0.988506 | 0.916667 |
| 003 | 0 | 0 | 46 | 1 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 |
| 004 | 11 | 37 | 1 | 0 | 1.000000 | 0.026316 | 0.000000 | 0.973684 | 0.513158 | 0.978469 | 0.933976 |
| 005 | 0 | 0 | 34 | 0 | undefined | 1.000000 | undefined | 0.000000 | undefined | undefined | undefined |

Undefined rates have missing denominators. Both ranking metrics and BA are
undefined for a single-class test fold under the frozen policy. Precision is
undefined if there are no positive predictions; F1 is `2TP/(2TP+FP+FN)`, undefined
when its denominator is zero. Machine-readable missing values are JSON null or
empty CSV cells. Template 005 remains in aggregate classification and ranking.
`fold_results.csv` also includes precision, F1 and all fixed-0.5 fold metrics.

## Aggregate held-out results

All metrics below use the same 200 concatenated held-out rows, exactly once each.
Fixed-0.5 predictions were recomputed from these scores, not copied from v2 counts.

| Classification metric | Training-selected threshold | Fixed 0.5 |
|---|---:|---:|
| TP / FP / TN / FN | 15 / 38 / 142 / 5 | 3 / 0 / 180 / 17 |
| Sensitivity / recall | 0.750000 | 0.150000 |
| Specificity | 0.788889 | 1.000000 |
| FNR | 0.250000 | 0.850000 |
| FPR | 0.211111 | 0.000000 |
| Balanced accuracy | 0.769444 | 0.575000 |
| Precision | 0.283019 | 1.000000 |
| F1 | 0.410959 | 0.260870 |

Confusion matrices in row order actual negative/positive and column order
predicted negative/positive are `[[142,38],[5,15]]` (selected) and
`[[180,0],[17,3]]` (fixed). Selected thresholds catch 12 additional failures at
the cost of 38 additional false alarms. Fixed 0.5 is poor for recall and the
prespecified balanced-accuracy objective on these records; this comparison does
not establish economic superiority without a cost/utility model.

Aggregate ROC-AUC is **0.9472222222222222** and PR-AUC, defined specifically as
**average precision**, is **0.7805768293204627**. AP is not trapezoidal PR area.
The historical v2 pooled ROC-AUC and recomputed pooled ROC-AUC are both
0.9472222222222222. Recomputed pooled AP is also 0.7805768293204627.

This exact equality is necessary: the analytic score is fixed and untrained, and
all original score/label pairs occur once in the held-out concatenation. Excluding
templates from threshold training cannot change score ranking. Consequently,
aggregate AUC equality is not independent evidence that ranking generalizes.
Within-template ROC-AUC is 0.969697–1.0 in the four evaluable folds, supporting
ranking within these synthetic templates, with very few positives in some folds.
No significance claim or real-world generalization claim is justified.

## Dispersion, weak folds and limitations

Primary uncertainty reporting is descriptive fold dispersion, not a confidence
interval. Only five synthetic templates are available, training folds overlap,
and one fold contains no failures. A group bootstrap would depend heavily on
which of these few templates were drawn. Neither the old scenario bootstrap nor
a newly manufactured precise interval is used.

| Quantity | Defined folds | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Threshold | 5 | 0.00967805410948959 | 0.04735235093945553 | 0.04780082990379734 |
| Sensitivity | 4 | 0.000000 | 0.600000 | 1.000000 |
| Specificity | 5 | 0.026316 | 1.000000 | 1.000000 |
| FNR | 4 | 0.000000 | 0.400000 | 1.000000 |
| FPR | 5 | 0.000000 | 0.000000 | 0.973684 |
| Balanced accuracy | 4 | 0.500000 | 0.556579 | 0.982759 |
| ROC-AUC | 4 | 0.969697 | 0.983487 | 1.000000 |
| Average precision | 4 | 0.871111 | 0.925321 | 1.000000 |

Thresholds span almost fivefold. Template 004's threshold is much lower than the
other four; it produces 37 false positives out of 38 non-failures, precision
0.229167 and BA 0.513158, below its fixed-0.5 BA of 0.636364. This fold contains
11 of 20 failures and contributes 37 of 38 aggregate false positives. Template
001 misses four of five failures. Template 003 misses its only failure despite
perfect within-fold ranking. These are substantive threshold-transfer failures,
not obscured by aggregate improvement. Template 002 transfers well; template 005
only demonstrates correct classification of 34 non-failures.

The low median fold BA (0.556579) versus aggregate BA (0.769444) reflects differing
class counts across templates; the aggregate is not an equal-weight template
average. Full dispersion, including precision/F1 and fixed-0.5 metrics, is in
`summary.json`. The five structural templates and seed-42 synthetic draw are not
a representative population sample. The exercise validates exclusion during
threshold fitting; it cannot undo any earlier benchmark design choices or assess
new real operators. Strong ranking is compatible with unstable score levels and
poor threshold transfer. No score recalibration was attempted.

## Verification and stage boundary

All 34 template tests pass. They verify actual training membership, held-out label
and score mutation independence in every toy fold, rejection of duplicate IDs
within/across groups, lowest tied optimum, a toy pooled optimum different from
the training optimum, hand-counted aggregate confusion/ranking metrics, exact
held-out coverage, row-order determinism, single-class policies, invalid inputs,
boundary candidates and rejection of corrupted held-out artifacts.

Runtime sanity gates passed before aggregation: exhaustive unique coverage,
disjoint train/test IDs, unchanged original rows, deterministic selection under
reversed training order, and exact selected/fixed prediction formulas. A repeat
run reproduced every generated file byte-for-byte at the source commit; source
CSV scores and labels also match the archived benchmark JSON exactly.

Validation commands and results:

- `python -m pytest -q --no-cov tests/test_template_evaluation.py`: 34 passed.
- `python -m pytest -q --no-cov tests/test_returns_independent.py tests/test_runtime_invariants.py`: 14 passed.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed (32 files).
- `python -m pytest -q --cov=src/lbo --cov-report=term-missing --cov-fail-under=52`:
  124 passed, 57.95% aggregate statement coverage; full simulation 100%.

Only template-held-out threshold evaluation is completed here. Bayesian admission
and validation, posterior simulation, financing optimization admission/validation
and v3 manuscript/figure work remain outside this stage.

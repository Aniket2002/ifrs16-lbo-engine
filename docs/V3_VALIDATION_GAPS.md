# V3 baseline checkpoint and remaining validation gaps

## Validation already complete

Canonical branch: `v3/validated-rebuild`, created directly from reviewed artifact
head `a990dd2166f47917781112a7e12bbc16f33605fd`. Reviewed manuscript source
`f07796efc537468122804e6eae25b1a4f284efbb` and repaired financial baseline
`3903eccbf4650e84691aac9834caacdda0c7f5e5` are ancestors. The initial working tree
was clean. After fetching, no reachable commits existed outside reviewed v2;
local main was 16 commits behind and remote main four commits behind. No merge,
cherry-pick, force-push, history rewrite or branch deletion was necessary.

Reviewed v2 is the validated starting baseline for v3:

- Ruff lint and formatting passed; all 76 existing tests passed.
- Aggregate statement coverage remains 52.13%; full simulation coverage is 100%.
- The existing v2 artifact verifier passed before regeneration.
- The existing full benchmark/figure generator, Tectonic build and PDF verifier
  passed: 200 scenarios, five annual observations each, six figures and 16 pages.
- Every archived financial/statistical report field and scenario record reproduced
  exactly. Only timing measurements and the actual run's Git SHA differ.
- Original reviewed v2 files were backed up and restored byte-for-byte. Fresh
  results are preserved separately in `results/v3/baseline/`; no v2 value was
  silently replaced. No corrected v2.x baseline is needed.

The existing waterfall regressions, source/use checks, zero-equity handling,
overfunded-entry rejection, default aggregation and v2 artifact verification
should be reused, not recreated. See `results/v3/baseline/verification.json`,
`branch_start.json` and the archived pipeline logs for this checkpoint.

## Foundation validation completed

- Independent geometric IRR and MOIC expectations now match production exactly;
  a five-year debt/interest schedule also matches its hand-derived values exactly.
- Negative exit equity remains an unfloored diagnostic, producing negative MOIC
  and no finite IRR. Zero exit proceeds likewise have no finite IRR. The sponsor
  vector supports only entry, zero intermediate flows and one exit flow, so it
  cannot generate multiple sign changes.
- `src/lbo/validation.py` now checks all material closing, cash, debt, revolver,
  amortisation, default, sweep, lease, roll-forward and finite-value invariants.
  It records zero ratio denominators explicitly rather than treating them as valid
  ratios. The seed-42 benchmark's 1,000 simulation years all pass.
- Six required adversarial corruptions are detected with invariant, scenario,
  year, expected/actual values, assumptions, current/previous row state, tolerance
  and source commit available as JSON. V3 execution writes this record before
  propagating a failure.
- No production simulation or return logic changed. Reviewed-v2 results remain
  the comparison baseline; the complete reproduction check is recorded in the
  foundation report.

## Template-held-out evaluation completed

- Frozen training-only balanced-accuracy selection and lowest-threshold tie rule
  now have adversarial leakage and aggregation tests. All 200 archived scenarios
  appear exactly once in five held-out folds; no engine or generator change.
- Aggregate recall improves from 0.15 to 0.75 and balanced accuracy from 0.575 to
  0.769444, with 38 false positives. Thresholds range from 0.009678 to 0.047801;
  poor transfer in templates 001, 003 and 004 materially qualifies pooled results.
- Fixed-score aggregate ROC-AUC necessarily equals pooled v2 AUC (0.947222);
  this identity is not independent evidence of ranking generalization. Fold
  dispersion is reported without a precise confidence interval from five groups.
- All 34 new tests and 14 foundation tests pass; the full suite passes 124 tests
  with 57.95% coverage and full simulation coverage of 100%. See
  `docs/V3_TEMPLATE_EVALUATION.md` and `results/v3/template_evaluation/`.

## Genuinely missing work

- A validation gate for the existing Bayesian module: data/provenance,
  likelihood/identifiability, prior predictions, independent recovery,
  posterior predictions, diagnostics and additional value. No current test module
  exercises the calibrator. Its named MAP/Laplace fallback computes sample moments,
  and predictive generation uses fitted hyperparameter summaries; these need
  explicit scrutiny before calling outputs a posterior. This inspection is not
  a completed Bayesian validation or an admission decision.
- Independent sponsor-return validation and an economic admission decision for
  financing optimization. Any retained search needs independently solved toy cases
  and infeasible/tied/boundary/binding-constraint tests. Threshold tuning alone
  is not an economic optimization.
- V3 publication artifacts, only after numerical methods pass their gates.

## Proposed v3 implementation order

1. Template-held-out selection and adversarial tests are complete; preserve the
   frozen protocol and its documented threshold-transfer limitations.
2. Audit/test the existing Bayesian module against the stated admission gate.
   Retain posterior simulation only if the gate passes; otherwise record exclusion.
3. Define and independently solve a small genuine financing problem before any
   larger search. Retain optimization only with defensible economics and tests.
4. Run the fixed v3 protocol, archive machine-readable outputs and provenance,
   then extend figure/table generation and update the manuscript and changelog.

## Files expected to change

- Foundation files now added: `tests/test_returns_independent.py`,
  `tests/test_runtime_invariants.py`, `src/lbo/validation.py`, and
  `analysis/run_v3_foundation.py`. `tests/test_template_evaluation.py` and
  `analysis/run_v3_template_evaluation.py` are now also complete;
  Bayesian/optimization tests depend on their validation and admission decisions.
- Audit `analysis/calibration/bayes_calibrate.py`; amend it only if justified by
  independent failures. Any retained optimizer belongs in a separate tested module.
- Add versioned configuration and reports under `docs/` and `results/v3/`.
- After numerical validation, add v3 figure/table generation and
  `paper/ifrs16_lbo_ssrn_v3.tex`/PDF, with reproduction instructions and changelog.
  Preserve the reviewed v2 source and artifacts.

Branch consolidation, baseline reproduction, foundation financial validation and
template-held-out threshold evaluation are complete. No Bayesian analysis,
optimization, financial model change or manuscript rewrite was undertaken in this stage.

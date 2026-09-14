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

## Genuinely missing work

- Independent geometric IRR expectation and a hand-calculated fixed debt/interest
  schedule. The existing IRR check uses `numpy_financial.irr`, also used by
  production; it is not independent. Supplement existing zero-equity/empty-return
  regressions with negative-exit and no-valid-root cases without duplicating them.
- A reusable runtime invariant validator that fails every affected run and archives
  failing inputs/year state. Existing tests and v2 post-run cash checks do not
  enforce all requested invariants on every newly simulated path.
- Leave-one-template-out threshold selection with training-only decisions,
  held-out aggregate classification metrics, and adversarial leakage checks.
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

1. Independent return and debt checks; reusable runtime invariants with preserved
   failure state. Keep the financial engine unless an independent check fails.
2. Record the threshold selection objective, tie rule and uncertainty protocol
   before evaluating; implement template-held-out selection and adversarial tests.
3. Audit/test the existing Bayesian module against the stated admission gate.
   Retain posterior simulation only if the gate passes; otherwise record exclusion.
4. Define and independently solve a small genuine financing problem before any
   larger search. Retain optimization only with defensible economics and tests.
5. Run the fixed v3 protocol, archive machine-readable outputs and provenance,
   then extend figure/table generation and update the manuscript and changelog.

## Files expected to change

- Add focused `tests/test_returns_independent.py`, `tests/test_runtime_invariants.py`
  and `tests/test_template_evaluation.py`; Bayesian/optimization tests depend on
  their validation and admission decisions.
- Add `src/lbo/validation.py` and a v3 evaluation entry point such as
  `analysis/run_v3.py`, reusing the existing benchmark sampler and financial engine.
- Audit `analysis/calibration/bayes_calibrate.py`; amend it only if justified by
  independent failures. Any retained optimizer belongs in a separate tested module.
- Add versioned configuration and reports under `docs/` and `results/v3/`.
- After numerical validation, add v3 figure/table generation and
  `paper/ifrs16_lbo_ssrn_v3.tex`/PDF, with reproduction instructions and changelog.
  Preserve the reviewed v2 source and artifacts.

This checkpoint completes branch consolidation and baseline reproduction only.
No v3 numerical methodology, financial-model changes or manuscript rewrite has
been undertaken, and no Bayesian or optimization result has been admitted.

# V3 post-optimization diagnostics

## Scope and guardrails

This checkpoint is intentionally narrow: it only reads the already-frozen optimization outputs and template-evaluation artifacts, and it does not reopen the optimization admission decision or modify the frozen financing-policy search design. The purpose is to document the diagnostic evidence that emerged after the optimization validation stage without altering any methodology or manuscript scope.

## 1. Optimization-only coverage

The optimization validation run produced the following coverage summary for the optimization code path itself:

- analysis/optimization/financing_policy.py: 84% statement coverage, 70% branch coverage
- analysis/run_v3_optimization.py: 59% statement coverage, 85% branch coverage

The coverage is enough to confirm the main candidate-grid and risk-constraint logic, selection logic, and the deterministic single-run path were exercised, while leaving artifact-writing, end-to-end orchestration, and some error-handling paths outside the focused unit suite.

Key functions that were exercised include:

- candidate_grid
- limited_liability_returns
- summarize_outcomes
- select_candidate
- solve_known_optimum_toy
- build_operating_scenarios
- assumptions_for
- evaluate_one
- experiment_at_rate
- leave_one_template_out
- main

The interpretation is therefore not that the optimization is unvalidated, but that the validation is strongest on the search kernel and weaker on a few runner-level convenience paths and output plumbing.

## 2. Within-grid debt boundary behavior

The frozen candidate grid spans debt / EBITDA from 1.50x to 3.00x in 0.25x increments. The debt-boundary profile generated from the candidate results shows the following pattern:

- All five held-out folds classify as monotone decreasing with debt.
- In each fold, the objective deteriorates as debt rises across the tested grid.
- All five fold-level selections landed at the minimum debt boundary, 1.50x.
- Risk metrics also trend in the same direction: the best feasible candidates at higher debt levels have larger broad-failure and payment-default exposure.

This yields the following diagnostic conclusion:

- Within the frozen search range, lower debt is systematically favored.
- The result is consistent with a boundary-limited optimization rather than a true interior optimum.
- The protocol correctly forbids expanding the grid after results are observed, so the experiment cannot establish whether performance would continue improving below 1.50x.

The key point is not that the model says 1.50x is globally optimal; it is that, within the pre-specified grid, the search consistently sits at the lower leverage edge.

## 3. Template 004 regime breakdown

Template 004 remains the weakest fold in the leave-one-template-out optimization and remains structurally weak after the policy improvement. The regime decomposition from the held-out results shows that the poor performance is not caused by an unlucky regime mix alone.

The template-specific counts in the held-out comparison are:

- optimized: base 48, downside 31, distressed 21
- reference: base 48, downside 31, distressed 21

The intended regime probabilities were 50% base, 30% downside, 20% distressed; actual realized regime counts differ because the synthetic sample is finite, but the pattern is stable across the different policy comparisons.

For Template 004, the risk profile under both policies remains poor even within the same regime buckets:

- optimized base: median annualized return 2.58%, broad failure 0%, payment default 0%
- optimized downside: median -7.84%, broad failure 0%, payment default 0%
- optimized distressed: median -18.85%, broad failure 33.3%, payment default 14.3%
- reference base: median 2.16%, broad failure 2.1%, payment default 2.1%
- reference downside: median -11.44%, broad failure 29.0%, payment default 22.6%
- reference distressed: median -29.99%, broad failure 85.7%, payment default 76.2%

The distressed regime is the principal driver of the weakness, but the weakness remains present even in the base/downside cases, which indicates that Template 004 is structurally difficult rather than merely regime-misaligned.

## 4. Structural comparison of Template 004

The synthetic operator file shows that Template 004 is not a generic “normal” archetype. In terms of balance-sheet and operating structure, it sits at the more stressed end of the synthetic operator mix, with higher leverage and lower cash cushion than the other templates. This is consistent with the optimization result that it remains the worst fold even after policy selection.

The operative interpretation is that the model is not finding a single universal financing policy that neutralizes a structurally weak archetype. Instead, the same policy remains challenged where the operating structure carries elevated debt burden and lower resilience.

## 5. Cross-stage evidence: threshold transfer and structural heterogeneity

The template-evaluation artifact for Template 004 shows the same weak archetype under a separate decision layer:

- selected threshold: 0.00967805410948959
- test false positives: 37
- specificity: 0.02631578947368421
- false-positive rate: 0.9736842105263158
- balanced accuracy: 0.5131578947368421
- fixed 0.5 threshold balanced accuracy: 0.6363636363636364

This is not a proof of a model failure in the abstract; rather, it is evidence that the same synthetic archetype is weak under both a learned threshold rule and a fixed threshold rule. Combined with the financing-optimization result, this gives convergent descriptive evidence that structural heterogeneity matters across decision layers and that Template 004 is a consistently difficult case rather than a one-off artifact.

## 6. Diagnostic conclusion

The checkpoint does not alter the classification or the methodological status of the financing-optimization stage. The evidence is best summarized as follows:

- The optimization search kernel is mechanically valid and deterministic.
- All fold-level selections sit at the low-debt boundary in the frozen grid.
- The objective declines as debt rises within the tested range.
- Template 004 remains structurally weak under both the optimized and reference policies and across multiple decision layers.
- The evidence is consistent with a within-grid preference for lower leverage and a heterogeneous structural penalty, not with a claim that 1.50x is a globally optimal debt level.

This means the result remains a methodological demonstration rather than an economically substantive financing recommendation. The frozen protocol and B classification remain intact.

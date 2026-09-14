# Validation summary

## 1. Validation philosophy

The validated framework separates three questions:

```text
engine correctness -> ranking performance -> decision usefulness
```

Evidence at one layer does not establish the next. Mechanical tests can show that
the implemented cash and debt logic behaves consistently on tested paths. Ranking
metrics can show ordering within the frozen synthetic sample. A usable decision
rule additionally requires threshold transfer, explicit costs or utility, and
evidence relevant to the intended population.

## 2. Engine correctness

Independent return and debt schedules agree with production calculations, and
the external runtime validator checks every material annual reconciliation.
Adversarial tests detect corrupt cash, debt, revolver, amortisation, sweep, and
opening-balance states. The fixed seed-42 foundation run validated all 1,000
annual rows from 200 synthetic paths.

This supports tested accounting and financing mechanics. It does not prove the
engine correct for every possible input or validate its economic conventions
against real transactions. See [foundation validation](V3_FOUNDATION_VALIDATION.md)
and `results/v3/foundation_validation.json`.

## 3. Ranking performance

The reduced-form score ranks broad financial failure strongly in the pooled frozen
synthetic records. The aggregate held-out AUC equals the pooled result because the
score itself is fixed and every original score/label pair appears once; exclusion
from threshold fitting cannot change score order. The identity is a consistency
check, not independent proof of ranking generalization.

The score is not a calibrated probability of default. Broad failure is a union
of payment, funding, covenant, and nonpositive-EBITDA conditions.

## 4. Threshold portability

Thresholds chosen only on four archetypes improve aggregate recall and balanced
accuracy relative to the fixed 0.5 rule, at the cost of 38 false positives. Their
fold behavior is uneven. Templates 001 and 003 miss most or all positive cases;
Template 004 contributes 37 of the 38 false positives; Template 005 contains no
positive cases, so several fold metrics are undefined.

These results qualify the pooled ranking result. They do not establish a universal
threshold, a deployable policy, or performance on a borrower population. The
detailed results and frozen selection rule remain in
[the evaluation report](V3_TEMPLATE_EVALUATION.md) and
[protocol](V3_TEMPLATE_EVALUATION_PROTOCOL.md).

## 5. Structural heterogeneity

Template 004 is consistently difficult across separate decision layers. Its
transferred screening threshold has very low specificity, and it remains the
principal source of financing-experiment risk. Regime-conditioned diagnostics
show that distress contributes heavily, while the weakness is not explained by
regime mix alone. This is descriptive evidence of heterogeneity among five
stipulated archetypes, not a causal subgroup finding.

See [post-optimization diagnostics](V3_POST_OPTIMIZATION_DIAGNOSTICS.md) and the
machine-readable diagnostics under `results/v3/post_optimization_diagnostics/`.

## 6. Bayesian experiment — excluded

The corrected Bayesian implementation improved model honesty and passed much of
the synthetic recovery exercise, but the frozen technical gates did not all pass,
prior sensitivity was material for selected parameters, and the named-firm input
variables lack repository-level provenance. The admission decision is therefore
exclusion from substantive v3 results. No posterior-driven LBO integration was
performed.

See the [frozen protocol](V3_BAYESIAN_VALIDATION_PROTOCOL.md),
[validation report](V3_BAYESIAN_VALIDATION.md), and
`results/v3/bayesian_validation/`.

## 7. Financing experiment — methodological only

The separate financing experiment validates a deterministic, leakage-safe
constrained grid-selection procedure. Selected policies improve the reported
aggregate risk measures and modestly improve return summaries relative to the
stipulated reference on held-out synthetic paths. Every fold selects the minimum
debt boundary, most select the maximum sweep, and rate sensitivity changes policy
components in two folds.

The result is retained as a methodological demonstration. It does not establish
that the boundary debt level is globally or economically optimal. See the
[protocol](V3_OPTIMIZATION_VALIDATION_PROTOCOL.md),
[validation report](V3_OPTIMIZATION_VALIDATION.md), and
`results/v3/optimization_validation/`.

## 8. Remaining limitations

- The benchmark has five synthetic archetypes and sparse failures, not a sampled
  borrower population.
- No external validation, causal design, calibrated probability model, or
  decision-cost model is present.
- The analytic and simulation models differ in structure, timing, and inputs.
- The financing experiment omits endogenous pricing, recoveries, refinancing,
  and stakeholder utility.
- Post-default continuation and negative raw exit equity are diagnostics rather
  than recovery estimates.

## 9. Detailed provenance

The [documentation index](README.md) lists all frozen protocols, detailed
validation reports, manuscript-governance records, and machine-readable evidence.
The claim ledger and prohibited-claim registry remain internal audit records and
are not substitutes for the paper's reader-facing conclusions.

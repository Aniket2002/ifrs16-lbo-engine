# V3 financing-optimization validation

## Admission decision

**B — RETAIN ONLY AS A TOY / METHODOLOGICAL DEMONSTRATION.** The exhaustive
search, independent toy, risk constraints, ex-ante timing, leave-one-template-out
selection and runtime invariants validate mechanically. The result does not earn a
substantive financing-design claim because all five folds choose the minimum debt
boundary, four choose the maximum cash-sweep boundary, and a modest fixed-rate
change alters two fold policies. The experiment has no endogenous credit pricing,
default recovery or empirical financing calibration.

The frozen protocol and source were committed as
`fdc949f4ff0c8ef4b3f318e31aff7a4a9cd6cccf` before final outcomes were computed.
Machine-readable results and hashes are under
`results/v3/optimization_validation/`. No v2, financial-engine, foundation,
template or Bayesian artifact changed.

## Economic audit

**EXISTING V2 BENCHMARK IS NOT A VALID RETURN-OPTIMIZATION DATASET.** In
`analysis/run_benchmark.py`, entry EV is `7.5 * revenue_0`; in the financial
engine, exit EV is final EBITDA times the default 8.0 multiple. Repository
documentation calls entry EV a stipulated acquisition use but does not establish
7.5x revenue as an intentional valuation basis comparable to the EBITDA exit
basis. V2 explicitly states that its benchmark supplies no equity-return or
optimization result.

V2 also changes opening debt and cash after drawing a downside/distressed regime.
That is acceptable as a stress-scenario construction for its ranking experiment,
but it cannot represent an ex-ante financing choice. These findings do not affect
v2 ranking, covenant, failure or template-threshold conclusions. The reviewed v2
benchmark remains byte-unchanged.

## Separate synthetic experiment

The v3 experiment uses five existing synthetic archetypes and 100 shared operating
paths per template. Entry EV is 7.5 times known entry EBITDA and exit EV is 8.0
times final EBITDA. Both therefore use EBITDA. Fees are 3%, the holding period is
five years, term interest is fixed at 6%, lease interest at 5%, minimum cash at
half known opening cash, and revolver capacity at 0.75 times entry EBITDA.

Seed 314159 draws base/downside/distressed regimes with probabilities 50%/30%/20%.
Only future growth and EBITDA margin vary. Opening debt, cash, lease liability and
financing policy never depend on the realized regime. Every candidate and
reference policy sees identical paths.

The model still has an entry/exit lease-treatment limitation: entry uses do not
adjust acquisition price for the assumed opening lease, while exit equity deducts
the remaining lease. This is the validated engine convention, retained without
change and one reason results remain methodological. The engine also continues
after payment default or funding deficit and supplies no recovery process.

## Frozen policy, reference and selection rule

The common decision vector is opening term debt / known entry EBITDA, annual
scheduled amortisation / opening term debt, and cash sweep. The complete 112-point
grid is:

- debt multiple: 1.50x to 3.00x in 0.25x increments;
- amortisation rate: 5.0%, 7.5%, 10.0%, 12.5%;
- cash sweep: 40%, 50%, 60%, 70%.

This brackets existing debt/EBITDA of 1.667x–2.600x and cash sweep of 48%–60%.
Existing amortisation rates are 4.615%–7.692%; the 5% lower grid point is within
0.385 percentage points of the lowest reference and the grid extends above all
references. No grid dimension was changed after outcomes were seen.

| Template | Debt / EBITDA | Annual amortisation | Amortisation / debt | Sweep | Opening cash | Opening lease |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 2.000x | 30 | 5.769% | 55% | 45 | 390 |
| 002 | 2.293x | 30 | 6.383% | 50% | 35 | 360 |
| 003 | 1.848x | 30 | 4.918% | 58% | 55 | 470 |
| 004 | 2.600x | 30 | 7.692% | 48% | 30 | 300 |
| 005 | 1.667x | 30 | 4.615% | 60% | 60 | 500 |

The reference is only the existing synthetic policy, not market practice.
Candidate debt must remain below entry uses and leave positive sponsor equity.

Raw exit equity is preserved. Limited-liability sponsor proceeds are
`max(0, raw exit equity)`. MOIC is proceeds / positive initial sponsor equity;
annualized return is `MOIC^(1/5)-1`. Zero recovery is MOIC 0 and annualized return
-100%, not NaN.

Each fold selects on the other four templates. Feasible candidates must have both
training broad-failure and payment-default rates no greater than the reference on
the same paths. The objective is maximum training median limited-liability
annualized return. Ties within 1e-12 prefer lower broad failure, lower payment
default, lower debt, lower sweep and higher amortisation, in that order.

## Independent toy validation

For entry uses 100, exit EV 120 and debt `D` in [0,60], with no interest, lease,
amortisation or uncertainty, MOIC is `(120-D)/(100-D)`. Its derivative is
`20/(100-D)^2`, strictly positive throughout the domain. The independently known
optimum is therefore D=60. Exhaustive search over 0, 15, 30, 45 and 60 selects
**D=60 with MOIC 1.5**, as required.

Tests also establish single-candidate selection, explicit infeasibility,
conservative ties, a risk constraint removing an unconstrained optimum,
determinism, held-out mutation independence, row-order invariance, zero-recovery
returns, rejection of zero sponsor equity and rejection of debt reaching entry
uses.

## Leave-one-template-out selections

All folds are feasible. Depending on the held-out template, 36–56 of 112
candidates satisfy the training risk constraints.

| Held-out template | Debt | Amortisation | Sweep | Training median return | Held-out optimized median | Held-out reference median |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 1.50x | 5.0% | 70% | 1.247% | 0.111% | -0.298% |
| 002 | 1.50x | 7.5% | 70% | 1.117% | 0.861% | 0.302% |
| 003 | 1.50x | 5.0% | 60% | 0.354% | 1.819% | 1.666% |
| 004 | 1.50x | 7.5% | 70% | 1.799% | -3.181% | -5.116% |
| 005 | 1.50x | 5.0% | 70% | -0.024% | 2.745% | 2.682% |

All five select minimum debt; four select maximum sweep; none selects maximum
amortisation. Template 004 remains economically weak even after improvement: its
median return is -3.18%, mean -6.86%, and 10th percentile -23.04%. The fold has
7% broad failure and 3% payment default. The result does not average away this
structurally weak template.

## Aggregate held-out comparison

Each of 500 operating scenarios occurs exactly once under its training-selected
policy and once under its archetype reference.

| Return / capital metric | Optimized | Reference |
|---|---:|---:|
| Median annualized return | 1.101% | 0.612% |
| Mean annualized return | -1.111% | -3.261% |
| Median MOIC | 1.0563x | 1.0310x |
| 10th-percentile annualized return | -15.206% | -18.183% |
| Total equity-loss probability | 0.2% | 1.8% |
| Median initial sponsor equity | 1,663.5 | 1,533.5 |
| Median opening debt | 390.0 | 520.0 |
| Median ending term debt | 0.0 | 146.2 |
| Mean maximum revolver | 5.4 | 15.5 |

The “optimized” policy commits more sponsor equity and less debt. Its higher
return is therefore not leverage amplification; lower interest/default exposure
and faster debt reduction dominate in these stipulated paths. The median-return
difference is 0.488 percentage points, modest relative to the assumptions and
negative downside returns.

| Held-out risk metric | Optimized | Reference | Difference |
|---|---:|---:|---:|
| Broad financial failure | 1.4% | 6.8% | -5.4 pp |
| Payment default | 0.6% | 5.0% | -4.4 pp |
| Funding deficit / insolvency flag | 1.2% | 5.8% | -4.6 pp |
| Covenant breach | 1.4% | 4.8% | -3.4 pp |

No held-out template has a higher optimized broad-failure or payment-default rate
than its reference. Template 003 has zero failures under either policy. Template
004 improves broad failure from 28% to 7% and payment default from 24% to 3%, but
remains the sole material source of aggregate risk.

## Fixed-rate sensitivity

At a 7.5% term rate, selected debt remains 1.50x in every fold. Templates 002 and
003 change other policy components: 002 amortisation moves from 7.5% to 5.0%; 003
sweep moves from 60% to 70%. Under the frozen criterion, two policy changes make
rate sensitivity material. Aggregate optimized median return declines from 1.101%
to 0.938%, an absolute 0.163-percentage-point change, below the separate two-point
return threshold.

Higher-rate optimized/reference broad failure is 1.8%/8.8%, and payment default
1.2%/6.8%. The direction of the comparison is stable, but component-level policy
selection is assumption-sensitive. Because borrowing costs never rise with debt,
this sensitivity cannot stand in for an endogenous credit-spread model.

## Interpretation and limitations

Mechanics validate and the constrained policies improve all reported aggregate
risk measures and modestly improve return metrics. Yet the apparent result is a
corner solution: minimum debt in all five folds and maximum sweep in four. It
mainly shows that this synthetic engine and fixed assumptions favor a more
equity-funded, faster-paydown structure than the supplied reference.

That is useful as a methodological demonstration of leakage-safe constrained grid
selection, but it is not a defensible claim that 1.50x debt is economically
optimal. There is no leverage-dependent pricing, refinancing behavior, default
recovery, lender/sponsor utility, interim distribution, tax-shield calibration,
market evidence or transaction-level uncertainty. The engine's mechanical
post-default continuation makes default-path returns diagnostics rather than
recovery estimates. The grid boundary prevents knowing whether still lower debt
would be preferred; the frozen protocol correctly forbids expanding it after the
result.

Classification A required no pervasive boundary behavior, stable rate-policy
selection and defensible incremental information. It fails the boundary and rate
criteria. Classification B is therefore the pre-specified outcome. Historical v1
IRR-uplift numbers remain superseded and were neither used nor reconstructed.

## Verification

The run validated **113,000 fresh five-year simulation paths**. The artifact gate
confirms 560 fold-candidate rows, 1,000 held-out comparison rows, unique exhaustive
coverage, exact limited-liability formulas, and unchanged protected artifacts.

- Optimization tests: 15 passed.
- Optimization, foundation, template and Bayesian tests: 77 passed.
- Full suite: 153 passed, 57.95% statement coverage.
- Full simulation coverage: 100%.
- Ruff lint and formatting: passed.

Optimization validation is complete. Final manuscript and figure work remain a
separate stage. Bayesian results remain excluded, and no posterior-driven LBO
integration was performed.

# SSRN v2 revision audit

The baseline implementation is remote head
`3903eccbf4650e84691aac9834caacdda0c7f5e5` of
`agent/ifrs16-final-waterfall-consistency`, including `5446bff`'s final waterfall
and opening-cash repairs. The new branch is `paper/ssrn-v2-reconciliation`.
No simulation, covenant helper, calibration, test, or benchmark algorithm is
changed by this revision.

The complete requested manuscript was recovered from
`d99064367b144a147457bc67ab992e599e3b5c25:analysis/paper/main.tex` and preserved
byte-for-byte in `paper/historical/main.tex`, with its bibliography and separate
appendix sources. All old line numbers below refer to that recovered main.tex.
The current short note and empty placeholder files were not used as manuscripts.
The revised canonical source preserves the section-based academic presentation,
equations, bibliography, mathematical appendices and all six major figure sections.

Statuses describe the disposition of the old claim: SUPPORTED (retained with
evidence), RECOMPUTED (new results under the explicitly different protocol),
REFRAMED (narrower statement), REMOVED (not a revised-paper claim), and
NOT REPRODUCIBLE (no current executable evidence for the old result; excluded).
Recomputed metrics are not substitutions into the old four-method table.

## Empirical and methodological claim ledger

| Old claim | Source in old paper | Current support status | Action and current evidence |
|---|---|---|---|
| AUC 0.76 | Abstract L113; Introduction L135; baseline table L365; L396; conclusion L551 | RECOMPUTED | New synthetic union-failure AUC 0.9472222222; `benchmark_seed42.json:auc`. No protocol-equivalence claim. |
| 95% CI [0.71, 0.81] | Same locations; L374 | RECOMPUTED | [0.8974104054, 0.9828273300] from current scenario bootstrap; JSON `auc_ci_95`. |
| Headroom RMSE 0.28 vs 0.52; CI [0.24,0.33] | L113,135,362–365,397 | NOT REPRODUCIBLE | Comparator workflow absent. Report new headroom MAE 0.7777089369 and RMSE 1.4884123260, with explicit aggregation. |
| 46% RMSE improvement | L135,397,551 | REMOVED | No current traditional comparator, no improvement estimate. |
| +0.18 AUC; CI [0.12,0.24] | L135,378 | REMOVED | One analytic ranking evaluated against simulation labels; no fabricated comparator. |
| Delta RMSE -0.24; CI [-0.31,-0.17] | L378 | REMOVED | No paired comparator experiment. |
| Expected IRR 19.6%; 80% CI [18.2,21.1] | L365 | NOT REPRODUCIBLE | Benchmark emits no equity returns; opening cash changes sponsor equity. Remove empirical IRR estimate. |
| Expected IRR +3.4pp | L551 | REMOVED | No current return optimization protocol. Conceptual objective retained with explicit scope. |
| Traditional: AUC .58, RMSE .52, IRR 16.2% and all intervals | L362 | NOT REPRODUCIBLE | Remove result row; current ROC diagonal is a mathematical chance reference only. |
| Naive: AUC .64, RMSE .45, IRR 17.1% and all intervals | L363 | NOT REPRODUCIBLE | Remove result row; current implementation does not execute this comparator. |
| Bayesian: AUC .72, RMSE .34, IRR 18.4% and all intervals | L364 | NOT REPRODUCIBLE | Remove result row; benchmark never calls calibration module. |
| Ten operators, fifty scenarios each, 500 total | L372 | RECOMPUTED | Five synthetic templates sampled with replacement, 200 total; operator frequencies plotted from records. |
| Five-year, 20-quarter evaluation | L372,376 | REFRAMED | Five annual observations per scenario. No quarterly testing; 1,000 scenario-years. |
| Geographic regions and $1.2–4.8B borrower scale | L372; F13 caption L431 | REMOVED | Synthetic operator IDs/regimes replace unsupported borrower/geographic data interpretation. |
| 2,000 stratified operator-clustered bootstrap samples | L374 | RECOMPUTED | 500 scenario bootstrap attempts with seed 42, skip single-class samples. Not clustered or stratified. |
| BCa verification agrees within ±0.01 AUC | L374 | NOT REPRODUCIBLE | Current implementation uses percentile interval only. |
| Clustered RMSE CIs | L374 | NOT REPRODUCIBLE | Current report has no RMSE CIs. Distinguish mean scenario RMSE from pooled RMSE. |
| 70/30 train/test split, held-out quarters | L376 | REMOVED | No fitting or holdout. AUC is conditional synthetic discrimination, not borrower generalization. |
| 5-fold within-operator cross-validation | L376 | REMOVED | Not called by current benchmark. |
| Equal operator weighting | L376 | REFRAMED | Equal scenario weighting after sampling templates with replacement. |
| p < 0.001 clustered permutation test | L378 | NOT REPRODUCIBLE | No current statistical test or significance claim. |
| 72%/84%/78% time-points certified | L411 | NOT REPRODUCIBLE | No proved simulation budgets or current certification experiment. Preserve section as scope discussion. |
| 45%/68%/57% scenarios fully certified | L412 | NOT REPRODUCIBLE | Same; no certification percentages in v2. |
| 2.1%/1.8%/1.9% false non-certification | L413 | NOT REPRODUCIBLE | No current certification labels; not interchangeable with false-negative rate. |
| Bound utilization 68%/71%/70% | L414 | NOT REPRODUCIBLE | Replace figures with actual cross-model differences. |
| ICR 95th percentile .089 within bound .120 | L394 | NOT REPRODUCIBLE | F12 is signed error distribution, not a bound-validation plot. Current heuristic envelope .53312 is not a theorem. |
| Leverage 95th percentile .112 within bound .143 | L395 | NOT REPRODUCIBLE | Same; current heuristic envelope .3492391731 is not a theorem. |
| Median <3% relative error | F14 caption L424 | NOT REPRODUCIBLE | Replace with current ROC and MAE/RMSE panels. |
| Posterior frontiers, 80%/95% credible bands | L135; F14 L424 | NOT REPRODUCIBLE | Current score is deterministic conditional on drawn inputs; no posterior inference in this run. |
| Quarterly/annual sensitivity: IRR 17.6/18.2%, breach .08/.11, RMSE .28/.31, all CIs | L446–447 | NOT REPRODUCIBLE | Remove numerical sensitivity table; explicitly list untested interventions in retained subsection. |
| Cure sensitivities: IRR 17.6/18.3/18.1/18.0%, breach .08/.04/.05/.05, RMSE .28/.25/.26/.26, all CIs | L449–452 | NOT REPRODUCIBLE | No current cure intervention workflow; no replacement IRRs. |
| Hedge/floor sensitivities: IRR 16.8/17.6/17.9/17.7%, breach .12/.08/.06/.08, RMSE .32/.28/.25/.28, all CIs | L454–457 | NOT REPRODUCIBLE | No benchmark hedging/floor intervention. |
| Base vs optimized breach composition | F16 caption L469 | RECOMPUTED | Plot disjoint priority labels and overlapping flags; 17 payment defaults, 3 other reserve/covenant failures, 180 nonfailures. |
| Failure at volatility >12%, leverage >8x, leases >5x EBITDA | L480 | REFRAMED | No causal attribution or reproduced frequency. F15 plots actual score/outcome distributions and confusion matrix. |
| 20–30% excess conservatism; utilization 85–90%,70–80%,50–65% | L540–544; L681 | NOT REPRODUCIBLE | Remove percentages and tightness assertions; conditional theory does not establish empirical tightness. |
| Bayesian data-informed priors drive benchmark | L113,125,132,159,249–261,327–342 | REFRAMED | Keep conceptual hierarchical equations and separate calibration discussion; current sampler is clipped normal with regime adjustments, not the old mixture/LKJ experiment. |
| Benchmark tasks include optimal covenant design | L344–348 | REFRAMED | Current executable measures synthetic ranking, classification, path errors and timing; optimization is future work. |
| Error is analytic approximation error alone | L270–320; F12 and F14 | REFRAMED | Explicit four-part interpretation: approximation, structural-model, timing and input/convention differences. No estimated decomposition weights. |
| Deterministic safety of the full simulation | L282–320; appendices | REFRAMED | Conditional matched-path ratio theorem only, with valid denominator conditions and error budgets. Does not certify payment/reserve failures. |
| IFRS-16 effective in 2019, lease capitalization | L123,149 | SUPPORTED | Retained with IFRS Foundation primary source; caveat applicable exemptions. |
| LBO AUM over $3T | L121 | REMOVED | Unnecessary, unsourced market statistic; does not support this experiment. |
| Ratio effects 0.5–1.5 turns and 35–45% renegotiated agreements | L149–151 | NOT REPRODUCIBLE | No checked primary evidence from the cited entries; exclude quantitative literature claims. |

## Accor claim ledger

Every old Accor table cell is superseded by the current CSV; v2's input table and
year-by-year ratio table are generated, not hand-transcribed. Ratios use the same
`case_study_accor.py` and `covenants.py` as the tested repository workflow.

| Old claim | Source in old paper | Current support status | Action and current evidence |
|---|---|---|---|
| Real-world validation | L398,485 and case-study implications | REFRAMED | Illustrative real-company accounting-convention calculation; not predictive validation. |
| Accor 2018–2022 | L485,493,563 | RECOMPUTED | Current input dates 2019–2023. Archive full CSV inputs and generated outputs. |
| 5,000+ hotels, 110 countries | L489 | REMOVED | Not needed for stipulated calculation and not established by case-study data/script. |
| Revenue EUR2.6–4.1B, mean leases EUR2.8B | L489 | RECOMPUTED | Current inputs have revenue 1,621–5,056 and mean leases 2,980 (EUR millions); v2 prints per-year input table rather than company profile claims. |
| 2018: revenue4059, EBITDA687, margin16.9%, debt1823, leases2840 | L499 | REMOVED | No 2018 row in current case-study input. |
| 2019: 4048,692,17.1%,1456,2950 | L500 | RECOMPUTED | Current input: revenue4049, EBITDA825, net debt2500, leases3000, rent120, interest190; old derived margin omitted. |
| 2020: 2741,196,7.1%,2134,2845 | L501 | RECOMPUTED | Current input: 1621,-34,3350,3200,128,180; negative EBITDA is retained and ratios undefined. |
| 2021: 2616,273,10.4%,1987,2756 | L502 | RECOMPUTED | Current input: 2204,364,3100,3050,125,175. |
| 2022: 3510,598,17.0%,1534,2698 | L503 | RECOMPUTED | Current input: 4224,675,2800,2900,118,170. Current CSV additionally supplies 2023: 5056,1003,2550,2750,110,168. |
| ICR 10.6x → 2.6x, -8x | L113,519 | RECOMPUTED | Positive-EBITDA-year means: 4.764419 → 2.463733, difference -2.300686. Means exclude only undefined 2020 ratios. |
| Leverage 5.1x → 12.6x, +7.5x | L113,398,520,528 | RECOMPUTED | Four valid-year means: 4.559327 → 9.322716, difference +4.763389. |
| ICR threshold4.0, leverage threshold3.5 | L514,521 | SUPPORTED | Same constants in case-study script, always described as hypothetical; not actual Accor covenants. |
| Breaches 5/5 vs 2/5 | L521 | RECOMPUTED | 5/5 vs 3/5; includes explicit negative-EBITDA failure in both views in 2020. |
| Frozen GAAP stabilizes pandemic performance; optimization necessary | L522,526–531 | REFRAMED | Convention sensitivity under stipulated inputs only; no causal, actual-contractual or return-improvement claim. |
| All inputs directly validate public filings | L485; accorURD bibliography | REFRAMED | CSV contains reported/reconstructed inputs and source-page annotations. Those annotations are not independently authenticated in this revision. Require net-debt lease-exclusion convention for an economic bridge. |

## Mathematical audit (line-by-line logical steps)

| Old statement/step | Source | Status | Correction |
|---|---|---|---|
| Debt closed form without nonnegative-floor qualification | L269–272 | REFRAMED | Retain closed form only when floor never binds; present implemented projected recursion. |
| CPI lease recursion is the benchmark lease schedule | L203–210,273–275 | REFRAMED | Keep CPI schedule as a separately stipulated theoretical model; benchmark uses run-off additions/principal, with different bases across families. |
| Ratio formula requires only unperturbed denominator nonzero | L643–645 | REMOVED | False for finite negative denominator perturbations; supply exact identity, counterexample and common denominator floors. |
| Debt error derived from FCF level without a forcing-error premise | L650–654; L722–724 | REFRAMED | Prove recurrence stability under an explicit bound on FCF differences. No claim to have measured/established that budget. |
| Signed l'Hopital limit as written | L654 | REFRAMED | Use consistent difference quotient `(a^t-b^t)/(a-b)` and limit `t*a^(t-1)`. |
| CPI expression `1-(1+pi_min)^(-T)` is positive bound denominator | L656–659; L727–731 | REMOVED | It is negative at pi_min=-.02, T>0. Replace with finite payment perturbation via mean-value theorem and recurrence propagation. |
| EBITDA error includes capex drag as if changing EBITDA | L661–664; L734–736 | REFRAMED | Capex conversion enters FCF; use exact growth sensitivity bound with interval derivative supremum. |
| Leverage bound omits cash and absolute net debt | L292–293; L670–671 | REFRAMED | Add cash error, use absolute analytic numerator upper bound and common positive EBITDA floor. |
| Interest budget automatically covers mismatched rates/timing | L290–291,666–669 | REFRAMED | Require matched rates/timing or additional error budgets; no transfer to benchmark. |
| Numerical constants .05 FCF, .1 lease, .05 EBITDA suffice | Separate historical mathematical_appendix.tex, Proof Structure | NOT REPRODUCIBLE | Constants asserted without established budgets; preserve historical file only, do not include it in v2 build. |
| ICR expression missing numerator factor | Separate historical mathematical_appendix.tex, Explicit Error Bound Formulas | REMOVED | Correct exact ratio inequality includes numerator magnitude with denominator perturbation. |
| Expected IRR typically increases with leverage tolerance | L310–313 | REFRAMED | Feasible-set inclusion implies weakly nondecreasing supremum for fixed objective/mapping; does not change a fixed transaction's return. |
| Certification proof | L683–690 | REFRAMED | Valid triangle-inequality implication retained only with proved/assumed ratio budgets; covers tested covenant inequalities, not financial-failure union. |
| Assumptions growth[-.8,.5], CPI[-.02,.06], quarterly tests and hedge set establish guarantees | L695–712 | REFRAMED | Replace with explicit matched definitions, positive floors, finite perturbations and common recurrence coefficients. Historical ranges are not the current benchmark DGP. |
| Worst-case denominator schedules from listed corners | L738–739 | NOT REPRODUCIBLE | No proof those corners minimize both denominators; require independently established common positive floors. |
| Tightness, volatility conservatism and utilization | L680–681 | NOT REPRODUCIBLE | No retained empirical tightness statement. |

The revised appendix proves only the narrower algebraic claims described above.
It does not retrofit a stronger theorem onto the full financing simulation.

## Financial-correctness verification against code and tests

`src/lbo/full_simulation.py` was read alongside the full acceptance waterfall
tests and integration classification tests, not just the documentation.

| Required behavior | Implementation / regression evidence | Status |
|---|---|---|
| Opening cash funded as a use; sponsor funds purchase + fees + cash | `entry_sources_and_uses`; `test_opening_cash_is_funded_as_a_use` (0,40,100) | SUPPORTED |
| Amortisation draw not also credited to cash | `cash_after_mandatory = cash_before_financing - cash_funded_amortisation`; nine-case conservation test | SUPPORTED |
| Negative operating deficits retained | Negative-margin cases with ample and exhausted capacity in conservation test | SUPPORTED |
| Partial refinancing reduces unpaid principal | Actual amortisation = cash funded + amortisation draw; partial-payment and partial-refinancing cases | SUPPORTED |
| Sweep capped at remaining term debt | Three-way min in sweep; `test_cash_sweep_cannot_exceed_remaining_term_debt` | SUPPORTED |
| Unpaid principal remains outstanding | `test_unpaid_amortisation_remains_in_debt_after_default` over two years | SUPPORTED |
| Excess debt rejected | `test_overfunded_entry_is_rejected_instead_of_unbalanced` | SUPPORTED |
| Zero sponsor equity exact | `test_zero_sponsor_equity_is_not_replaced_with_epsilon` checks zero and undefined return ratios | SUPPORTED |
| Payment default joins financial failure union | `run_benchmark`; isolated `test_benchmark_counts_payment_default_without_other_failure` | SUPPORTED |
| Separate refinancing and liquidity draw fields | Conservation test checks components and sum; priority/carryforward regression | SUPPORTED |
| Reserve below mandatory principal; revolver repayment before sweep | Multi-year priority test and reserve shortfall cases | SUPPORTED |

## Figure-source ledger

Every v2 figure has exact generation code in
`analysis/scripts/generate_paper_v2_figures.py`, vector PDF and preview PNG output
in `paper/figures/v2/`, and a caption in the canonical manuscript. Source SHA and
seed are embedded in each synthetic figure; Accor is deterministic and uses no
random seed. The manifest checksums all 12 files and the relevant scripts/inputs.

| Existing section / filename | Old content | Current replacement | Current data source |
|---|---|---|---|
| F12_theoretical_guarantees | Claimed universal bounded errors | Signed leverage and ICR difference histograms; no bound line | Reconstructed `scenario_paths.csv`, 1,000 scenario-years |
| F14_method_comparison | Posterior frontiers and IRR comparisons | Analytic ROC vs simulation labels and actual discrepancy metrics | `benchmark_seed42.json` |
| F13_benchmark_overview | Regions/geographic mix | Actual regime and synthetic operator sampling composition | JSON scenario records; 105 base,55 downside,40 distressed |
| F16_breach_composition | Base/optimized package results | Disjoint priority labels plus overlapping flags | JSON counts; flags: payment17,reserve20,covenant19 |
| F15_failure_modes | Claimed bound conservatism/utilization | Score distributions and threshold confusion matrix | Scenario records; actual rows/predicted columns [[180,0],[17,3]] |
| accor_case_study | Obsolete convention values | Current year-by-year dual ratios, hypothetical thresholds and explicit 2020 gap | Existing case-study script/helper and `data/case_study/accor.csv` |

## Reproducibility and bibliography ledger

| Old claim | Old source | Status | Action |
|---|---|---|---|
| `v1.0-camera-ready` is current reproducibility target | L380,553 | REMOVED | No new tag. Exact source and benchmark SHAs generated into manifest/provenance block. |
| Git hash `9cedbf0` | L756 | RECOMPUTED | Baseline repaired SHA archived; final benchmark uses committed v2 source SHA. |
| Figure timestamp 2025-08-14 23:36 UTC | L757 | REMOVED | No historical timestamp used for new figures. Source/seed/checksums identify outputs. |
| Python3.11.5, NumPy1.24.3, PyMC5.7.2 | L758–760 | RECOMPUTED | Actual Python3.14.0 and pip freeze exported; benchmark does not require PyMC. |
| `sha256:a8f9b2c...` environment hash | L762 | RECOMPUTED | Full actual SHA-256 of environment.txt generated, no placeholder hash. |
| environment.yml ensures exact package versions | L380 | REFRAMED | Archive actual installed environment and explain editable path relocation. No lockfile or cross-platform bit identity claim. |
| Generator DOI/Zenodo8234567 and full release | L553,744 | REMOVED | No verified current deposit; no invented replacement DOI. |
| CC-BY-4.0 repository | L553 | REFRAMED | Repository LICENSE is MIT. |
| Old bibliography supports all contextual claims | Related Work and L560–639 | REFRAMED | Preserve entire historical bibliography verbatim. Revised bibliography retains checked, actually used references; remove unsupported literature effect sizes and unused/unverified entries from the active bibliography. |

Primary references checked for retained context: IFRS Foundation's IFRS 16 page
(https://www.ifrs.org/issued-standards/list-of-standards/ifrs-16-leases/),
Dichev and Skinner's publisher entry (https://doi.org/10.1111/1475-679X.00083),
and Nini, Smith and Sufi's publisher entry
(https://www.sciencedirect.com/science/article/pii/S0304405X09000233).
No population percentage or treatment effect is inferred from these references.
The Accor source annotations are explicitly not independently audited.

## Validation and exact-commit protocol

The clean baseline passed all 76 tests with 52.13% aggregate coverage and 100%
statement coverage of full_simulation.py. Ruff check and format check passed.
The full seed-42 baseline report is archived before manuscript changes; current
financial results agree with the repaired implementation's documented behavior.
CI's Python3.10–3.12 matrix is not a local result and must be reported separately.

Final source is committed before the publication generation pass. The next
artifact-only commit stores the benchmark, figures, generated table/provenance
inputs, PDF and validation records. A Git commit cannot contain its own hash;
the paper therefore names the exact source commit, and the artifact commit
provides the containing snapshot. This is an explicit provenance convention,
not a fabricated self-referential SHA. See `REPRODUCE_V2.md` and the generated
`results/paper_v2/validation.json` for final checks and PDF inspection evidence.

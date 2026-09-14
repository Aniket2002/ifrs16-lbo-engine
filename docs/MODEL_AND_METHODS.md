# Model and methods

This is the reader-facing specification of the financial model, synthetic
benchmark, and validation protocols. Executable code and frozen protocol files
remain authoritative where implementation-level detail matters.

## Full annual simulation

`src/lbo/full_simulation.py` models annual operating and financing cash flows
under explicit assumptions. It is not audited transaction data or a complete
accounting ledger. Year-one revenue equals the opening value and growth begins in
year two. Interest uses opening financial debt, revolver, and lease balances; new
revolver drawings incur interest from the following year.

EBITDA is measured before modeled lease interest and principal. Cash taxes are
floored at zero on EBITDA less assumed depreciation, financial interest, and
lease interest. The model has no loss carryforward or separate right-of-use asset
depreciation schedule. Lease additions are noncash and proportional to revenue;
lease principal reduces the opening liability.

## Closing sources and uses

`entry_enterprise_value` is a stipulated debt-free acquisition use. Opening
operating cash is also a funded use. The simplified closing is:

```text
uses = purchase price + transaction fees + opening operating cash
sources = acquisition debt + sponsor equity
```

The model does not infer seller cash, existing-debt refinancing, or an entry lease
bridge. Debt above total uses is rejected because excess-debt distributions are
not modeled. Zero sponsor equity is represented exactly, with undefined IRR and
MOIC.

## Financing waterfall

For each annual period, the model:

1. calculates cash before financing after operations, interest, taxes, capital
   expenditure, working capital, and lease principal;
2. pays mandatory amortisation from available cash and then revolver capacity;
3. records actual and unpaid amortisation separately;
4. draws remaining revolver capacity toward the minimum-cash reserve;
5. repays the revolver from excess cash, then sweeps term debt; and
6. retains cash left after debt repayment.

Mandatory debt service has priority over the cash reserve. Refinancing draws used
for amortisation pay debt directly; only liquidity draws increase retained cash.
Unpaid principal stays outstanding and sets the payment-default flag. Simulation
continues after default for diagnostics and does not model acceleration,
restructuring, default interest, creditor recovery, or a legal insolvency process.

The validator checks sources and uses, every cash/debt/revolver/lease roll-forward,
draw decomposition and capacity, amortisation, sweeps, reserve deficits, default
flags, ratio denominators, and finite outputs.

## Returns and covenant views

The sponsor cash-flow vector contains the entry equity outflow, zero interim
distributions, and end-of-hold exit equity. Exit equity equals EBITDA times the
exit multiple, less sale costs, financial debt, and lease liability, plus cash.
Negative exit residuals remain diagnostic in the engine. The separate financing
experiment applies a stated limited-liability transformation at the analysis
layer.

The benchmark's IFRS 16-style net leverage includes leases and subtracts cash;
coverage divides EBITDA by financial plus lease interest. Its stipulated failure
thresholds are 6.0 for leverage and 1.8 for interest coverage. The frozen-GAAP
helper excludes leases from net debt and uses the project's supplied rent-adjusted
coverage convention. These are modeled covenant conventions, not universal
accounting rules.

## Reduced-form screening model

`src/lbo/lbo_model_analytic.py` is structurally distinct from the annual
simulation. It uses reduced-form cash conversion and debt recursion and omits the
simulation's explicit tax, working-capital, mandatory-amortisation, revolver, and
payment-default waterfall. Lease additions, growth timing, interest timing, and
some opening assumptions also differ.

The screening score is:

```text
score = 1 / (1 + exp(2 * minimum_headroom))
```

It is a monotonic ranking transform, not a calibrated probability of default.
Cross-model error statistics combine structural, timing, and input differences;
they are not pure numerical approximation error under matched assumptions.

## Synthetic benchmark design

`data/synthetic/operators.csv` defines five stipulated borrower archetypes. The
benchmark samples 200 five-year scenarios with replacement using a fixed NumPy
seed. Base, downside, and distressed regimes have stipulated probabilities of
0.5, 0.3, and 0.2. Draws and regime adjustments affect operating growth, margins,
debt, leases, and cash according to `analysis/run_benchmark.py`.

A broad financial failure is the union of any nonpositive EBITDA, payment default,
reserve funding deficit, net leverage above 6.0, or interest coverage below 1.8
in any year. Separate subtype flags preserve overlapping conditions. Execution
exceptions are reporting failures, not financially distressed observations, and
abort the benchmark rather than being silently dropped.

The benchmark reports ranking and classification metrics, covenant-ratio errors,
failure counts, scenario records, assumption-based diagnostic envelopes, data
hashes, timing summaries, and source provenance. Timing depends on the machine
and load. The scenarios are synthetic and do not identify performance in a real
borrower population.

## Threshold-transfer protocol

The final v3 evaluation uses the unchanged archived score/label records and treats
each archetype as one group. For each fold, it chooses a threshold on the other
four groups by maximizing training balanced accuracy, with the frozen lowest-
threshold tie rule, and applies that threshold unchanged to the held-out group.
Every scenario appears in held-out evaluation exactly once. Test labels and scores
never enter candidate generation or selection.

The protocol reports fold dispersion and aggregate point estimates. Five
synthetic groups, sparse positives, and overlapping training sets do not justify
a precise confidence interval. The fixed score's pooled and concatenated
held-out AUC must be identical because the score is not retrained; that identity
does not independently establish generalization. See the
[frozen protocol](V3_TEMPLATE_EVALUATION_PROTOCOL.md).

## Separate financing-design experiment

The financing experiment uses the same five archetypes but a separate ex-ante
synthetic design, seed, operating paths, consistent EBITDA-based entry/exit
valuation convention, and frozen 112-policy grid. Policies vary opening debt,
scheduled amortisation, and cash sweep. Selection uses only four training
templates, subject to reference-relative broad-failure and payment-default
constraints, and is then evaluated on the held-out template.

Limited-liability proceeds and returns are applied in the analysis layer while raw
engine exit equity is preserved. A fixed-rate sensitivity reruns the same frozen
selection rule. The grid, objective, constraints, and tie rules cannot be changed
after observing held-out results. See the
[frozen financing protocol](V3_OPTIMIZATION_VALIDATION_PROTOCOL.md).

## Methodological boundaries

The model has no external borrower validation, endogenous credit pricing,
refinancing behavior, default recovery, lender/sponsor utility, interim sponsor
distributions, or comprehensive input-domain validation. The Bayesian experiment
uses unverified/stipulated calibration inputs and is excluded from substantive
results. The financing experiment is a methodological demonstration rather than
an economically substantive optimum. Passing mechanical validation does not
establish ranking performance, and ranking performance does not establish a
portable or useful decision threshold.

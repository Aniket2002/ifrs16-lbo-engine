# Model specification and conventions

## Full annual simulation

`src/lbo/full_simulation.py` is the canonical full simulation. It models annual
cash flows under explicit assumptions, not audited transaction data or a complete
accounting ledger. Revenue in year 1 equals `revenue_0`; growth starts in year 2.
Interest uses opening financial debt (including revolver) and opening lease
liability. New revolver drawings incur interest from the following year.

EBITDA is before modeled lease interest and principal. Cash taxes are floored at
zero on EBITDA less assumed depreciation, financial interest and lease interest.
There is no loss carryforward or separate right-of-use asset depreciation schedule.
Lease additions are noncash, proportional to revenue; principal is a proportion
of opening lease liability. Closing liability is opening plus additions less
principal. This is a stylized roll-forward, not a lease-by-lease valuation.

## Closing sources and uses

`initial_cash` means cash funded and retained for the post-close business.
`entry_enterprise_value` is the stipulated debt-free acquisition purchase use;
there is no inferred seller cash, existing-debt refinancing, or entry lease bridge.
Users must supply a purchase value consistent with that simplified convention.

- Uses = purchase price + transaction fees + opening operating cash.
- Sources = acquisition debt + sponsor equity. `cash_sources` is retained as a
  compatibility field and is always zero.
- Sponsor equity = total uses - acquisition debt.
- `opening_cash_use` identifies the retained cash; `total_sources = total_uses`.

Default uses are 1,070, debt is 450 and sponsor equity is 620. Increasing retained
cash by 1 increases sponsor funding by 1 and opening operating cash by 1.
Debt exceeding uses is rejected because excess-debt distributions are not modeled.
Zero sponsor equity is represented exactly; IRR and MOIC are undefined in that case.

## Financing waterfall

1. Cash before financing equals opening cash + operating cash generation - lease
   principal. Operating generation already deducts working-capital increases,
   cash taxes, financial interest, lease interest and capex.
2. Pay mandatory amortisation using positive available cash, then available
   revolver capacity. The reserve has lower priority than mandatory debt service.
3. Record actual payment and unpaid amortisation separately. Unpaid principal
   stays in term debt; the row records payment default. There is no automatic
   acceleration, arrears catch-up schedule, default interest or restructuring.
4. Draw remaining revolver capacity to restore minimum cash, including any
   negative operating cash deficit. Refinancing drawings do not add retained cash.
5. Repay the revolver from cash above the reserve, then sweep term debt, capped
   by the remaining term balance. Excess cash after all debt is repaid is retained.

`revolver_draw` is the sum of `revolver_draw_for_amortisation` and
`revolver_draw_for_liquidity`. Every row must satisfy:

```text
ending cash = opening cash + operating cash generation - lease principal
              - actual mandatory amortisation - cash sweep
              + total revolver draw - revolver repayment
ending term debt + ending revolver = opening term debt + opening revolver
              - actual mandatory amortisation - cash sweep
              + total revolver draw - revolver repayment
scheduled amortisation = actual mandatory amortisation + unpaid amortisation
```

Negative ending cash denotes an unfunded deficit, not an authorized overdraft.
`funding_deficit = max(0, min_cash - ending_cash)`; `insolvency_flag` means failure
to fund the modeled reserve, not a legal insolvency determination. Simulation
continues after default for diagnostics; post-default returns are not recovery estimates.
Assumptions should use nonnegative balances, limits and amortisation, and fraction
parameters in [0, 1]; comprehensive input-domain validation is not implemented.

## Returns and covenant views

The sponsor vector is `[-sponsor_equity, 0, ..., exit_equity]`, with end-of-year
exit proceeds and no interim distributions. Exit equity is EBITDA times the exit
multiple, less sale costs, term debt, revolver and lease liability, plus cash.
A negative exit residual is retained as a diagnostic; limited-liability payoffs
and creditor recoveries are not modeled.

The benchmark tests IFRS-16-style net leverage `(term debt + revolver + leases -
cash) / EBITDA` against 6.0 and EBITDA / (financial + lease interest) against 1.8.
These are hypothetical covenant thresholds, not universal accounting rules.

The separate `covenants.py` frozen-GAAP helper excludes leases from net debt,
uses the supplied EBITDA for leverage, and `(EBITDA + rent) / financial interest`
for coverage. This is the project's stipulated rent-adjusted coverage convention;
it is not an automatic conversion of IFRS-16 statements to historical GAAP.
Historical operating-lease EBITDA deducts rent; IFRS-16 EBITDA generally excludes
that operating rent charge. Users must align inputs with their covenant definition.

## Reduced-form screening differences

`lbo_model_analytic.py` uses `(alpha - kappa) * EBITDA` for cash conversion and
capitalizes a weighted debt rate in its debt recursion before subtracting a sweep.
It has no explicit taxes, working-capital schedule, mandatory amortisation,
revolver constraint or payment-default waterfall. Cash is set to a fixed minimum
after time zero without a financing reconciliation. It is a screening approximation.

Run-off lease additions are proportional to prior-period EBITDA in the analytic
model, versus current revenue in the full simulation. Analytic growth applies at
time 1, whereas full-simulation growth starts in year 2. Analytic coverage uses
closing debt and leases, versus opening balances for simulation cash interest.
The benchmark also retains distinct opening EBITDA, cash, minimum cash and rate
assumptions across these models. Its error metrics therefore reflect structural
and input differences, not solely approximation error under matched assumptions.
The legacy `lbo_model.py` API is separate from the benchmark full simulation.

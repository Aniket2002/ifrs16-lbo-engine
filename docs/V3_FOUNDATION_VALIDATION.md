# V3 foundation financial validation

This stage adds independent return/debt checks and an external runtime validator.
It does not change `src/lbo/full_simulation.py`, `analysis/run_benchmark.py`, the
reviewed v2 methodology or any economic assumption.

## Independent results

For sponsor cash flows `[-100, 0, 0, 0, 0, 200]`, the independently derived
annual IRR is `(200 / 100) ** (1 / 5) - 1 = 0.1486983549970351`. Production returns
the identical value within absolute tolerance `1e-12`. Independently derived and
production MOIC are both `200 / 100 = 2.0`.

The hand-derived fixed schedule has opening debt 100, 80, 60, 40 and 20; opening-
balance cash interest 10, 8, 6, 4 and 2; mandatory principal 20 each year; and
closing debt 80, 60, 40, 20 and 0. Production matches every value exactly, with
zero revolver and zero unpaid principal.

Negative exit equity is retained rather than floored. A constructed full-model
case returns exit equity -100, MOIC -2.0 and NaN IRR. A separate zero-exit case
returns MOIC 0 and NaN IRR; neither produces infinity, arbitrary zero IRR or an
uncaught numerical exception. The current sponsor vector has one entry flow,
zero interim flows and one exit flow, so multiple sign changes cannot arise
without expanding the economic model. No such expansion was made.

## Runtime invariants

`validate_simulation` checks every supplied year and fails immediately with a
`SimulationInvariantError` containing a JSON-serializable reproduction record.
It checks:

- total and component entry sources/uses;
- year-one starting balances and all later cash/debt/revolver/lease roll-forwards;
- decomposition, nonnegativity and capacity of revolver draws;
- cash before financing, cash after mandatory service and ending cash, counting
  amortisation refinancing as direct debt funding rather than retained cash;
- term debt, revolver and lease reconciliation and implemented nonnegative floors;
- scheduled, paid and unpaid amortisation and payment-default equivalence;
- reserve deficit and its exact positive-deficit flag convention;
- nonnegative sweeps bounded by remaining term debt, excess cash and sweep rate;
- revolver repayment bounded by outstanding revolver after draws;
- finite numeric simulation outputs; and
- explicit recording of zero EBITDA and interest denominators for downstream
  ratio handling.

The fixed seed-42 scenario generator produced 200 paths and 1,000 annual rows;
all passed. If a v3 run fails, `analysis/run_v3_foundation.py` serializes scenario
ID, year, invariant, assumptions, row, previous row, expected, actual, tolerance
and source commit below `results/v3/invariant_failures/` before re-raising.

## Adversarial evidence

Focused tests deliberately corrupt ending cash, closing term debt, revolver draws
above capacity, unpaid amortisation, sweep above remaining term debt, and year-two
opening cash. Each corruption raises the structured exception at the intended
year and preserves the supplied scenario ID. The year-two test also verifies the
previous row is preserved; every record successfully serializes to JSON.

The exact machine-readable independent schedules and results are in
`results/v3/foundation_validation.json`. Full-suite, Ruff and reviewed-v2
reproduction results are added after the source commit so the report identifies
the exact code under test. The small strictly-positive reserve-deficit behavior
observed during validation is the documented production convention, including
floating-point residuals; it did not require an engine change.

Template-held-out evaluation, Bayesian validation, optimization and manuscript
work remain outside this stage and are not claimed complete.

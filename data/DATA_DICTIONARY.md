# Data Dictionary

## data/synthetic/operators.csv

- `operator_id`: unique synthetic operator key; unit: none; source: synthetic design; status: simulated; range: string pattern; seed: 42.
- `operator_name`: human-readable synthetic name; unit: none; source: synthetic design; status: simulated; range: free text; seed: 42.
- `revenue_0`: starting-year revenue; unit: currency (same units across file); source: synthetic parameterization; status: simulated; range: > 0; seed: 42.
- `ebitda_0`: starting-year EBITDA; unit: currency; source: synthetic parameterization; status: simulated; range: > 0; seed: 42.
- `revenue_growth_mean`: mean annual revenue growth used for scenario draws; unit: ratio; source: synthetic design; status: simulated; range: -0.10 to 0.20; seed: 42.
- `revenue_growth_std`: growth volatility parameter for scenario draws; unit: ratio; source: synthetic design; status: simulated; range: 0.00 to 0.20; seed: 42.
- `ebitda_margin_mean`: expected EBITDA margin; unit: ratio; source: synthetic design; status: simulated; range: 0.12 to 0.40; seed: 42.
- `financial_debt_0`: opening financial debt; unit: currency; source: synthetic parameterization; status: simulated; range: >= 0; seed: 42.
- `lease_liability_0`: opening lease liability; unit: currency; source: synthetic parameterization; status: simulated; range: >= 0; seed: 42.
- `cash_0`: opening cash balance; unit: currency; source: synthetic parameterization; status: simulated; range: >= 0; seed: 42.
- `lambda_lease`: steady-state lease-to-EBITDA coefficient; unit: ratio multiple; source: model assumption; status: simulated; range: 1.0 to 6.0; seed: 42.
- `cash_sweep`: debt paydown share from positive free cash flow; unit: ratio; source: model assumption; status: simulated; range: 0.30 to 0.80; seed: 42.
- `lease_principal_rate`: principal run-off ratio on opening lease liability; unit: ratio; source: model assumption; status: simulated; range: 0.05 to 0.20; seed: 42.
- `lease_additions_rate`: new lease additions as a share of revenue; unit: ratio; source: model assumption; status: simulated; range: 0.00 to 0.05; seed: 42.
- `seed`: generation seed for row template; unit: integer; source: generation config; status: simulated; range: integer; seed: 42.

## data/synthetic/scenario_parameters.csv

- `parameter_name`: parameter identifier.
- `definition`: plain-language description.
- `unit`: expected unit.
- `parameter_source`: source category (synthetic_design or model_assumption).
- `reported_or_simulated`: provenance flag.
- `permitted_range`: valid range enforced in benchmark simulation.
- `seed`: generation seed.

## data/case_study/accor.csv

- `entity`: reporting entity; status: reported.
- `year`: fiscal year.
- `revenue`: reported revenue.
- `ebitda`: reported EBITDA.
- `net_debt`: reported net debt.
- `lease_liability`: reported lease liability where available/reconstructed.
- `interest_expense`: reported interest expense.
- `source_note`: source annotation for manual traceability.

# Data Dictionary

## operators_dataset.csv

- `name`: Synthetic operator identifier.
- `sector`: Sector category.
- `region`: Region category.
- `revenue_2019`: Baseline revenue (millions).
- `ebitda_2019`: Baseline EBITDA (millions).
- `capex_2019`: Baseline CapEx (millions).
- `revenue_growth_historical`: Growth assumption used for simulation.
- `projected_recovery_rate`: Recovery-speed assumption.
- `lease_liability_2020`: Lease liability in the synthetic scenario (millions).
- `lease_ebitda_multiple`: Lease/EBITDA multiple assumption.
- `avg_lease_rate`: Lease interest-rate assumption.
- `lease_maturity_profile`: Lease profile across year buckets.
- `senior_debt_2019`: Baseline senior debt (millions).
- `total_debt_2019`: Baseline total debt (millions).
- `interest_coverage_2019`: Baseline interest coverage ratio.
- `trading_multiple_2019`: EV/EBITDA anchor assumption.
- `credit_rating`: Optional rating label.
- `covenant_breach_2020_2021`: Simulated breach label.
- `actual_recovery_timeline`: Simulated recovery timeline (months).
- `source`: Provenance source descriptor.
- `year`: Reference year.
- `reported_or_simulated`: Provenance flag.
- `transformation`: Transformation/proxy method.

## benchmark_tasks.csv

- `name`: Task identifier.
- `description`: Task description.
- `target_variable`: Prediction target.
- `evaluation_metric`: Metric used for evaluation.
- `baseline_score`: Repository baseline score for reference.

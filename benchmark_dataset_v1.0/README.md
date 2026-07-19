# IFRS-16 LBO Synthetic Benchmark Dataset

## Overview

This dataset is a synthetic benchmark for method comparison in IFRS-16 LBO covenant analysis.

- It contains 5 anonymized operator archetypes.
- Financials and outcomes are simulated from documented assumptions.
- It is not a cleaned historical panel of company-reported financial statements.

## Files

- `operators_dataset.csv`: operator-level synthetic features and labels.
- `benchmark_tasks.csv`: standardized benchmark task definitions.
- `benchmark_metadata.json`: package metadata and provenance notes.
- `data_dictionary.md`: column-level definitions.
- `data_integrity.json`: SHA256 integrity hash for exported CSVs.

## Provenance

Each operator row includes provenance columns:
- `source`
- `year`
- `reported_or_simulated`
- `transformation`

## Tasks

1. Covenant breach prediction.
2. Headroom estimation.
3. Covenant design objective comparison.

## Baseline Scores

Baseline scores in this package are illustrative repository baselines for the synthetic setup and should not be interpreted as external leaderboard claims.

## License

CC-BY-4.0

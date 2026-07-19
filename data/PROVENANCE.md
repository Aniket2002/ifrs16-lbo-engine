# Data Provenance

## Synthetic IFRS 16 Covenant Benchmark

The synthetic benchmark data under data/synthetic are simulated artifacts for method evaluation.

- Generation type: synthetic templates and seeded scenario draws.
- Primary seed: 42.
- Intended use: benchmarking approximation and classification behavior.
- Not intended use: inference about real company population performance.

## Reported Case Study Data

The case-study file under data/case_study/accor.csv is kept separate from synthetic benchmark data.

- Entity: Accor SA.
- Nature: reported/reconstructed case-study inputs from public filings.
- Usage: illustrative walkthrough and sanity checks, not benchmark training labels.

## Separation Rule

Synthetic benchmark observations and reported case-study observations are never mixed in the same training/evaluation table.

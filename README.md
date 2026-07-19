# IFRS-16 LBO Engine

Practical research code for comparing covenant behavior under IFRS-16 and frozen-GAAP conventions.

## Scope

This repository contains:
- A simulation-oriented LBO workflow.
- A reduced-form analytic screening approximation.
- Synthetic benchmark utilities for method comparison.

Research framing: assumption-bounded analytic approximations evaluated against simulation.

## What Is Implemented

- IFRS-16 lease-liability mechanics in the model workflow.
- Dual-convention covenant reporting (IFRS-16 and frozen-GAAP views).
- Bayesian-style calibration scripts and sensitivity analysis utilities.
- Test modules for integration and acceptance checks.
- Two distinct model families:
	- Reduced-form analytical model for fast screening.
	- Full simulation model with explicit operating and financing line items.

## Benchmark Data Positioning

The benchmark package is synthetic and intended for method evaluation.

- Data package: `data/synthetic/`
- Core table: `data/synthetic/operators.csv`
- Data dictionary: `data/DATA_DICTIONARY.md`
- Provenance notes: `data/PROVENANCE.md`
- Separate reported case study: `data/case_study/accor.csv`

Use the benchmark as a transparent synthetic testbed, not as a cleaned panel of public-company financial statements.

## Quick Start

```bash
git clone https://github.com/Aniket2002/ifrs16-lbo-engine.git
cd ifrs16-lbo-engine
pip install -e .
```

Run a small reproducible case:

```bash
python analysis/scripts/case_study_accor.py
```

Run benchmark regeneration (single command):

```bash
python -m analysis.run_benchmark --seed 42
```

Run tests:

```bash
pytest -q
```

## Reproducibility

- Core assumptions are explicitly parameterized in the workflow code.
- Benchmark reporting is scripted in `analysis/run_benchmark.py`.
- Outputs include metrics, calibration curve, scenario/failure counts, speed benchmarks, and git SHA in `output/benchmark/benchmark_report.json`.

## Research Framing

Recommended wording for external summaries:

> Developed a reproducible framework for comparing IFRS-16 and frozen-GAAP covenant metrics using simulation, calibration, and assumption-bounded analytic screening approximations under explicit modeling assumptions.

## Citation

If you use this repository, cite the project and your exact commit hash. If referencing the associated paper, use the published SSRN citation details in your manuscript or report.

## License

MIT

# IFRS-16 LBO Engine

Practical research code for comparing covenant behavior under IFRS-16 and frozen-GAAP conventions.

## Scope

This repository contains:
- A simulation-oriented LBO workflow.
- A reduced-form analytic screening approximation.
- Synthetic benchmark utilities for method comparison.

This repository does not claim deterministic mathematical guarantees in its current implementation.

## What Is Implemented

- IFRS-16 lease-liability mechanics in the model workflow.
- Dual-convention covenant reporting (IFRS-16 and frozen-GAAP views).
- Bayesian-style calibration scripts and sensitivity analysis utilities.
- Test modules for integration and acceptance checks.

## Benchmark Data Positioning

The benchmark package is synthetic and intended for method evaluation.

- Data package: `benchmark_dataset_v1.0/`
- Core table: `benchmark_dataset_v1.0/operators_dataset.csv`
- Data dictionary: `benchmark_dataset_v1.0/data_dictionary.md`

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

Run tests:

```bash
pytest -q
```

## Reproducibility

- Core assumptions are explicitly parameterized in the workflow code.
- Benchmark generation is scripted in `analysis/scripts/benchmark_creation.py`.
- Integrity hashes are emitted for benchmark CSV artifacts.

## Research Framing

Recommended wording for external summaries:

> Developed a reproducible framework for comparing IFRS-16 and frozen-GAAP covenant metrics using simulation, calibration, and analytic screening approximations under explicit modeling assumptions.

## Citation

If you use this repository, cite the project and your exact commit hash. If referencing the associated paper, use the published SSRN citation details in your manuscript or report.

## License

MIT

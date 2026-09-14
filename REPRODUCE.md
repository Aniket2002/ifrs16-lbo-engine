# Reproduce the current workflow

From the repository root, create and activate a virtual environment (CI uses
Python 3.10, 3.11 and 3.12), then install the package and development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=src/lbo --cov-report=term-missing --cov-fail-under=52
python -m analysis.run_benchmark --smoke-test
python -m analysis.run_benchmark --seed 42
```

The benchmark writes `output/benchmark/benchmark_report.json`. The full command
overwrites the smoke report. See [the protocol](docs/experimental_design.md) for
metrics, classification and reproducibility limits. No calibration curve or Brier
score is generated. Preserve the exact code revision (including any local diff),
data checksums and `python -m pip freeze` with archived results.

The financial regression suite covers the full annual simulation, but aggregate
coverage remains about 52% because legacy simulation and analytic utilities have
substantial untested paths. The attempted 70% gate is not supported by the suite;
CI retains 52% without excluding code from measurement.

Run the separate Accor illustration with:

```powershell
python analysis/scripts/case_study_accor.py
```

It writes `analysis/figures/accor_case_study.png` and
`output/accor_case_study_results.csv`. The optional calibration CLI is a separate
workflow requiring `--input` and a pandas Parquet engine; it is not run by the
benchmark. Consult `python analysis/calibration/bayes_calibrate.py --help` before
using it. There is no current `output/manifest.json` or `setup.py` archive contract.
Older paper snapshots are not updated by these commands.

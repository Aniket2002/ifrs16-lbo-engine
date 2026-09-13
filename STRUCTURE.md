# Repository structure

- `src/lbo/full_simulation.py`: canonical full annual operating/financing waterfall,
  closing sources and uses, and sponsor returns.
- `src/lbo/lbo_model_analytic.py`: reduced-form screening and diagnostic utilities.
- `src/lbo/lbo_model.py`: separate legacy LBO API, not used by the benchmark.
- `src/lbo/covenants.py`, `data.py`, `analytic_bounds.py`: covenant views, loaders
  and assumption-based diagnostic envelopes.
- `analysis/run_benchmark.py`: current synthetic benchmark entry point.
- `analysis/scripts/`, `analysis/calibration/`: case study and separate calibration.
- `data/synthetic/`: current benchmark inputs; `data/case_study/`: separate case data.
- `tests/`: acceptance, regression, integration and utility tests.
- `docs/`: current model conventions and benchmark protocol.
- `analysis/paper/main.tex`: current reproducibility note. Other paper directories,
  `benchmark_dataset_v1.0/` and the bundled ZIP contain older snapshots.
- `output/`, `analysis/figures/`: ignored generated artifacts.
- `pyproject.toml`, `.github/workflows/ci.yml`: packaging and CI checks.

Use `README.md` and `REPRODUCE.md` for the supported commands. No `optimization/`
or `workflows/` package is present. Generated output and archive snapshots should
not be interpreted as automatically synchronized with local source edits.

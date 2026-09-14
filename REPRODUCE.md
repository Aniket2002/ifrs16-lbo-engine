# Reproduce the current repository

`main` is the authoritative current branch. The final paper is reproduced from
committed, frozen evidence; ordinary validation does not require rerunning the
archived stochastic experiments.

## Environment and checks

From the repository root, create and activate a virtual environment, then install
the package and development tools. CI supports Python 3.10, 3.11, and 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --no-cov
python analysis/validate_manuscript_freeze.py
```

## Final manuscript

The exact manuscript workflow, Tectonic version, visual-review requirement, and
artifact conventions are documented in [`paper/REPRODUCE_V3.md`](paper/REPRODUCE_V3.md).
The commands are:

```powershell
python -m analysis.scripts.render_manuscript_v3
python -m analysis.scripts.build_manuscript_v3
python -m analysis.scripts.audit_manuscript_v3
```

The renderer produces seven public tables and six figures from archived inputs.
The final build records its source commit in generated provenance, so a build from
a later commit need not be byte-identical to the archived publication PDF. A new
PDF is not considered visually approved until its own review record is completed.

## Optional benchmark regeneration

The older benchmark command remains available for controlled reproduction:

```powershell
python -m analysis.run_benchmark --smoke-test
python -m analysis.run_benchmark --seed 42
```

It writes `output/benchmark/benchmark_report.json`; the full run overwrites the
smoke report. Preserve the exact revision, local diff, data checksums, dependency
versions, and output if conducting a new run. This benchmark command is separate
from the frozen manuscript build and is unnecessary for validating the final PDF.

The Accor illustration remains available as:

```powershell
python analysis/scripts/case_study_accor.py
```

It writes ignored outputs under `analysis/figures/` and `output/`. The optional
calibration CLI is also separate from the benchmark and final manuscript.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete artifact
map and [`docs/MODEL_AND_METHODS.md`](docs/MODEL_AND_METHODS.md) for benchmark and
model conventions.

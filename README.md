# Ranking Is Not Threshold Portability

**Validating IFRS 16 Covenant Screening in a Synthetic Benchmark**

This repository contains the final validated research workflow, frozen evidence,
and publication manuscript for a synthetic study of IFRS 16 covenant screening.
Its central result is that mechanical correctness and strong pooled ranking do
not establish that one decision threshold will transfer across structurally
different borrower archetypes.

The authoritative current branch is `main`. The research and manuscript are
complete; no release, archival DOI, or external-data validation is claimed.

## Paper

- [Final paper (PDF)](paper/ifrs16_lbo_ssrn_v3.pdf)
- [LaTeX source](paper/ifrs16_lbo_ssrn_v3.tex)
- [Manuscript reproduction instructions](paper/REPRODUCE_V3.md)
- [Final manuscript build report](docs/V3_MANUSCRIPT_BUILD_REPORT.md)

## Reproduce and validate

Create a Python environment and install the package with development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the repository checks:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --no-cov
python analysis/validate_manuscript_freeze.py
```

Current source-integrity checks use canonical Git blobs and are cross-platform. The
original freeze manifest remains unchanged as historical provenance; see
[V3 integrity governance](docs/V3_INTEGRITY_GOVERNANCE.md).

Render and build the final manuscript from frozen evidence:

```powershell
python -m analysis.scripts.render_manuscript_v3
python -m analysis.scripts.build_manuscript_v3
python -m analysis.scripts.audit_manuscript_v3
```

The manuscript workflow reads archived evidence. It does not rerun a simulation,
fit, threshold search, bootstrap, optimization, or Bayesian recovery experiment.
See [reproducibility](docs/REPRODUCIBILITY.md) for the evidence map and toolchain
details.

## Research scope

The repository implements a full annual financing simulation and a distinct
reduced-form screening model. The benchmark contains 200 scenarios generated
from five stipulated synthetic archetypes. The score is a ranking score, not a
calibrated probability of default, and the results do not establish performance
on real borrowers.

Foundation tests support accounting and financing mechanics on the tested paths.
The fixed score ranks the frozen synthetic cases strongly in aggregate, while
training-only thresholds transfer unevenly across archetypes. The Bayesian
calibration experiment is excluded from substantive results. The separate
financing-design experiment is retained only as a methodological demonstration.

Source inputs remain under [`data/synthetic/`](data/synthetic/). Frozen validation
and manuscript evidence remains under [`results/v3/`](results/v3/).

## Documentation

| Topic | Document |
| --- | --- |
| Documentation index | [docs/README.md](docs/README.md) |
| Model and benchmark methods | [docs/MODEL_AND_METHODS.md](docs/MODEL_AND_METHODS.md) |
| Validation conclusions | [docs/VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md) |
| Research contribution and limits | [docs/RESEARCH_CONTRIBUTION.md](docs/RESEARCH_CONTRIBUTION.md) |
| Reproducibility | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| Repository history | [docs/REPOSITORY_HISTORY.md](docs/REPOSITORY_HISTORY.md) |

Detailed protocols, claim governance, source audits, and historical records are
indexed separately as internal provenance in the documentation index.

## Citation and license

Use the metadata in [`CITATION.cff`](CITATION.cff) and cite the exact commit used.
The code is available under the [MIT License](LICENSE).

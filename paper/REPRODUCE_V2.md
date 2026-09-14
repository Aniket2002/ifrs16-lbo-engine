# Reproduce the revised SSRN manuscript

The canonical source is `paper/ifrs16_lbo_ssrn_v2.tex`. The exact source and
benchmark commits are recorded in `results/paper_v2/manifest.json`. The revision
branches from repaired remote head `3903eccbf4650e84691aac9834caacdda0c7f5e5`,
including waterfall repair `5446bff`. Model and benchmark code are unchanged.

Use Python with the project installed (`python -m pip install -e ".[dev]"`).
The measured environment is archived as `results/paper_v2/environment.txt` with
an actually computed SHA-256. The editable project entry identifies its Git
revision; install the checked-out repository in your own location.
PyMuPDF is used only for PDF verification; it is not a model dependency.

From the repository root:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=src/lbo --cov-report=term-missing --cov-fail-under=52
python -m analysis.run_benchmark --seed 42
python -m analysis.scripts.generate_paper_v2_figures --run-benchmark
latexmk -pdf -outdir=paper paper/ifrs16_lbo_ssrn_v2.tex
```

The last Python command intentionally runs a fresh full benchmark, archives it,
reconstructs and verifies the paths, runs the existing Accor case study and writes
all six figure pairs (PDF and PNG), numerical table inputs and provenance.
Omit `--run-benchmark` to plot the archived report; this still verifies that its
scores and extremal ratios reproduce from current code and inputs. Both ordinary
benchmark modes overwrite `output/benchmark/benchmark_report.json`; the paper
archive is separate. Run tests before the final benchmark because smoke tests
also write to the ordinary benchmark output directory.

The baseline report `benchmark_baseline_seed42.json` preserves the clean repaired
head. The final report preserves the committed paper source revision. Financial
results must agree; timings need not agree.

If latexmk is unavailable, run pdflatex three times with
`pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/ifrs16_lbo_ssrn_v2.tex`.
This manuscript uses a manual bibliography, so no BibTeX pass is required.
On the revision workstation neither latexmk nor pdflatex was installed. The
build used the official portable Tectonic 0.17.0 Windows MSVC release:

```powershell
tectonic --keep-logs paper/ifrs16_lbo_ssrn_v2.tex
```

Tectonic downloads its TeX support bundle and automatically repeats passes.
See https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0.
It was stored outside the repository in the Windows temporary directory.
The PDF must have no undefined references, missing citations or missing graphics.
Review all pages, especially the intentionally missing 2020 Accor ratios.

```powershell
python -m pip install pymupdf
python -m analysis.scripts.verify_paper_v2
```

This verifies hashes, exact baseline financial agreement, cash reconciliation,
LaTeX diagnostics and figure presence, then renders every page and contact sheets
to `output/paper_v2_review/` for manual inspection. It writes
`results/paper_v2/pdf_verification.json`; successful rendering alone is not a
substitute for visual inspection. It expects the retained `.log` from the build.

Provenance uses a source commit followed by an artifact-only commit. The source
commit contains the final `.tex`, audit and generator. After committing those,
run the generator and build at that exact SHA, then commit only generated outputs
and validation records. This avoids claiming that a Git commit can contain its
own hash. No release tag is created. Re-running at another clean commit records
that actual commit, not a fabricated original identity. PDF bytes can also vary
with TeX metadata and environment; source and financial reproducibility are
distinct from bit-for-bit PDF reproducibility.

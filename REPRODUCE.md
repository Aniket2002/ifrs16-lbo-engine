REPRODUCE
=========

This file lists the minimal, exact steps to reproduce the core results and to prepare the archive for Zenodo.

Environment (Windows PowerShell)
--------------------------------
1. Create and activate a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package (editable) and core dependencies:

```powershell
pip install -e .
pip install -r requirements.txt
```

Reproduce key outputs
---------------------
1. Run the Accor case study (small, fast):

```powershell
python analysis/scripts/case_study_accor.py
```

Expected: prints summary to stdout and writes `analysis/figures/accor_case_study.png` and `output/accor_case_study_results.csv`.

2. Run the Bayesian calibration pipeline (may take longer):

```powershell
python analysis/calibration/bayes_calibrate.py
```

Expected: produces posterior samples and diagnostic figures under `analysis/figures/` or `output/` as described in the manifest.

3. Run tests (quick sanity check):

```powershell
pytest -q
```

Expected: all tests pass (green).

Optional: Build paper figures and manuscript
------------------------------------------
If you have LaTeX installed and want to regenerate paper figures and build the manuscript:

```powershell
make figures
# then (optional)
make paper
```

Docker (optional)
-----------------
Build and run (single command example):

```powershell
docker build -t ifrs16-lbo .
docker run --rm ifrs16-lbo python analysis/scripts/case_study_accor.py
```

Archive for Zenodo
------------------
The repository ZIP prepared for Zenodo contains: `src/lbo`, `analysis/paper`, `analysis/scripts`, `analysis/calibration`, `data`, `benchmark_dataset_v1.0`, `tests`, `README.md`, `LICENSE`, `CITATION.cff`, `requirements.txt`, `setup.py`, `STRUCTURE.md`, and `output/manifest.json`.

Notes
-----
- Choose either `requirements.txt` or `environment.yml` for environment reproducibility (this project keeps `requirements.txt`).
- If large raw datasets are hosted externally, include checksums and a download script rather than the raw data.
- Create a GitHub release/tag before uploading to Zenodo so Zenodo can link to the exact commit.

Contact
-------
Aniket Bhardwaj — bhardwaj.aniket2002@gmail.com

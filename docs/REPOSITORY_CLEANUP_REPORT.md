# Repository cleanup report

## Scope and outcome

- Starting canonical HEAD: `8e69d4911d352c6572c2a083e99e216fa5179644`
- Cleanup branch: `v3/repository-cleanup`
- Cleanup commit: `10d387f608c8afd24b08a59a1e41f844012724bc`
- Report commit and final branch HEAD: the commit containing this report; its SHA
  is recorded in the final handoff because a commit cannot contain its own SHA.
- Inventory examined: 55,822 working-tree files in 6,578 directories, including
  all 246 tracked files and every ignored or untracked path outside `.git`.

The cleanup removed 11 tracked files and 55,576 ignored or untracked local files,
for 55,587 deleted working-tree files in total. It eliminated 6,543 directories.
Two cleanup documents were added, leaving 237 tracked files.

| Metric | Before | After |
| --- | ---: | ---: |
| Tracked files | 246 | 237 |
| Working-tree files | 55,822 | 237 |
| Working-tree directories, excluding `.git` | 6,578 | 35 |
| Working-tree size, excluding `.git` | 1,607,445,643 bytes (1,532.98 MiB) | 5,392,716 bytes (5.14 MiB) |

## Files removed

Tracked deletions:

- Redundant v1 binary: `ifrs16-lbo-engine-v1.0.zip`.
- Stale generated inventory: `folder_tree.txt`.
- Dead zero-byte root placeholders: `generate_arxiv_figures.py`,
  `validate_academic_setup.py`, `validate_setup.py`, and `setup_venv.bat`.
- Unused zero-byte manuscript placeholders:
  `arxiv_submission/main_clean.tex`,
  `arxiv_submission/mathematical_appendix.tex`,
  `paper/mathematical_appendix.tex`,
  `paper/mathematical_appendix_new.tex`, and
  `paper/theoretical_assumptions_new.tex`.

Ignored or untracked cleanup removed Python and Ruff/pytest caches, two local
virtual environments, package-install metadata, ordinary LaTeX intermediates,
local benchmark/case-study results, local legacy figure renders, PDF page/contact
sheet previews, and empty runtime directories. The largest removed items were:

| Path/category | Bytes removed |
| --- | ---: |
| `.venv-bayes/` | 878,990,067 |
| `.venv/` | 711,936,045 |
| `output/paper_v2_review/` | 4,858,837 |
| `output/paper_v3_review/` | 3,677,096 |
| `ifrs16-lbo-engine-v1.0.zip` | 620,621 |
| `analysis/paper/figures/` | 588,301 |

The 51 local figure and preview files under `analysis/figures/`,
`analysis/paper/figures/`, and `output/paper_*_review/` were reproducible render
copies. No byte-identical tracked duplicate was removed. The sole tracked hash
match found during inventory is an intentional two-stage verification record and
was preserved.

## Legacy material retained

The corrected v2 manuscript source and PDF, its twelve reviewed figure files,
reproduction instructions, generated tables, and 14 archived result files remain.
They are linked from the README and support the documented corrected-manuscript
lineage. `paper/historical/`, `paper/REVISION_AUDIT.md`, the older implementation
note under `analysis/paper/`, and `benchmark_dataset_v1.0/` also remain because
they explain superseded claims or retain unique historical source.

The only substantive v1 artifact deleted was the 620,621-byte bundled ZIP. It had
no exact path consumer, duplicated an old repository snapshot, and is recoverable
from commit `b7daa3b` and later Git history. References that described a bundled
ZIP in `README.md`, `REPRODUCE.md`, and `STRUCTURE.md` were corrected. The five
other legacy-path deletions were zero-byte manuscript placeholders; the historical
provenance note already documented that they were never used as sources.

No tracked old manuscript PDF, tracked legacy figure, tracked result, substantive
legacy source, or provenance document was deleted.

## Material considered and preserved

The following potentially stale or redundant material was intentionally retained:

- `output/manifest.json`, because its historical role is unclear and it has unique
  content.
- All older non-v3 result trees, because they support corrected v2 reproduction
  and validation lineage.
- `FIXES_SUMMARY.md`, `analysis/calibration/`, `analysis/paper/`,
  `benchmark_dataset_v1.0/`, and legacy documentation, because each is documented,
  unique, or could support historical reconstruction.
- Both committed PDF and PNG forms of v2/v3 figures, because LaTeX, repository
  previews, rendering tests, manifests, or visual-review workflows use the
  separate forms.
- `results/paper_v2/pdf_verification.json` and
  `results/v3/baseline/reviewed_artifact_verification.json`, despite byte identity,
  because their paths record distinct stages of the evidence chain.
- Every non-empty script. Only four empty, unreferenced script placeholders met
  the dead-script standard.

## `.gitignore` changes

The existing Python, environment, packaging, coverage, Ruff/pytest, output,
figure, IDE, and common LaTeX rules were retained. New rules cover `.toc`, `.bbl`,
and `.blg` manuscript intermediates; `.DS_Store`, `Thumbs.db`, and `desktop.ini`;
and Vim swap/backup files (`*.swp`, `*.swo`, `*~`). Existing exceptions continue
to allow intentional final v2/v3 PDFs and figures.

## Verification

| Check | Result |
| --- | --- |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed; 41 Python files already formatted |
| `python -m pytest -q --no-cov` | Passed; 175 tests |
| `python analysis/validate_manuscript_freeze.py` | Passed; 100 claims, 34 quantitative claims, 50 prohibited-claim families, and 66 frozen source hashes checked |
| `python -m analysis.scripts.render_manuscript_v3` | Passed; seven tables and six figures rendered from frozen inputs |
| `python -m analysis.scripts.build_manuscript_v3` | Passed with Tectonic 0.17.0; 16 pages, no serious warnings, bibliography resolved, build source clean |
| `python -m analysis.scripts.audit_manuscript_v3` | Passed all claim, prohibited-text, hash, exhibit-count, bibliography, layout, visual-review, and committed-source checks |

The rebuild temporarily changed only provenance-bearing generated files because
the PDF records the current source commit. Those generated files were restored to
their validated versions after build success, and the final audit was rerun on
the archived final artifact. A path-restricted diff from the starting canonical
HEAD is empty for `results/v3/`, the v3 manuscript source/PDF/instructions,
`paper/figures/v3/`, `paper/generated/v3/`, and `data/synthetic/`.

## Why these deletions are safe

Every tracked deletion was searched by filename and path across source, tests,
build files, manifests, TeX, and documentation. The empty placeholders contained
no behavior or evidence. The stale tree listing was reproducible and unused. The
v1 ZIP was a redundant Git-recoverable snapshot, while the source and audit
material needed to understand v1-to-v2 correction remains. Local caches,
environments, previews, and compiler outputs are recreated by retained commands
and were never part of the committed evidence chain.

No model, simulation, threshold, score, input value, manuscript conclusion,
claim-ledger entry, research result, source data, or validated v3 evidence was
altered. The cleanup branch remains unmerged.

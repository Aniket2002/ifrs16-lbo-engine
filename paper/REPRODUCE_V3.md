# Reproduce the v3 manuscript from frozen evidence

The versioned source is `paper/ifrs16_lbo_ssrn_v3.tex`. The reviewed v2 manuscript
and all frozen numerical and claim artifacts remain unchanged. This workflow
renders archived data; it does not execute a simulation, fit, threshold search,
bootstrap or recovery experiment.

Use the project Python dependencies plus PyMuPDF for PDF inspection. The recorded
toolchain uses Tectonic 0.17.0, as documented for v2. Put `tectonic` on PATH or use
the build script's `--tectonic PATH` option. On Windows, it also checks the temporary
`ifrs16-paper-tools/tectonic.exe` location used for the prior build. The official
Windows archive is
`https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip`;
its release-asset SHA-256 is
`f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`.
Tool installation is outside the repository. The Windows build creates a temporary
fontconfig configuration pointing to the system font directory.

From the repository root:

```powershell
python -m analysis.scripts.render_manuscript_v3
python -m analysis.scripts.build_manuscript_v3
python -m analysis.scripts.audit_manuscript_v3
```

Equivalent Make targets are `render-v3`, `paper-v3`, and `audit-v3`. Use
`python -m analysis.scripts.build_manuscript_v3 --draft` for pre-commit layout
inspection. A final build requires committed manuscript/render/audit source and
records that source SHA in a generated provenance fragment. The PDF and build
audit are archived in a subsequent artifact commit; no self-referential commit
hash is claimed. `SOURCE_DATE_EPOCH` uses the source commit time. The renderer
removes variable PDF creation/modification dates; repeated rendering in the same
environment is tested byte-for-byte. Cross-toolchain PDF byte identity is not
assumed.

The source claim comments remain invisible in the PDF. The render manifest records
source selectors, numeric cells and render-time transformations. The source audit
checks numerical tokens against referenced frozen values/display rounding and
requires denominator context for observed percentages. These checks supplement
sentence-level review; numeric coincidence alone is not semantic traceability.
Symbolic equation subtraction operands are distinguished from signed return values.

The final audit also requires `results/v3/manuscript/visual_review.json` to identify
the inspected PDF SHA-256 and have status `passed`. Do not copy this status onto a
different PDF without inspecting it. Before archiving a new build, render and
inspect every page, check all exhibit-heavy pages, and update that record with the
actual reviewed hash. Intermediate build success is not final visual approval.

Frozen planning IDs remain in source labels. Publication numbers are Tables 1--4
for T1--T4, then Tables A1--A3 for T5/A1/A2. Main figures are 1--3; appendix
figures are numbered in their manuscript order: A-F2, A-F3, A-F1 become A1, A2, A3.
All seven public tables and six figures come from the frozen plans. The focused
optimization coverage formerly shown as Table A4 remains in repository validation
metadata and the build/audit report.

Lightweight checks:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --no-cov tests/test_manuscript_freeze.py tests/test_manuscript_rendering.py
python analysis/validate_manuscript_freeze.py
```

The build and audit scripts do not call the old v2 generator or any model runner.
Do not use the legacy `all` or `figures` targets for this frozen manuscript stage.

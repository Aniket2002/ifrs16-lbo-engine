# V3 manuscript build report

The manuscript and frozen exhibits are complete. The final PDF builds without unresolved LaTeX warnings, and the final manuscript audit passes. No numerical experiment, model change, admission change, release, tag, DOI or submission was performed.

- Branch: `v3/validated-rebuild`.
- Manuscript/source commit: `4b39bfd8b512d9ed0f84f2073d5ecf5ea184e4ea`.
- Claim-freeze commit: `21d726606f3208bcee6e3e84ae4c357e3a33d79b`.
- Title: **Ranking Is Not Threshold Portability: Validating IFRS-16 Covenant Screening in a Synthetic Benchmark**.
- Author: Aniket Bhardwaj.
- Source: `paper/ifrs16_lbo_ssrn_v3.tex`.
- PDF: `paper/ifrs16_lbo_ssrn_v3.pdf`.
- Length: 14 pages; abstract 162 words; main text 2099 words.
- Word-count policy: TeX prose after macro expansion; excludes abstract from main count, floats, table fragments, displayed equations, bibliography and appendices; includes headings and inline numeric tokens.
- Exhibits: eight tables, six figures; three appendix sections.
- Bibliography: two existing repository references, both cited and resolved. No literature expansion or fabricated citation.

## Claims and scope

Primary claim IDs used: SCR-001, SCR-002, SCR-004, SCR-006, SCR-007, SCR-009, SCR-011, SCR-012, SCR-013.

Secondary claim IDs used: HET-001, HET-002, HET-009.

All quantitative paragraphs have claim annotations; their numeric tokens trace to the linked frozen values or permitted display transformations. 280 generated numeric cells were checked against exact source selectors. The render manifest supplies the table/figure plans, sources, transformations and exhibit hashes. Numeric matching supplements manual sentence-level whitelist review; it does not alone establish semantic support. Both source and rendered-PDF prohibited phrase scans pass, and all 50 prohibited families were reviewed without importing historical headline numbers.

Observed subgroup percentages retain counts/n or adjacent sample sizes. Stipulated policy parameters and the deterministic limited-liability floor are distinguished from event-frequency estimates. No new confidence intervals, smoothing, bootstrap, significance test or fitted model was introduced.

Bayes remains C: excluded from main results, with a concise exclusion appendix and no posterior-driven LBO integration. Financing remains B: a methodological appendix only; its numerical results are absent from the abstract and conclusion.

The Template 004 synthesis uses: ?Template 004 shows weak threshold transfer and comparatively weak financing-policy outcomes, providing evidence within the same synthetic system consistent with structural heterogeneity affecting downstream transfer.? Adjacent text explicitly excludes independent replication, external validation, causal evidence and empirical borrower evidence, and identifies small regime-conditioned cells.

The limited-liability note states: ?Template 004 reference distressed P10 annualized sponsor return is -100% (n=21). Seven of the 21 scenarios have zero recovery, so the 10th percentile lies on that zero-recovery mass.? It explicitly distinguishes sponsor proceeds from raw equity, an unconstrained IRR, and solver failure.

## Exhibit numbering and inspection

Frozen planning IDs map to publication numbering as follows:

| Planning ID | Publication number | Page |
|---|---|---:|
| F1 | Figure 1 | 2 |
| T1 | Table 1 | 3 |
| T2 | Table 2 | 4 |
| F2 | Figure 2 | 5 |
| T3 | Table 3 | 5 |
| T4 | Table 4 | 6 |
| F3 | Figure 3 | 7 |
| T5 | Table A1 | 9 |
| A-F2 | Figure A1 | 10 |
| A1 | Table A2 | 11 |
| A2 | Table A3 | 12 |
| A-F3 | Figure A2 | 13 |
| A-F1 | Figure A3 | 13 |
| A3 | Table A4 | 14 |

Every page was visually inspected, including all exhibit-heavy pages. The final PDF SHA-256 and per-page preview hashes are in `results/v3/manuscript/visual_review.json`. No missing glyphs, clipped exhibits, placeholders, visible claim IDs, broken references, duplicate pages or blank pages remain. Appendix hyperlinks have distinct anchors. Reproduction commands and the bibliography render correctly.

## Verification and provenance

```json
{
  "ruff_check": "passed",
  "ruff_format_check": "passed (44 files)",
  "tests": {
    "command": "python -m pytest -q --no-cov tests/test_manuscript_freeze.py tests/test_manuscript_rendering.py",
    "passed": 22,
    "freeze_tests": 13,
    "rendering_tests": 9
  },
  "freeze_validator": "passed: 100 claims; 66 frozen source hashes",
  "final_manuscript_audit": "passed",
  "protected_diff_from_freeze_head": "",
  "experiments_run": false,
  "frozen_results_changed": false,
  "v2_preserved": true,
  "release_or_tag_created": false
}
```

Build commands and tool installation details are in `paper/REPRODUCE_V3.md`. Tectonic 0.17.0 built the final PDF from committed source. The build record archives the executable hash, source SHA and timestamp convention. Repeated exhibit rendering is byte-identical in the focused test environment. The final audit and build outputs are committed separately from the source so that the PDF records a real source commit, not a self-referential artifact SHA. The containing build/audit commit is discoverable from Git history.

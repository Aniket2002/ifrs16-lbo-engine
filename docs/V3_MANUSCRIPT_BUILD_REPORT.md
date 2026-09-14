# V3 manuscript build report

The publication-polished manuscript and frozen exhibits are complete. The final PDF builds without unresolved LaTeX warnings, and the manuscript audit passes. No numerical experiment, model change, threshold change, simulation, optimization, Bayesian result, admission change, release, tag, DOI or submission was performed.

- Branch: `v3/validated-rebuild`.
- Manuscript/source commit: `52401f00292a9feee762075dcfcc1f92b45be231`.
- Claim-freeze commit: `21d726606f3208bcee6e3e84ae4c357e3a33d79b`.
- Title: **Ranking Is Not Threshold Portability: Validating IFRS 16 Covenant Screening in a Synthetic Benchmark**.
- Author: Aniket Bhardwaj.
- Source: `paper/ifrs16_lbo_ssrn_v3.tex`.
- PDF: `paper/ifrs16_lbo_ssrn_v3.pdf`.
- Length: 16 pages; abstract 179 words; main text 2,713 words.
- Literature context: 616 words.
- Word-count policy: TeX prose after macro expansion; excludes abstract from main count, floats, table fragments, displayed equations, bibliography and appendices; includes headings and inline numeric tokens.
- Exhibits: seven public tables, six figures and three appendix sections.
- Bibliography: nine cited and resolved entries.

## Publication-polish record

Reader-facing references to the accounting standard now use **IFRS 16**, and the title-page label “V3 manuscript” has been replaced by “September 2026.” Internal B/C admission labels were removed from manuscript prose, captions and tables while their substantive treatment and internal audit metadata remain unchanged. The financing-design experiment remains a methodological demonstration only; the Bayesian calibration experiment remains excluded from substantive results.

The abstract was rewritten within the requested range while preserving every frozen headline value. It now foregrounds Template 004: 37 of 38 false positives and balanced accuracy of 0.513 under the transferred threshold versus 0.636 under the fixed 0.5 comparator. Table 4 now states: “Template 003 contains one positive observation; AUC = 1.000 therefore means only that this single positive ranked above the 46 non-failures in the frozen realization and should not be interpreted as a precise estimate of discrimination.”

The new literature section positions the three-layer framework using seven added verified sources: Dichev and Skinner (2002), Christensen and Nikolaev (2012), Fawcett (2006), Hand (2009), Steyerberg et al. (2010), Vickers and Elkin (2006), and Quiñonero-Candela et al. (2008). The existing IFRS Foundation (2016) and Nini, Smith and Sufi (2009) references remain. Metadata and claims were checked against IFRS Foundation, journal publisher, PubMed/PMC and MIT Press records.

A visible code and data availability statement points to `https://github.com/Aniket2002/ifrs16-lbo-engine`, the canonical branch and the exact manuscript-source lineage. It makes no archival-permanence claim and creates no DOI.

Former Table A4 was removed from the public PDF because focused code coverage is repository validation metadata. The frozen coverage artifact remains unchanged, and the render manifest and final audit preserve its exact public-report values:

| Module | Statement coverage | Branch coverage |
|---|---:|---:|
| Search kernel | 88.68% | 70.0% |
| Runner | 59.89% | 55.0% |

All numerical conclusions are unchanged.

## Claims and scope

Primary claim IDs used: SCR-001, SCR-002, SCR-004, SCR-006, SCR-007, SCR-009, SCR-011, SCR-012, SCR-013.

Secondary claim IDs used: HET-001, HET-002, HET-009.

All quantitative paragraphs have claim annotations; their numeric tokens trace to linked frozen values or permitted display transformations. The audit checked 268 generated numeric cells against exact source selectors. The render manifest supplies the table/figure plans, sources, transformations and exhibit hashes. Numeric matching supplements manual sentence-level whitelist review; it does not alone establish semantic support. Both source and rendered-PDF prohibited-phrase scans pass, and all 50 prohibited families were reviewed without importing historical headline numbers.

Observed subgroup percentages retain counts/n or adjacent sample sizes. Stipulated policy parameters and the deterministic limited-liability floor are distinguished from event-frequency estimates. No new confidence interval, smoothing, bootstrap, significance test or fitted model was introduced.

Bayes remains C in internal audit metadata: excluded from substantive results, with a concise exclusion appendix and no posterior-driven LBO integration. Financing remains B in internal audit metadata: a methodological appendix only, with its numerical results absent from the abstract and conclusion.

## Exhibit numbering and inspection

Frozen planning IDs map to publication numbering as follows:

| Planning ID | Publication number | Page |
|---|---|---:|
| F1 | Figure 1 | 2 |
| T1 | Table 1 | 4 |
| T2 | Table 2 | 5 |
| F2 | Figure 2 | 6 |
| T3 | Table 3 | 6 |
| T4 | Table 4 | 7 |
| F3 | Figure 3 | 8 |
| T5 | Table A1 | 10 |
| A-F2 | Figure A1 | 11 |
| A1 | Table A2 | 12 |
| A2 | Table A3 | 13 |
| A-F3 | Figure A2 | 14 |
| A-F1 | Figure A3 | 14 |

Every page was visually inspected in four contact sheets generated from the final PDF. The review included the title page, revised abstract, literature section, Table 4 note, all exhibit-heavy pages, appendix transitions, code statement and bibliography. No missing glyph, clipped exhibit, placeholder, visible claim ID, broken reference, duplicate page or blank page remains. The final PDF SHA-256 is `f113a3ba6d057c02f51fd6347131230bbe802139cd25c8a09c7a277cc57a0bc9`; per-page preview hashes are in `results/v3/manuscript/visual_review.json`.

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
  "prohibited_scan": "source and rendered PDF passed",
  "bibliography": "nine cited entries; no placeholders or unresolved citations",
  "protected_diff_from_freeze_head": "",
  "experiments_run": false,
  "frozen_results_changed": false,
  "v2_preserved": true,
  "release_or_tag_created": false
}
```

Build commands and tool installation details are in `paper/REPRODUCE_V3.md`. Tectonic 0.17.0 built the final PDF from committed source. The build record archives the executable hash, source SHA and timestamp convention. Repeated exhibit rendering is byte-identical in the focused test environment. The final audit and build outputs are committed separately from the source so that the PDF records a real source commit, not a self-referential artifact SHA. The containing build/audit commit is discoverable from Git history.

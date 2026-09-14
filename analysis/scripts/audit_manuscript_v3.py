"""Audit source annotations, frozen values and rendered PDF without model execution."""

import hashlib
import json
import math
import re
import unicodedata

import pymupdf

from analysis.scripts.render_manuscript_v3 import FREEZE, ROOT, count_rate, display, load
from analysis.validate_manuscript_freeze import resolve, validate

SOURCE = ROOT / "paper/ifrs16_lbo_ssrn_v3.tex"
PDF = SOURCE.with_suffix(".pdf")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
DANGEROUS = [
    r"calibrated\s+PD",
    r"real[- ]world\s+validation",
    r"empirical\s+validation",
    r"robustly\s+generalizes",
    r"optimal\s+capital\s+structure",
    r"\+3\.4",
    r"46\s*%\s*RMSE",
    r"0\.28\s*(?:vs\.?|versus)\s*0\.52",
    r"AUC\s*0\.76\b",
    r"calibrated\s+probability\s+of\s+default",
    r"predicts\s+borrower\s+default",
]


def normalize(text):
    return unicodedata.normalize("NFKC", text).replace("−", "-").replace("\\%", "%")


def scan_prohibited(text):
    text = normalize(text)
    hits = []
    for pattern in DANGEROUS:
        for match in re.finditer(pattern, text, re.I):
            before = text[max(0, match.start() - 100) : match.start()]
            negated = bool(re.search(r"\b(?:not|no|neither|excluded|preclude)\b", before, re.I))
            hits.append(
                {
                    "phrase": match.group(),
                    "context": text[max(0, match.start() - 100) : match.end() + 80],
                    "explicit_negative_context": negated,
                }
            )
    return {
        "passed": all(x["explicit_negative_context"] for x in hits),
        "hits": hits,
        "patterns": DANGEROUS,
        "scope": "Automatic phrase detection with explicit-negative context; paired with manual semantic review against all 50 prohibited claim families.",
    }


def flatten_numbers(obj):
    values = []
    if isinstance(obj, bool) or obj is None:
        return values
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            values += flatten_numbers(k) + flatten_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            values += flatten_numbers(v)
    elif isinstance(obj, str):
        values += [float(m.group()) for m in NUMBER.finditer(obj.replace(",", ""))]
    return values


def plain_tex(text, macros=None):
    text = re.sub(r"(?m)^%.*$", "", text)
    for k, v in (macros or {}).items():
        text = text.replace("\\val{" + k + "}", v)
    text = re.sub(
        r"\\(?:ref|label|citep|citet|exhibit|includegraphics|path|url)(?:\[[^]]*\])?\{[^}]*\}",
        "",
        text,
    )
    text = re.sub(r"\\(?:begin|end|section|subsection|setcounter|renewcommand)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    return normalize(text).replace("{", " ").replace("}", " ").replace("$", "")


def source_audit(text, ledger, macros):
    indexed = {c["claim_id"]: c for c in ledger}
    markers = list(re.finditer(r"(?m)^% (Claims|Literature|Reproducibility): ([^\n]+)\n", text))
    blocks = []
    errors = []
    for i, marker in enumerate(markers):
        raw = text[marker.end() : markers[i + 1].start() if i + 1 < len(markers) else len(text)]
        raw = raw.split("\\begin{thebibliography}")[0]
        raw = re.sub(r"(?m)^% Architecture:.*$", "", raw)
        content = plain_tex(raw.replace("10^{-12}", "1e-12"), macros)
        if marker.group(1) != "Claims":
            blocks.append(
                {
                    "line": text[: marker.start()].count("\n") + 1,
                    "category": marker.group(1),
                    "evidence": marker.group(2),
                    "status": "metadata or existing repository citation",
                }
            )
            continue
        ids = re.findall(r"[A-Z]+-\d{3}", marker.group(2))
        bad = [cid for cid in ids if cid not in indexed or not indexed[cid]["manuscript_eligible"]]
        if bad:
            errors.append(f"ineligible claims: {bad}")
        allowed = [0.0, 1.0]
        for cid in ids:
            if cid not in indexed:
                continue
            claim = indexed[cid]
            allowed += flatten_numbers(claim["denominator_or_n_if_applicable"])
            allowed += flatten_numbers(claim["exact_allowed_wording"])
            for reference in claim["source_fields_or_rows"]:
                allowed += flatten_numbers(resolve(ROOT, reference))
        numbers_text = re.sub(r"IFRS[- ]?16|V3|P10", "", content, flags=re.I).replace(
            "1,000", "1000"
        )
        numbers_text = numbers_text.replace("10^-12", "1e-12")
        found = []
        for match in NUMBER.finditer(numbers_text):
            token = match.group()
            value = float(token)
            # Only frozen display rounding, direct values, or fraction-to-percent conversion.
            candidates = [x for a in allowed for x in (a, 100 * a)]
            # A subtraction operand in a displayed identity is not a negative outcome.
            if "\\begin{equation}" in raw:
                candidates += [-a for a in allowed]
            ok = any(
                math.isclose(value, x, abs_tol=1e-12, rel_tol=1e-12)
                or any(token.lstrip("+") == f"{x:.{digits}f}" for digits in [0, 1, 2, 3, 6])
                for x in candidates
            )
            found.append({"token": token, "matched_frozen_value_or_display": ok})
            if not ok:
                errors.append(
                    f"line {text[: marker.start()].count(chr(10)) + 1}: untraced number {token} in {ids}"
                )
        percent = "%" in content
        n_present = bool(
            re.search(r"\d+\s*/\s*\d+|\bn\s*=|\b\d+\s+(?:scenarios|held-out|operating)", content)
        )
        parameter_or_floor = any(cid in ids for cid in ["LIM-001", "LIM-002"]) or bool(
            re.search(r"stipulated probabilities|parameter", content)
        )
        if percent and not (n_present or parameter_or_floor):
            errors.append(
                f"percentage lacks adjacent n/count at line {text[: marker.start()].count(chr(10)) + 1}"
            )
        blocks.append(
            {
                "line": text[: marker.start()].count("\n") + 1,
                "category": "Claims",
                "claim_ids": ids,
                "numerical_tokens": found,
                "percentage_present": percent,
                "adjacent_n_or_count": n_present,
                "parameter_or_floor_exception": parameter_or_floor,
                "traceability": "claim IDs -> frozen artifact selectors; numeric identity/rounding checked; semantics manually reviewed",
            }
        )
    abstract = text.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    conclusion = text.split("\\section{Conclusion}")[1].split("\\appendix")[0]
    for label, part, flag in [
        ("abstract", abstract, "abstract_eligible"),
        ("conclusion", conclusion, "conclusion_eligible"),
    ]:
        for line in re.findall(r"(?m)^% Claims: (.*)$", part):
            for cid in re.findall(r"[A-Z]+-\d{3}", line):
                if not indexed[cid][flag]:
                    errors.append(f"{label} promotes {cid}")
    return {"passed": not errors, "errors": errors, "blocks": blocks}


def word_count(text, macros):
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.S)
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", text, flags=re.S)
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", plain_tex(text, macros)))


def run_audit(source_commit, build_result, visual_review=None):
    folder = ROOT / "results/v3/manuscript"
    folder.mkdir(exist_ok=True)
    text = SOURCE.read_text(encoding="utf-8-sig")
    ledger = load(FREEZE + "claim_ledger.json")
    render = load("results/v3/manuscript/render_manifest.json")
    macros = {c["macro"]: c["rendered"] for c in render["numeric_cells"] if c.get("macro")}
    audit = source_audit(text, ledger, macros)
    source_scan = scan_prohibited(plain_tex(text, macros))
    doc = pymupdf.open(PDF)
    pages = [normalize(page.get_text()) for page in doc]
    full = "\n".join(pages)
    flat_full = re.sub(r"\s+", " ", full)
    (folder / "pdf_text.txt").write_text(full, encoding="utf-8")
    pdf_scan = scan_prohibited(full)
    labels = {
        m.group(1): {"number": m.group(2), "page": int(m.group(3))}
        for m in re.finditer(
            r"\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}",
            SOURCE.with_suffix(".aux").read_text(),
        )
    }
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[pt]\{([^}]+)\}", text)
        for key in group.split(",")
    }
    bib = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", text))
    abstract = text.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    main = text.split("\\section{Introduction}")[1].split("\\appendix")[0]
    literature = text.split("\\section{Related literature and validation perspective}")[1].split(
        "\\section{IFRS 16 covenant-screening setup}"
    )[0]
    used = sorted(set(cid for b in audit["blocks"] for cid in b.get("claim_ids", [])))
    freeze = validate(ROOT)
    cell_errors = []
    for cell in render["numeric_cells"]:
        current = resolve(ROOT, cell["reference"])
        if cell.get("transformation") == "length":
            expected = str(len(current))
        else:
            if isinstance(current, list):
                current = next(iter(current[0].values()))
            expected = (
                count_rate(current, cell["denominator"])
                if cell["denominator"]
                else display(current, cell["style"])
            )
        if current != cell["raw_value"] or expected != cell["rendered"]:
            cell_errors.append(cell)
    outputs_match = all(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        for path, digest in render["output_sha256"].items()
    )
    checks = {
        "source_annotations_and_numbers": audit["passed"],
        "source_prohibited_scan": source_scan["passed"],
        "pdf_prohibited_scan": pdf_scan["passed"],
        "protected_hashes": freeze["passed"],
        "rendered_numeric_cells": not cell_errors,
        "exhibit_hashes": outputs_match,
        "seven_tables": len([k for k in labels if k.startswith("tab:")]) == 7,
        "six_figures": len([k for k in labels if k.startswith("fig:")]) == 6,
        "bibliography": cited == bib and 8 <= len(bib) <= 12,
        "publication_title_uses_ifrs_16": "Validating IFRS 16 Covenant Screening"
        in doc.metadata["title"],
        "internal_version_label_removed": "V3 manuscript" not in full,
        "reader_facing_admission_labels_removed": not re.search(
            r"classification\s+[BC]\b", full, re.I
        ),
        "abstract_length": 160 <= word_count(abstract, macros) <= 190,
        "template_003_auc_note": (
            "Template 003 contains one positive observation; AUC = 1.000 therefore means only "
            "that this single positive ranked above the 46 non-failures in the frozen realization "
            "and should not be interpreted as a precise estimate of discrimination."
        )
        in flat_full,
        "code_availability": bool(
            re.search(r"github\.com/Aniket2002/ifrs16\s*-\s*lbo\s*-\s*engine", full)
        ),
        "coverage_metadata_preserved": render.get("non_public_coverage_metadata", {})
        == {
            "decision": "Former Table A4 removed from the public manuscript; exact coverage remains repository validation metadata.",
            "source": "results/v3/post_optimization_diagnostics/coverage.json",
            "search_kernel": {
                "statement_coverage_percent": 88.68,
                "branch_coverage_percent": 70.0,
            },
            "runner": {
                "statement_coverage_percent": 59.89,
                "branch_coverage_percent": 55.0,
            },
        },
        "no_placeholder_or_broken_reference": not re.search(
            r"\?\?|\bTODO\b|\bPLACEHOLDER\b|\ufffd|[A-Z]+-\d{3}", full
        ),
        "no_blank_pages": all(len(p.strip()) > 80 for p in pages),
        "no_duplicate_pages": len(set(pages)) == len(pages),
        "title_and_author": doc.metadata["author"] == "Aniket Bhardwaj"
        and doc.metadata["title"].startswith("Ranking Is Not Threshold Portability:"),
        "build": build_result["returncode"] == 0 and not build_result["serious_warnings"],
    }
    report = {
        "manuscript_source_path": SOURCE.relative_to(ROOT).as_posix(),
        "pdf_path": PDF.relative_to(ROOT).as_posix(),
        "source_commit": source_commit,
        "claim_freeze_commit": render["claim_freeze_commit"],
        "title": doc.metadata["title"],
        "author": doc.metadata["author"],
        "page_count": len(doc),
        "abstract_word_count": word_count(abstract, macros),
        "main_text_word_count": word_count(main, macros),
        "literature_context_word_count": word_count(literature, macros),
        "word_count_policy": "TeX prose after macro expansion; excludes abstract from main count, floats, table fragments, displayed equations, bibliography and appendices; includes headings and inline numeric tokens.",
        "table_count": 7,
        "figure_count": 6,
        "appendix_count": 3,
        "exhibit_labels": labels,
        "claim_ids_actually_used": used,
        "primary_claim_ids_used": [
            c for c in load(FREEZE + "freeze_summary.json")["primary_claim_ids"] if c in used
        ],
        "secondary_claim_ids_used": [
            c for c in load(FREEZE + "freeze_summary.json")["secondary_claim_ids"] if c in used
        ],
        "numerical_claims_found": audit["blocks"],
        "rendered_number_traceability": {
            "manifest": "results/v3/manuscript/render_manifest.json",
            "numeric_cells_checked": len(render["numeric_cells"]),
            "errors": cell_errors,
            "all_exhibit_hashes_match": outputs_match,
        },
        "traceability_status": audit,
        "prohibited_phrase_scan": {"source": source_scan, "pdf": pdf_scan},
        "small_n_compliance": {
            "source": audit["passed"],
            "tables": "Observed risk cells carry count/n; fold return cells have n in same row or immediately adjacent table note; stipulated policy percentages and deterministic return floor are distinguished.",
            "figures": "Class counts, total n or within-regime n in plot labels/captions; axes are display scales, not estimated rates.",
        },
        "bayes_treatment": "C: excluded main results; concise exclusion appendix; no posterior LBO integration",
        "optimization_treatment": "B: methodological appendix only; no numerical result in abstract or conclusion",
        "bibliography_status": {
            "passed": checks["bibliography"],
            "cited_keys": sorted(cited),
            "entries": len(bib),
            "provenance": "IFRS Foundation and peer-reviewed publisher, journal, PubMed/PMC, and MIT Press records verified during the publication-polish pass.",
        },
        "publication_polish": {
            "literature_sources_added": [
                "Dichev and Skinner (2002)",
                "Christensen and Nikolaev (2012)",
                "Fawcett (2006)",
                "Hand (2009)",
                "Steyerberg et al. (2010)",
                "Vickers and Elkin (2006)",
                "Quinonero-Candela et al. (2008)",
            ],
            "title_normalization": "Reader-facing references use IFRS 16; repository identifiers remain unchanged.",
            "internal_version_label": "V3 manuscript removed; title page dated September 2026.",
            "abstract_revision": "Retains all frozen headline results and adds the Template 004 balanced-accuracy comparison.",
            "template_003_auc_note": "Table 4 states that the single positive ranked above 46 non-failures and that AUC 1.000 is not a precise discrimination estimate.",
            "coverage_table": render["non_public_coverage_metadata"],
            "code_availability": "Public repository, canonical branch and exact manuscript-source lineage stated in the reproducibility appendix.",
            "numerical_conclusions": "Unchanged; no experiment, model, threshold, simulation, optimization, Bayesian result or admission decision was changed.",
        },
        "build_status": build_result,
        "protected_artifact_hash_status": freeze,
        "visual_review": visual_review or {"status": "pending"},
        "checks": checks,
        "passed": all(checks.values()),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "manual_review_scope": "Substantive sentences reviewed against eligible wording and exact source references; automatic numeric matching alone is not semantic proof. All 50 prohibited families reviewed; no prior headline numbers imported.",
    }
    (folder / "final_claim_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    build_record = load("results/v3/manuscript/build_record.json")
    review_path = ROOT / "results/v3/manuscript/visual_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    result = run_audit(build_record["source_commit"], build_record, review)
    reviewed = bool(
        review
        and review.get("status") == "passed"
        and review.get("pdf_sha256") == result["pdf_sha256"]
    )
    result["checks"]["final_visual_review"] = reviewed
    result["checks"]["committed_source_build"] = not build_record["source_files_dirty"]
    result["passed"] = all(result["checks"].values())
    (ROOT / "results/v3/manuscript/final_claim_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

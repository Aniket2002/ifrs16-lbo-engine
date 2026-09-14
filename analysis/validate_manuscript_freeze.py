"""Read-only validation of claim metadata and frozen source snapshots; no models run."""

import csv
import hashlib
import json
from pathlib import Path

CLASSES = {
    "VERIFIED_MECHANICAL",
    "SYNTHETIC_RESULT",
    "CONDITIONAL_SYNTHETIC_RESULT",
    "DESCRIPTIVE_DIAGNOSTIC",
    "EXCLUDED_METHOD",
    "LIMITATION",
    "INTERPRETIVE_SYNTHESIS",
    "PROHIBITED_CLAIM",
}
REQUIRED = {
    "claim_id",
    "short_name",
    "exact_allowed_wording",
    "claim_class",
    "evidence_type",
    "source_artifact",
    "source_fields_or_rows",
    "numerical_value_if_any",
    "unit",
    "denominator_or_n_if_applicable",
    "uncertainty_status",
    "qualification_required",
    "manuscript_section",
    "prominence",
    "manuscript_eligible",
    "abstract_eligible",
    "conclusion_eligible",
    "figure_or_table_candidate",
    "prohibited_extensions",
    "supersedes_prior_claim",
    "notes",
    "admission_status",
    "quantitative",
    "subgroup_percentage",
}


def resolve(root, reference):
    path = root / reference["artifact"]
    if "lines" in reference:
        start, end = reference["lines"]
        lines = path.read_text(encoding="utf-8").splitlines()
        if not 1 <= start <= end <= len(lines):
            raise ValueError("invalid source line range")
        return lines[start - 1 : end]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            row
            for row in rows
            if all(row.get(k) == str(v) for k, v in reference.get("where", {}).items())
        ]
        if not rows:
            raise ValueError("CSV selector matched no rows")
        return [{k: row[k] for k in reference.get("fields", row)} for row in rows]
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in reference.get("keys", []):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def validate_claims(root, claims):
    errors = []
    seen = set()
    for claim in claims:
        cid = claim.get("claim_id", "<missing>")

        def fail(message):
            errors.append(f"{cid}: {message}")

        missing = REQUIRED - claim.keys()
        if missing:
            fail(f"missing fields: {sorted(missing)}")
            continue
        if cid in seen:
            fail("duplicate claim ID")
        seen.add(cid)
        if claim["claim_class"] not in CLASSES:
            fail("invalid admissibility class")
        if claim["claim_class"] == "PROHIBITED_CLAIM" and any(
            claim[k] for k in ("manuscript_eligible", "abstract_eligible", "conclusion_eligible")
        ):
            fail("prohibited claim promoted")
        if claim["manuscript_eligible"] and not claim["manuscript_section"]:
            fail("eligible claim missing section")
        if claim["subgroup_percentage"] and not claim["denominator_or_n_if_applicable"]:
            fail("subgroup percentage missing denominator")
        if cid.startswith("BAY-") and (
            claim["claim_class"] != "EXCLUDED_METHOD"
            or claim["admission_status"] != "C / excluded methodological experiment"
        ):
            fail("Bayesian admission changed")
        if (
            cid.startswith("OPT-")
            and claim["admission_status"] != "B / methodological demonstration only"
        ):
            fail("optimization admission changed")
        if (
            cid.startswith(("OPT-", "BAY-"))
            and claim["quantitative"]
            and (claim["abstract_eligible"] or claim["conclusion_eligible"])
        ):
            fail("secondary/excluded numerical result promoted")
        references = claim["source_fields_or_rows"]
        if sorted(set(claim["source_artifact"])) != sorted({r["artifact"] for r in references}):
            fail("artifact list differs from exact references")
        if claim["quantitative"] and not references:
            fail("quantitative claim has no source")
        try:
            values = [resolve(root, r) for r in references]
            if claim["quantitative"] and values != claim["numerical_value_if_any"]:
                fail("source snapshot mismatch")
        except (OSError, ValueError, KeyError, IndexError) as exc:
            fail(f"unresolvable source: {exc}")
        if claim["manuscript_eligible"]:
            wording = claim["exact_allowed_wording"]
            if any(s in wording for s in ("84% statement", "59% statement", "85% branch")):
                fail("stale coverage wording")
    return errors


def validate(root):
    folder = root / "results/v3/manuscript_freeze"

    def read(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    claims = read("claim_ledger.json")
    errors = validate_claims(root, claims)
    indexed = {c["claim_id"]: c for c in claims}
    for path, expected in read("source_manifest.json")["sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            errors.append(f"frozen source changed: {path}")
    traces = read("claim_traceability.json")
    if len(traces) != len(claims) or {t["claim_id"] for t in traces} != set(indexed):
        errors.append("traceability coverage mismatch")
    for trace in traces:
        c = indexed[trace["claim_id"]]
        for tk, ck in [
            ("evidence", "source_fields_or_rows"),
            ("manuscript_section", "manuscript_section"),
            ("denominator_metadata", "denominator_or_n_if_applicable"),
        ]:
            if trace[tk] != c[ck]:
                errors.append(f"trace mismatch: {trace['claim_id']} {tk}")
    with (folder / "claim_ledger.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = [
        {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in c.items()}
        for c in claims
    ]
    if rows != expected_rows:
        errors.append("CSV/JSON ledgers differ")
    structure = read("manuscript_structure.json")
    for section in structure["sections"]:
        for cid in section["claim_ids_permitted"]:
            if cid not in indexed or not indexed[cid]["manuscript_eligible"]:
                errors.append(f"section promotes invalid claim: {cid}")
    for slot in structure["abstract_blueprint"]:
        for cid in slot["permitted_claim_ids"]:
            if not indexed[cid]["abstract_eligible"]:
                errors.append(f"abstract promotes ineligible claim: {cid}")
    for name, refs_key in [
        ("table_plan.json", "source_fields_or_rows"),
        ("figure_plan.json", "exact_data_source"),
    ]:
        for plan in read(name):
            for cid in plan["claim_ids"]:
                if cid not in indexed or not indexed[cid]["manuscript_eligible"]:
                    errors.append(f"plan references invalid claim: {cid}")
            for reference in plan[refs_key]:
                resolve(root, reference)
    prohibited = read("prohibited_claims.json")
    if {p["claim_id"] for p in prohibited} != {
        c["claim_id"] for c in claims if c["claim_class"] == "PROHIBITED_CLAIM"
    }:
        errors.append("prohibited register and ledger differ")
    return {
        "passed": not errors,
        "errors": errors,
        "claims_checked": len(claims),
        "quantitative_claims_checked": sum(c["quantitative"] for c in claims),
        "prohibited_claims_checked": len(prohibited),
        "frozen_source_files_hash_checked": len(read("source_manifest.json")["sha256"]),
        "checks": [
            "required fields and classes",
            "exact source selectors and snapshots",
            "CSV/JSON equality",
            "denominator metadata",
            "B/C admission restrictions",
            "prohibited and stale coverage restrictions",
            "traceability coverage",
            "section/abstract/visual claim links",
            "frozen file SHA-256",
        ],
    }


if __name__ == "__main__":
    report = validate(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

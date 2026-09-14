"""Git-object source integrity and explicit historical freeze auditing."""

import argparse
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

CURRENT_MANIFEST = "results/v3/integrity/current_git_blob_manifest.json"
HISTORICAL_MANIFEST = "results/v3/manuscript_freeze/source_manifest.json"


def _git(root, *args, text=False):
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )
    if process.returncode:
        stderr = process.stderr.strip() if text else process.stderr.decode(errors="replace").strip()
        raise ValueError(f"git {' '.join(args)} failed: {stderr}")
    return process.stdout


def resolve_commit(root, commit):
    """Resolve a revision to a canonical commit object ID."""
    return _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}", text=True).strip()


@lru_cache(maxsize=None)
def read_git_blob(root, commit, path):
    """Read exact blob bytes for path at commit, independent of the checkout."""
    object_id = _git(root, "rev-parse", f"{commit}:{path}", text=True).strip()
    return object_id, _git(root, "cat-file", "blob", object_id)


def _blob_observation(root, commit, entry):
    object_id, content = read_git_blob(root, commit, entry["path"])
    return {
        "path": entry["path"],
        "git_blob_sha": object_id,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_length": len(content),
    }


def validate_current_integrity(root, commit="HEAD", manifest_path=CURRENT_MANIFEST):
    """Verify protected identities from Git blobs, never from worktree bytes."""
    root = Path(root)
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    baseline = resolve_commit(root, manifest["current_integrity_commit"])
    verified_commit = resolve_commit(root, commit)
    errors = []
    observations = []

    if manifest.get("contract") != "SHA-256 of canonical Git blob bytes":
        errors.append("current integrity manifest has an unsupported contract")

    paths = [entry.get("path") for entry in manifest.get("protected_files", [])]
    if len(paths) != len(set(paths)):
        errors.append("current integrity manifest contains duplicate protected paths")

    required = {"path", "git_blob_sha", "sha256", "byte_length"}
    for entry in manifest.get("protected_files", []):
        if set(entry) != required:
            errors.append(
                f"invalid current integrity entry schema: {entry.get('path', '<missing>')}"
            )
            continue
        try:
            baseline_observed = _blob_observation(root, baseline, entry)
            observed = _blob_observation(root, verified_commit, entry)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        observations.append(observed)
        for field in ("git_blob_sha", "sha256", "byte_length"):
            if baseline_observed[field] != entry[field]:
                errors.append(f"integrity manifest baseline mismatch: {entry['path']} {field}")
            if observed[field] != entry[field]:
                errors.append(f"protected Git blob changed: {entry['path']} {field}")

    historical_path = manifest["historical_freeze_manifest_path"]
    historical_entry = {
        "path": historical_path,
        "git_blob_sha": manifest["historical_freeze_manifest_git_blob_sha"],
        "sha256": manifest["historical_freeze_manifest_sha256"],
        "byte_length": manifest["historical_freeze_manifest_byte_length"],
    }
    try:
        historical_baseline = _blob_observation(root, baseline, historical_entry)
        historical_current = _blob_observation(root, verified_commit, historical_entry)
        for field in ("git_blob_sha", "sha256", "byte_length"):
            if historical_baseline[field] != historical_entry[field]:
                errors.append(f"historical manifest baseline mismatch: {field}")
            if historical_current[field] != historical_entry[field]:
                errors.append(f"historical manifest changed: {field}")
    except ValueError as exc:
        errors.append(str(exc))

    return {
        "passed": not errors,
        "errors": errors,
        "contract": manifest["contract"],
        "manifest_path": manifest_path,
        "baseline_commit": baseline,
        "verified_commit": verified_commit,
        "protected_files_checked": len(observations),
        "historical_manifest_unchanged": not any(
            error.startswith("historical manifest") for error in errors
        ),
        "historical_manifest_status": manifest["historical_manifest_status"],
        "unresolved_historical_paths": manifest["unresolved_historical_paths"],
    }


def audit_historical_manifest(root, manifest_path=HISTORICAL_MANIFEST):
    """Report reproducibility of the immutable checkout-byte freeze manifest."""
    root = Path(root)
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    source_commit = resolve_commit(root, manifest["source_commit"])
    entries = []
    unresolved = []
    checkout_mismatches = []

    for path, expected in manifest["sha256"].items():
        _, source = read_git_blob(root, source_commit, path)
        worktree = (root / path).read_bytes()
        source_lf = source.replace(b"\r\n", b"\n")
        source_crlf = source_lf.replace(b"\n", b"\r\n")
        worktree_digest = hashlib.sha256(worktree).hexdigest()
        source_digest = hashlib.sha256(source).hexdigest()
        crlf_digest = hashlib.sha256(source_crlf).hexdigest()
        if expected == source_digest:
            classification = "GIT_BLOB_EXACT"
        elif expected == crlf_digest:
            classification = "CRLF_CHECKOUT_REPRODUCIBLE"
        else:
            classification = "UNRESOLVED_HISTORICAL_ENTRY"
            unresolved.append(path)
        if worktree_digest != expected:
            checkout_mismatches.append(path)
        entries.append(
            {
                "path": path,
                "expected_sha256": expected,
                "worktree_sha256": worktree_digest,
                "git_blob_sha256": source_digest,
                "crlf_git_blob_sha256": crlf_digest,
                "classification": classification,
            }
        )

    return {
        "passed": not unresolved,
        "status": "historical anomalies remain" if unresolved else "fully reproducible",
        "manifest_path": manifest_path,
        "source_commit": source_commit,
        "contract": "historical SHA-256 of checkout bytes; retained as provenance",
        "total_entries": len(entries),
        "reproducible_from_git_or_eol": len(entries) - len(unresolved),
        "worktree_exact_matches": len(entries) - len(checkout_mismatches),
        "checkout_mismatches": checkout_mismatches,
        "unresolved_historical_entries": len(unresolved),
        "unresolved_historical_paths": unresolved,
        "entries": entries,
        "note": "This audit does not repair, reinterpret, or overwrite the historical manifest.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["current", "historical"], default="current")
    parser.add_argument("--commit", default="HEAD", help="Git revision checked in current mode")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = (
        validate_current_integrity(root, args.commit)
        if args.mode == "current"
        else audit_historical_manifest(root)
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

# V3 source-integrity governance

## Governing distinction

The repository keeps two separate source-integrity records for different purposes:

1. **Historical provenance.**
   `results/v3/manuscript_freeze/source_manifest.json` is the immutable record written
   during the manuscript freeze. It documents the hashes recorded at that time, including
   their known limitations.
2. **Current reproducibility.**
   `results/v3/integrity/current_git_blob_manifest.json` is the operational record for
   source-identity validation. It identifies protected files by their canonical Git object
   content at a named commit.

The historical freeze manifest is preserved unchanged as provenance. Current
source-integrity verification uses canonical Git blob bytes at the recorded commit and does
not reinterpret or overwrite the historical manifest.

## Historical freeze limitation

The historical manifest intended to protect the source code, synthetic inputs, and archived
evidence supporting the v3 claim freeze. Its validator hashed `Path.read_bytes()` from the
active checkout. Those bytes depend on Git checkout configuration: for example, Git can
materialize the same LF blob as CRLF on Windows when `core.autocrlf=true`.

Forensic comparison found that 64 of 66 historical hashes can be reproduced from the
recorded Git blobs either directly or after the checkout's LF/CRLF representation. Two
recorded hashes cannot be reproduced from any committed version of the paths or from normal
newline and encoding transformations:

- `results/v3/baseline/pipeline_step_2.txt`
- `results/v3/foundation_validation.json`

The matching historical byte sequences are unavailable, so the original manifest cannot
be claimed as fully reproducible. The manifest is not corrected, normalized, regenerated,
or replaced. The explicit historical audit continues to return a non-green result and lists
both paths.

This anomaly does not imply post-freeze research mutation. All 66 protected Git blobs at
the current integrity baseline are byte-identical to the blobs at the historical manifest's
recorded source commit. Git history contains no protected-content change between those
points. See [the frozen-manifest forensic report](V3_FROZEN_MANIFEST_FORENSICS.md) for the
file-by-file evidence.

## Current operational contract

The current contract is:

> Protected source identity is the SHA-256, Git blob object ID, and byte length of canonical
> Git blob bytes at a named commit.

The validator resolves a commit and reads each object through Git plumbing. It does not hash
worktree bytes, decode text, normalize newlines, or accept LF/CRLF variants heuristically.
The same commit therefore has the same protected identity on Windows, macOS, and Linux,
independent of `core.autocrlf`, editor settings, or filesystem representation.

The operational manifest records:

- baseline commit `8e64d125eaa8fcb4edd3a138ae3048eaa6dea928`;
- all 66 protected paths and their Git blob object IDs;
- SHA-256 and byte length for every canonical blob;
- the canonical hash and object ID of the immutable historical manifest;
- the historical 64/66 reproducibility finding and two unresolved paths;
- an explicit statement that it does not repair or replace the historical freeze.

Operational validation compares the protected blobs at the current `HEAD` with this record.
A committed substantive text or binary change fails. An uncommitted checkout-only newline
change does not alter a Git object and does not affect source identity.

## Commands and gate behavior

Run the current operational source check directly:

```powershell
python -m analysis.source_integrity --mode current
```

Run the separate historical audit explicitly:

```powershell
python -m analysis.source_integrity --mode historical
```

The historical command is expected to exit nonzero while reporting exactly two unresolved
entries. That result is evidence visibility, not an operational build failure.

`python analysis/validate_manuscript_freeze.py` continues to validate the frozen claim
metadata, exact selectors, admission restrictions, traceability, and plans. Its operational
source gate now calls current Git-blob validation. The renderer, manuscript builder, and
manuscript audit use that current gate through the same function. Their audit output retains
the historical status and unresolved path list as metadata without presenting the anomaly
as repaired.

## Permitted claims

Future validation may claim that:

- current protected Git source identity is verified across platforms;
- the 66 protected blobs still equal the historical source-commit blobs;
- the historical manifest is preserved unchanged;
- 64 historical entries are reproducible from Git content or checkout EOL representation;
- two historical entries remain unresolved.

Future validation must not claim that:

- the historical freeze manifest has been repaired or fully reproduced;
- EOL normalization recovers the two unresolved historical byte sequences;
- a green current integrity check validates arbitrary uncommitted worktree content;
- the governance repair changes or revalidates any research result;
- current operational integrity erases the historical anomaly.


"""Build versioned v3 LaTeX with the existing Tectonic toolchain; no experiments."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis.scripts.audit_manuscript_v3 import ROOT, SOURCE, run_audit
from analysis.scripts.render_manuscript_v3 import Renderer


def build(draft=False, executable=None):
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = [
        "paper/ifrs16_lbo_ssrn_v3.tex",
        "analysis/scripts/render_manuscript_v3.py",
        "analysis/scripts/build_manuscript_v3.py",
        "analysis/scripts/audit_manuscript_v3.py",
        "tests/test_manuscript_rendering.py",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip()
    if dirty and not draft:
        raise RuntimeError("Commit final source first, or use --draft for pre-commit inspection.")
    Renderer().run()
    (ROOT / "paper/generated/v3/provenance.tex").write_text(
        "\\newcommand{\\buildsource}{" + source_commit + "}\n", encoding="utf-8"
    )
    tools_dir = Path(tempfile.gettempdir()) / "ifrs16-paper-tools"
    exe = executable or shutil.which("tectonic") or str(tools_dir / "tectonic.exe")
    if not Path(exe).is_file():
        raise FileNotFoundError(
            "Provide Tectonic via PATH or --tectonic; see paper/REPRODUCE_V3.md."
        )
    tools_dir.mkdir(exist_ok=True)
    conf = tools_dir / "fonts.conf"
    conf.write_text(
        '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>C:/Windows/Fonts</dir><cachedir>'
        + str(tools_dir / "font-cache").replace("\\", "/")
        + "</cachedir></fontconfig>",
        encoding="utf-8",
    )
    env = os.environ.copy()
    if os.name == "nt":
        env["FONTCONFIG_FILE"] = str(conf)
    env["SOURCE_DATE_EPOCH"] = subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", source_commit], cwd=ROOT, text=True
    ).strip()
    command = [exe, "--keep-logs", "--keep-intermediates", str(SOURCE.relative_to(ROOT))]
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    folder = ROOT / "results/v3/manuscript"
    (folder / "build_console.txt").write_text(process.stdout, encoding="utf-8")
    log = (
        SOURCE.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        if SOURCE.with_suffix(".log").exists()
        else ""
    )
    (folder / "latex_log.txt").write_text(log, encoding="utf-8")
    warnings = [
        line
        for line in (process.stdout + "\n" + log).splitlines()
        if re.search(
            r"Overfull|undefined|Missing character|already defined|^!|LaTeX Error", line, re.I
        )
    ]
    record = {
        "returncode": process.returncode,
        "command": command,
        "source_commit": source_commit,
        "source_files_dirty": bool(dirty),
        "draft": draft,
        "serious_warnings": warnings,
        "console_path": "results/v3/manuscript/build_console.txt",
        "tectonic_version": subprocess.check_output([exe, "--version"], text=True).strip(),
        "tectonic_executable_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
        "source_date_epoch": env["SOURCE_DATE_EPOCH"],
    }
    (folder / "build_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if process.returncode:
        raise RuntimeError(process.stdout)
    audit = run_audit(source_commit, record)
    print(
        json.dumps(
            {
                "build": record,
                "audit_checks": audit["checks"],
                "trace_errors": audit["traceability_status"]["errors"],
                "page_count": audit["page_count"],
                "abstract_words": audit["abstract_word_count"],
                "main_text_words": audit["main_text_word_count"],
            },
            indent=2,
        )
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--tectonic")
    args = parser.parse_args()
    result = build(args.draft, args.tectonic)
    raise SystemExit(0 if result["passed"] else 1)

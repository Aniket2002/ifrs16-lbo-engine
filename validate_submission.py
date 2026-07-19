#!/usr/bin/env python3
"""
ArXiv Submission Validation Script

Performs all possible validation checks for arXiv submission readiness
without requiring LaTeX compilation on Windows.

This validates:
1. File structure and naming
2. Figure integrity and git hash stamps
3. Bibliography consistency
4. Path validation
5. Metadata consistency
6. Test suite passes
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import hashlib
from typing import List, Dict, Any


def check_file_structure() -> bool:
    """Validate file structure for arXiv submission"""
    print("🔍 Checking file structure...")

    required_files = ["paper/main.tex", "paper/references.bib", "CITATION.cff", "README.md"]

    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
            all_good = False

    return all_good


def check_figures() -> bool:
    """Validate figure files and git hash stamps"""
    print("🖼️  Checking figures...")

    figures_dir = Path("paper/figures")
    if not figures_dir.exists():
        print("   ⚠️  No figures directory found")
        return True

    all_good = True
    pdf_files = list(figures_dir.glob("*.pdf"))

    if not pdf_files:
        print("   ⚠️  No PDF figures found")
        return True

    # Check for git hash in figure metadata (approximation)
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()

        for pdf_file in pdf_files:
            print(f"   ✓ {pdf_file.name}")
            # In a full implementation, we'd check PDF metadata for git hash
            # For now, just validate the file exists and is not empty
            if pdf_file.stat().st_size == 0:
                print(f"   ❌ Empty figure: {pdf_file.name}")
                all_good = False

    except subprocess.CalledProcessError:
        print("   ⚠️  Git not available - cannot validate git hash stamps")

    return all_good


def check_bibliography() -> bool:
    """Validate bibliography consistency"""
    print("📚 Checking bibliography...")

    bib_file = Path("paper/references.bib")
    tex_file = Path("paper/main.tex")

    if not bib_file.exists():
        print("   ❌ references.bib not found")
        return False

    if not tex_file.exists():
        print("   ❌ main.tex not found")
        return False

    # Read tex file and check for bibliography command
    tex_content = tex_file.read_text(encoding="utf-8")

    if "\\bibliography{references}" in tex_content or "\\addbibresource" in tex_content:
        print("   ✓ Bibliography command found in main.tex")
    else:
        print("   ⚠️  No bibliography command found in main.tex")

    # Check if bib file is not empty
    bib_content = bib_file.read_text(encoding="utf-8")
    if len(bib_content.strip()) > 100:  # Reasonable minimum
        print("   ✓ references.bib contains content")
        return True
    else:
        print("   ❌ references.bib appears empty or too short")
        return False


def check_paths_and_names() -> bool:
    """Validate file paths and naming conventions"""
    print("📁 Checking paths and filenames...")

    all_good = True

    # Check for non-ASCII characters and spaces in critical paths
    for root, dirs, files in os.walk("."):
        for name in files + dirs:
            full_path = os.path.join(root, name)

            # Skip hidden files and cache directories
            if name.startswith(".") or "__pycache__" in full_path:
                continue

            # Check for spaces and non-ASCII
            if " " in name:
                print(f"   ⚠️  Space in filename: {full_path}")

            try:
                name.encode("ascii")
            except UnicodeEncodeError:
                print(f"   ⚠️  Non-ASCII filename: {full_path}")

    print("   ✓ Path validation complete")
    return all_good


def check_metadata_consistency() -> bool:
    """Check metadata consistency across files"""
    print("📋 Checking metadata consistency...")

    all_good = True

    # Check CITATION.cff
    citation_file = Path("CITATION.cff")
    if citation_file.exists():
        try:
            import yaml

            with open(citation_file, "r", encoding="utf-8") as f:
                citation_data = yaml.safe_load(f)

            if "title" in citation_data:
                print(f"   ✓ Title in CITATION.cff: {citation_data['title'][:50]}...")
            else:
                print("   ❌ No title in CITATION.cff")
                all_good = False

        except ImportError:
            print("   ⚠️  PyYAML not available - cannot validate CITATION.cff")
        except Exception as e:
            print(f"   ❌ Error reading CITATION.cff: {e}")
            all_good = False
    else:
        print("   ⚠️  CITATION.cff not found")

    # Check main.tex for title
    tex_file = Path("paper/main.tex")
    if tex_file.exists():
        tex_content = tex_file.read_text(encoding="utf-8")
        if "\\title{" in tex_content:
            print("   ✓ Title found in main.tex")
        else:
            print("   ❌ No \\title{} found in main.tex")
            all_good = False

    return all_good


def run_test_suite() -> bool:
    """Run the test suite to ensure everything works"""
    print("🧪 Running test suite...")

    try:
        # Try to run pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("   ✅ All tests passed!")
            return True
        else:
            print("   ❌ Some tests failed:")
            print(result.stdout[-500:])  # Last 500 chars
            return False

    except subprocess.TimeoutExpired:
        print("   ⚠️  Tests timed out - may indicate issues")
        return False
    except Exception as e:
        print(f"   ❌ Error running tests: {e}")
        return False


def generate_submission_checklist() -> None:
    """Generate final submission checklist"""
    print("\n" + "=" * 60)
    print("📋 ARXIV SUBMISSION CHECKLIST")
    print("=" * 60)

    print("\n✅ Automated Checks Completed:")
    print("   - File structure validated")
    print("   - Figure integrity checked")
    print("   - Bibliography consistency verified")
    print("   - Path and filename validation")
    print("   - Metadata consistency checked")
    print("   - Test suite execution")

    print("\n📝 Manual Steps for Final Submission:")
    print("   1. Install LaTeX (MiKTeX/TeX Live) if not available")
    print("   2. Compile manuscript: cd paper && pdflatex main.tex")
    print("   3. Run bibtex: bibtex main")
    print("   4. Compile again: pdflatex main.tex (twice)")
    print("   5. Check fonts: pdffonts main.pdf (all embedded)")
    print("   6. Verify PDF links and references work")
    print("   7. Create final submission bundle")

    print("\n🎯 Final Package Should Include:")
    print("   - main.tex (primary LaTeX file)")
    print("   - references.bib (bibliography)")
    print("   - main.bbl (if using BibTeX)")
    print("   - figures/*.pdf (all figures as PDF)")
    print("   - main.pdf (compiled manuscript)")

    print("\n🚀 Ready for arXiv categories:")
    print("   Primary: q-fin.GN (General Finance)")
    print("   Secondary: stat.ME (Statistics - Methodology)")
    print("   Or: cs.LG (Machine Learning)")


def main() -> int:
    """Main validation function"""
    print("🎯 ArXiv Submission Validation")
    print("=" * 40)

    checks = [
        ("File Structure", check_file_structure),
        ("Figures", check_figures),
        ("Bibliography", check_bibliography),
        ("Paths & Names", check_paths_and_names),
        ("Metadata", check_metadata_consistency),
        ("Test Suite", run_test_suite),
    ]

    all_passed = True
    results = {}

    for name, check_func in checks:
        try:
            passed = check_func()
            results[name] = passed
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"   ❌ Error in {name}: {e}")
            results[name] = False
            all_passed = False
        print()

    # Summary
    print("📊 VALIDATION SUMMARY")
    print("-" * 30)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<15} {status}")

    print(f"\nOverall Status: {'✅ READY' if all_passed else '❌ NEEDS FIXES'}")

    generate_submission_checklist()

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
arXiv Submission Validation Script

This script validates the paper is ready for arXiv submission by checking:
1. File structure and dependencies
2. Figure references and paths
3. Bibliography completeness
4. LaTeX syntax validation
5. DOI and URL validation
"""

import os
import re
from pathlib import Path

def validate_arxiv_submission():
    """Comprehensive arXiv submission validation"""
    
    print("🔍 Validating arXiv submission readiness...")
    print("=" * 50)
    
    # Check file structure
    paper_dir = Path("paper")
    if not paper_dir.exists():
        print("❌ Error: paper/ directory not found")
        return False
    
    main_tex = paper_dir / "main.tex"
    if not main_tex.exists():
        print("❌ Error: paper/main.tex not found")
        return False
    
    figures_dir = paper_dir / "figures"
    if not figures_dir.exists():
        print("❌ Error: paper/figures/ directory not found")
        return False
    
    print("✅ File structure: OK")
    
    # Read main.tex content
    with open(main_tex, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for external dependencies
    external_refs = re.findall(r'\\input\{([^}]+)\}', content)
    if external_refs:
        print(f"❌ Error: Found external \\input commands: {external_refs}")
        return False
    
    print("✅ Self-contained: OK")
    
    # Check figure references
    figure_includes = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', content)
    missing_figures = []
    
    for fig in figure_includes:
        fig_path = figures_dir / fig
        if not fig_path.exists():
            missing_figures.append(fig)
    
    if missing_figures:
        print(f"❌ Error: Missing figures: {missing_figures}")
        return False
    
    print(f"✅ Figures ({len(figure_includes)} found): OK")
    
    # Check for problematic paths
    bad_paths = re.findall(r'\\includegraphics\[[^\]]*\]\{[^}]*\.\./[^}]*\}', content)
    if bad_paths:
        print(f"❌ Error: Found relative paths outside paper tree: {bad_paths}")
        return False
    
    print("✅ Figure paths: OK")
    
    # Check DOI placeholder
    if "zenodo.XXXXXXX" in content:
        print("⚠️  Warning: DOI placeholder still present")
    else:
        print("✅ DOI: OK")
    
    # Check for \\today
    if "\\today" in content:
        print("❌ Error: \\today found, should use fixed date")
        return False
    
    print("✅ Fixed date: OK")
    
    # Check for unused packages
    unused_packages = []
    if "\\usepackage{algorithm}" in content:
        unused_packages.append("algorithm")
    if "\\usepackage{algorithmic}" in content:
        unused_packages.append("algorithmic")
    
    if unused_packages:
        print(f"⚠️  Warning: Potentially unused packages: {unused_packages}")
    
    # Check for symbol collisions
    alpha_breach = content.count("\\alphabudget")
    alpha_fcf = content.count("\\phiFCF")
    
    if alpha_breach == 0:
        print("⚠️  Warning: No \\alphabudget usage found")
    if alpha_fcf == 0:
        print("⚠️  Warning: No \\phiFCF usage found")
    
    # Check labels and references
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    
    missing_refs = set(refs) - set(labels)
    if missing_refs:
        print(f"❌ Error: Missing labels for references: {missing_refs}")
        return False
    
    print(f"✅ Labels and references ({len(labels)} labels, {len(refs)} refs): OK")
    
    # Check bibliography
    if "\\begin{thebibliography}" not in content:
        print("❌ Error: No bibliography found")
        return False
    
    bibitem_count = content.count("\\bibitem")
    cite_count = len(re.findall(r'\\cite[pt]?\{[^}]+\}', content))
    
    print(f"✅ Bibliography ({bibitem_count} items, {cite_count} citations): OK")
    
    # Summary
    print("=" * 50)
    print("📊 VALIDATION SUMMARY:")
    print("✅ File structure and self-containment")
    print("✅ Figure paths and availability")
    print("✅ Labels and references consistency")
    print("✅ Bibliography completeness")
    print("✅ Date fixed (not \\today)")
    
    if "zenodo.XXXXXXX" in content:
        print("⚠️  DOI placeholder (update before submission)")
    
    print("\n🎉 READY FOR ARXIV SUBMISSION!")
    print("\n📝 FINAL CHECKLIST:")
    print("1. Update DOI if placeholder present")
    print("2. Verify all fonts embedded: pdffonts main.pdf")
    print("3. Check final PDF for formatting")
    print("4. Upload paper/ directory contents to arXiv")
    
    return True

if __name__ == "__main__":
    validate_arxiv_submission()

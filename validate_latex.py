"""
LaTeX Syntax Validator for IFRS-16 LBO Paper

This script checks for common LaTeX compilation issues without requiring pdflatex:
- Missing references/labels
- Unmatched braces/environments  
- Undefined citations
- Missing figure files
- Malformed commands
"""

import re
import os
from pathlib import Path

def validate_latex_syntax(tex_file):
    """Validate LaTeX file for common compilation issues"""
    
    print(f"🔍 Validating LaTeX syntax for {tex_file}")
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check for unmatched braces
    brace_count = content.count('{') - content.count('}')
    if brace_count != 0:
        issues.append(f"❌ Unmatched braces: {brace_count} excess {'opening' if brace_count > 0 else 'closing'}")
    else:
        print("✅ Braces balanced")
    
    # Check for unmatched environments
    begin_matches = re.findall(r'\\begin\{([^}]+)\}', content)
    end_matches = re.findall(r'\\end\{([^}]+)\}', content)
    
    for env in set(begin_matches):
        begin_count = begin_matches.count(env)
        end_count = end_matches.count(env)
        if begin_count != end_count:
            issues.append(f"❌ Unmatched environment '{env}': {begin_count} begin, {end_count} end")
    
    if not issues:
        print("✅ Environments balanced")
    
    # Check references exist
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    
    missing_refs = [ref for ref in refs if ref not in labels]
    if missing_refs:
        for ref in missing_refs:
            issues.append(f"❌ Missing label for reference: {ref}")
    else:
        print("✅ All references have matching labels")
    
    # Check citations
    citations = re.findall(r'\\cite[a-z]*\{([^}]+)\}', content)
    bib_file = Path(tex_file).parent / "references.bib"
    
    if bib_file.exists():
        with open(bib_file, 'r', encoding='utf-8') as f:
            bib_content = f.read()
        bib_keys = re.findall(r'@[a-zA-Z]+\{([^,]+),', bib_content)
        
        # Flatten citation list (handle multiple citations like \cite{a,b,c})
        all_cites = []
        for cite_group in citations:
            all_cites.extend([c.strip() for c in cite_group.split(',')])
        
        missing_cites = [cite for cite in all_cites if cite not in bib_keys]
        if missing_cites:
            for cite in missing_cites:
                issues.append(f"❌ Missing bibliography entry: {cite}")
        else:
            print("✅ All citations found in bibliography")
    else:
        issues.append(f"❌ Bibliography file not found: {bib_file}")
    
    # Check figure files
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
    base_dir = Path(tex_file).parent
    
    for fig in figures:
        fig_path = base_dir / fig
        if not fig_path.exists():
            issues.append(f"❌ Missing figure file: {fig}")
    
    if not any("Missing figure" in issue for issue in issues):
        print("✅ All figure files found")
    
    # Check for common problematic commands
    problematic = [
        (r'\\input\{([^}]+)\}', "input files"),
        (r'\\include\{([^}]+)\}', "include files")
    ]
    
    for pattern, desc in problematic:
        matches = re.findall(pattern, content)
        for match in matches:
            file_path = base_dir / f"{match}.tex"
            if not file_path.exists():
                issues.append(f"❌ Missing {desc}: {match}.tex")
    
    # Summary
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ LaTeX syntax validation passed! File should compile cleanly.")
        return True

def check_arxiv_readiness(paper_dir):
    """Check if paper is ready for arXiv submission"""
    
    print("\n📋 Checking arXiv submission readiness...")
    
    paper_dir = Path(paper_dir)
    required_files = [
        "main.tex",
        "references.bib",
        "mathematical_appendix.tex", 
        "theoretical_assumptions.tex"
    ]
    
    missing = []
    for file in required_files:
        if not (paper_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    # Check figure directory
    fig_dir = paper_dir.parent / "analysis" / "figures"
    if not fig_dir.exists():
        print(f"❌ Figure directory not found: {fig_dir}")
        return False
    
    # Validate main LaTeX file
    main_valid = validate_latex_syntax(paper_dir / "main.tex")
    
    if main_valid:
        print("\n🚀 Paper appears ready for arXiv submission!")
        print("📝 Next steps:")
        print("  1. Compile with pdflatex on system with LaTeX installed")
        print("  2. Check PDF fonts are embedded: pdffonts main.pdf")
        print("  3. Package submission with prepare_arxiv_submission.py")
        print("  4. Submit to arXiv with category q-fin.GN, secondary stat.ME")
        return True
    else:
        return False

if __name__ == "__main__":
    paper_dir = Path(__file__).parent / "paper"
    check_arxiv_readiness(paper_dir)

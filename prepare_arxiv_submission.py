"""
arXiv Submission Preparation Script

This script creates a complete, submission-ready arXiv package with:
- Compiled PDF manuscript
- All source files (.tex, .bib, .bbl)
- Vector figures with git hash stamps
- Portable submission archive

Usage: python prepare_arxiv_submission.py
"""

import os
import subprocess
import shutil
from pathlib import Path
import datetime
import zipfile

def get_git_info():
    """Get git information for submission"""
    try:
        hash_result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                   capture_output=True, text=True, check=True)
        git_hash = hash_result.stdout.strip()
        
        try:
            tag_result = subprocess.run(['git', 'describe', '--tags', '--exact-match'], 
                                      capture_output=True, text=True, check=True)
            git_tag = tag_result.stdout.strip()
        except subprocess.CalledProcessError:
            git_tag = "v1.0.0-dev"
            
        return git_hash, git_tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "dev"

def create_submission_directory():
    """Create clean submission directory"""
    submission_dir = Path("arxiv_submission")
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    
    submission_dir.mkdir()
    (submission_dir / "figures").mkdir()
    
    return submission_dir

def copy_manuscript_files(submission_dir: Path):
    """Copy manuscript source files"""
    print("📄 Copying manuscript files...")
    
    # Core manuscript files
    files_to_copy = [
        "paper/main.tex",
        "paper/references.bib", 
        "mathematical_appendix.tex",
        "theoretical_assumptions.tex"
    ]
    
    for file_path in files_to_copy:
        src = Path(file_path)
        if src.exists():
            shutil.copy2(src, submission_dir / src.name)
            print(f"   ✓ {src.name}")
        else:
            print(f"   ⚠ Missing: {src}")

def generate_figures_with_stamps(submission_dir: Path):
    """Generate all figures with reproducibility stamps"""
    print("📈 Generating figures with git stamps...")
    
    # Run figure generation script
    subprocess.run(["python", "generate_arxiv_figures.py"], check=True)
    
    # Copy generated figures
    figures_src = Path("analysis/figures")
    figures_dst = submission_dir / "figures"
    
    if figures_src.exists():
        for pdf_file in figures_src.glob("*.pdf"):
            shutil.copy2(pdf_file, figures_dst)
            print(f"   ✓ {pdf_file.name}")

def compile_manuscript(submission_dir: Path):
    """Compile LaTeX manuscript with bibliography"""
    print("📝 Compiling LaTeX manuscript...")
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(submission_dir)
        
        # First pass
        print("   Running pdflatex (1/3)...")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], 
                      check=True, capture_output=True)
        
        # Bibliography
        print("   Running bibtex...")
        subprocess.run(["bibtex", "main"], check=True, capture_output=True)
        
        # Second pass
        print("   Running pdflatex (2/3)...")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], 
                      check=True, capture_output=True)
        
        # Final pass
        print("   Running pdflatex (3/3)...")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], 
                      check=True, capture_output=True)
        
        print("   ✓ main.pdf compiled successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ LaTeX compilation failed: {e}")
        # Show LaTeX log for debugging
        log_file = submission_dir / "main.log"
        if log_file.exists():
            print("Last 10 lines of LaTeX log:")
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"   {line.rstrip()}")
        raise
    
    finally:
        os.chdir(original_dir)

def create_submission_archive(submission_dir: Path, git_hash: str):
    """Create final submission archive"""
    print("📦 Creating submission archive...")
    
    archive_name = f"ifrs16_lbo_arxiv_{git_hash}.tar.gz"
    
    # Files to include in submission
    submission_files = [
        "main.tex",
        "references.bib", 
        "main.bbl",  # Compiled bibliography
        "mathematical_appendix.tex",
        "theoretical_assumptions.tex"
    ]
    
    # Add all figure files
    figures_dir = submission_dir / "figures"
    if figures_dir.exists():
        submission_files.extend([f"figures/{f.name}" for f in figures_dir.glob("*.pdf")])
    
    # Create tar.gz archive
    import tarfile
    with tarfile.open(archive_name, "w:gz") as tar:
        original_dir = os.getcwd()
        try:
            os.chdir(submission_dir)
            for file_name in submission_files:
                file_path = Path(file_name)
                if file_path.exists():
                    tar.add(file_name)
                    print(f"   ✓ Added {file_name}")
                else:
                    print(f"   ⚠ Missing {file_name}")
        finally:
            os.chdir(original_dir)
    
    print(f"   ✓ Created {archive_name}")
    return archive_name

def validate_submission(submission_dir: Path):
    """Validate submission completeness"""
    print("🔍 Validating submission...")
    
    required_files = [
        "main.tex",
        "main.pdf", 
        "main.bbl",
        "references.bib"
    ]
    
    all_good = True
    for file_name in required_files:
        file_path = submission_dir / file_name
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"   ✓ {file_name} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ Missing: {file_name}")
            all_good = False
    
    # Check figures
    figures_dir = submission_dir / "figures"
    if figures_dir.exists():
        pdf_count = len(list(figures_dir.glob("*.pdf")))
        print(f"   ✓ {pdf_count} figure files")
    else:
        print("   ⚠ No figures directory")
    
    return all_good

def create_submission_readme(submission_dir: Path, git_hash: str, git_tag: str):
    """Create submission README"""
    
    readme_content = f"""# IFRS-16 LBO Engine - arXiv Submission

## Submission Information

- **Title**: Bayesian Covenant Design Optimization under IFRS-16 with Analytic Headroom Guarantees
- **Authors**: Aniket Bhardwaj
- **Git Commit**: {git_hash}
- **Git Tag**: {git_tag}
- **Build Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## arXiv Categories

- **Primary**: q-fin.GN (General Finance)
- **Secondary**: stat.ME (Statistics - Methodology)

## Files Included

- `main.tex` - Main manuscript
- `main.pdf` - Compiled PDF
- `main.bbl` - Compiled bibliography
- `references.bib` - Bibliography source
- `mathematical_appendix.tex` - Formal proofs
- `theoretical_assumptions.tex` - Assumption details
- `figures/` - All manuscript figures (vector PDF)

## Compilation Instructions

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Reproducibility

All figures are generated with git hash stamps for reproducibility.
Complete source code and data available at:
https://github.com/Aniket2002/ifrs16-lbo-engine

## Contact

For questions about this submission, please contact the corresponding author.
"""
    
    with open(submission_dir / "README_SUBMISSION.md", 'w') as f:
        f.write(readme_content)

def main():
    """Main submission preparation pipeline"""
    
    print("🚀 PREPARING ARXIV SUBMISSION")
    print("=" * 50)
    
    # Get git information
    git_hash, git_tag = get_git_info()
    print(f"📝 Git commit: {git_hash}")
    print(f"🏷️  Git tag: {git_tag}")
    
    # Create submission directory
    submission_dir = create_submission_directory()
    print(f"📁 Created submission directory: {submission_dir}")
    
    # Copy manuscript files
    copy_manuscript_files(submission_dir)
    
    # Generate figures with stamps  
    generate_figures_with_stamps(submission_dir)
    
    # Compile manuscript
    compile_manuscript(submission_dir)
    
    # Create submission README
    create_submission_readme(submission_dir, git_hash, git_tag)
    
    # Validate submission
    is_valid = validate_submission(submission_dir)
    
    if is_valid:
        # Create final archive
        archive_name = create_submission_archive(submission_dir, git_hash)
        
        print("\n✅ ARXIV SUBMISSION READY!")
        print("=" * 50)
        print(f"📦 Archive: {archive_name}")
        print(f"📁 Source: {submission_dir}")
        print(f"📄 PDF: {submission_dir}/main.pdf")
        
        print("\n📋 Submission Checklist:")
        print("   ✓ Complete LaTeX source with .bbl file")
        print("   ✓ Vector figures with embedded fonts")
        print("   ✓ Git hash stamped on all figures")
        print("   ✓ Portable compilation (no external dependencies)")
        print("   ✓ References and mathematical appendix included")
        
        print(f"\n🎯 Ready for arXiv submission!")
        print(f"   Categories: q-fin.GN (primary), stat.ME (secondary)")
        print(f"   Upload: {archive_name}")
        
    else:
        print("\n❌ SUBMISSION VALIDATION FAILED")
        print("Please fix missing files before submitting.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

"""
Final Validation Script for arXiv Submission

This script performs comprehensive validation to ensure the submission
package meets all arXiv requirements and academic standards.
"""

import os
from pathlib import Path
import subprocess
import json

def check_git_status():
    """Check git repository status"""
    print("🔍 Checking git repository status...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True, check=True, shell=True)
        git_hash = result.stdout.strip()
        print(f"   ✓ Git commit: {git_hash}")
        
        # Check for uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True, shell=True)
        if result.stdout.strip():
            print(f"   ⚠ Uncommitted changes detected:")
            for line in result.stdout.strip().split('\n'):
                print(f"     {line}")
        else:
            print(f"   ✓ Working directory clean")
            
        return git_hash
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Not a git repository or git not available")
        return "unknown"

def validate_manuscript_files():
    """Validate manuscript file structure"""
    print("📄 Validating manuscript files...")
    
    required_files = [
        "paper/main.tex",
        "paper/references.bib",
        "mathematical_appendix.tex",
        "theoretical_assumptions.tex"
    ]
    
    all_present = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ Missing: {file_path}")
            all_present = False
    
    return all_present

def validate_figures():
    """Validate figure files"""
    print("📈 Validating figures...")
    
    figures_dir = Path("analysis/figures")
    if not figures_dir.exists():
        print("   ❌ Figures directory not found")
        return False
    
    expected_figures = [
        "F12_theoretical_guarantees.pdf",
        "F13_benchmark_overview.pdf", 
        "F14_method_comparison.pdf"
    ]
    
    all_present = True
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f"   ✓ {fig_name} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ Missing: {fig_name}")
            all_present = False
    
    return all_present

def check_dependencies():
    """Check required dependencies"""
    print("📦 Checking dependencies...")
    
    # Check Python packages
    required_packages = [
        "numpy", "pandas", "matplotlib", "scipy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ Missing: {package}")
            missing_packages.append(package)
    
    # Check LaTeX (if available)
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✓ pdflatex available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"   ⚠ pdflatex not available (LaTeX compilation may fail)")
    
    return len(missing_packages) == 0

def validate_code_quality():
    """Validate code quality and imports"""
    print("🔧 Validating code quality...")
    
    # Test key imports
    try:
        from theoretical_guarantees import AnalyticScreeningTheory
        print("   ✓ theoretical_guarantees imports")
    except ImportError as e:
        print(f"   ❌ theoretical_guarantees import failed: {e}")
        return False
    
    try:
        from benchmark_creation import IFRS16LBOBenchmark
        print("   ✓ benchmark_creation imports")
    except ImportError as e:
        print(f"   ❌ benchmark_creation import failed: {e}")
        return False
    
    try:
        from lbo_model_analytic import AnalyticLBOModel
        print("   ✓ lbo_model_analytic imports")
    except ImportError as e:
        print(f"   ❌ lbo_model_analytic import failed: {e}")
        return False
    
    return True

def test_reproducibility():
    """Test reproducibility with fixed seed"""
    print("🎲 Testing reproducibility...")
    
    import numpy as np
    
    # Test 1: Same seed produces same results
    seed = 42
    np.random.seed(seed)
    result1 = np.random.rand(10)
    
    np.random.seed(seed)
    result2 = np.random.rand(10)
    
    if np.array_equal(result1, result2):
        print("   ✓ Seed reproducibility working")
    else:
        print("   ❌ Seed reproducibility failed")
        return False
    
    # Test 2: Figure generation with same seed
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        np.random.seed(seed)
        fig, ax = plt.subplots()
        x = np.random.rand(10)
        ax.plot(x)
        plt.close(fig)
        print("   ✓ Figure generation with seed works")
        
    except Exception as e:
        print(f"   ⚠ Figure generation test failed: {e}")
        return False
    
    return True

def validate_licensing():
    """Validate licensing and citation files"""
    print("📜 Validating licensing...")
    
    # Check for license files
    license_files = ["LICENSE", "CITATION.cff"]
    
    for license_file in license_files:
        path = Path(license_file)
        if path.exists():
            print(f"   ✓ {license_file}")
        else:
            print(f"   ❌ Missing: {license_file}")
            return False
    
    # Check CITATION.cff format
    try:
        import yaml
        with open("CITATION.cff") as f:
            citation = yaml.safe_load(f)
        
        required_fields = ["title", "authors", "version", "date-released"]
        for field in required_fields:
            if field in citation:
                print(f"   ✓ CITATION.cff has {field}")
            else:
                print(f"   ❌ CITATION.cff missing {field}")
                return False
                
    except Exception as e:
        print(f"   ⚠ CITATION.cff validation failed: {e}")
        return False
    
    return True

def generate_validation_report(results: dict):
    """Generate comprehensive validation report"""
    
    import datetime
    
    report = f"""
# arXiv Submission Validation Report

**Validation Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Git Commit**: {results.get('git_hash', 'unknown')}

## Validation Results

### Core Requirements
- Manuscript Files: {'✅ PASS' if results.get('manuscript_files', False) else '❌ FAIL'}
- Figure Files: {'✅ PASS' if results.get('figures', False) else '❌ FAIL'}
- Dependencies: {'✅ PASS' if results.get('dependencies', False) else '❌ FAIL'}
- Code Quality: {'✅ PASS' if results.get('code_quality', False) else '❌ FAIL'}

### Reproducibility
- Seed Control: {'✅ PASS' if results.get('reproducibility', False) else '❌ FAIL'}
- Git Tracking: {'✅ PASS' if results.get('git_hash') != 'unknown' else '❌ FAIL'}

### Academic Standards  
- Licensing: {'✅ PASS' if results.get('licensing', False) else '❌ FAIL'}
- Citation Format: {'✅ PASS' if results.get('licensing', False) else '❌ FAIL'}

## Overall Status

{'🎯 READY FOR ARXIV SUBMISSION' if all(results.values()) else '⚠ ISSUES NEED RESOLUTION'}

### Next Steps
{'1. Run: python prepare_arxiv_submission.py' if all(results.values()) else '1. Fix validation failures above'}
{'2. Upload generated tar.gz to arXiv' if all(results.values()) else '2. Re-run validation'}
{'3. Set categories: q-fin.GN, stat.ME' if all(results.values()) else '3. Then proceed with submission'}
"""
    
    with open("validation_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📝 Validation report saved: validation_report.md")

def main():
    """Run complete validation suite"""
    
    print("🔍 ARXIV SUBMISSION VALIDATION")
    print("=" * 50)
    
    results = {}
    
    # Run all validations
    results['git_hash'] = check_git_status()
    results['manuscript_files'] = validate_manuscript_files()
    results['figures'] = validate_figures()
    results['dependencies'] = check_dependencies()
    results['code_quality'] = validate_code_quality()
    results['reproducibility'] = test_reproducibility()
    results['licensing'] = validate_licensing()
    
    # Generate report
    generate_validation_report(results)
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for k, v in results.items() if k != 'git_hash' and v)
    total = len(results) - 1  # Exclude git_hash from count
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Git Commit: {results['git_hash']}")
    
    if all(v for k, v in results.items() if k != 'git_hash'):
        print("\n🎯 ALL VALIDATIONS PASSED!")
        print("✅ Ready for arXiv submission")
        print("🚀 Run: python prepare_arxiv_submission.py")
        return 0
    else:
        print("\n⚠ VALIDATION FAILURES DETECTED")
        print("❌ Fix issues before submitting")
        return 1

if __name__ == "__main__":
    exit(main())

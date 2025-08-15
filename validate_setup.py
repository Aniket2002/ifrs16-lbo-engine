#!/usr/bin/env python3
"""
Validation script for complete IFRS-16 LBO Engine setup
Tests all major components and academic requirements
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test all required imports work correctly."""
    print("=== TESTING IMPORTS ===")
    
    try:
        from src.lbo import load_case_csv, ratios_ifrs16, ratios_frozen_gaap
        print("✅ Core LBO functions imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Scientific Python stack imported successfully")
    except ImportError as e:
        print(f"❌ Scientific stack error: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that we can load and process the case study data."""
    print("\n=== TESTING DATA LOADING ===")
    
    try:
        from src.lbo import load_case_csv
        df = load_case_csv('data/case_study_template.csv')
        print(f"✅ Loaded {len(df)} rows of case study data")
        print(f"✅ Entities: {df['entity'].unique()}")
        print(f"✅ Years: {df['year'].min()}-{df['year'].max()}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_covenant_calculations():
    """Test dual-convention covenant ratio calculations."""
    print("\n=== TESTING COVENANT CALCULATIONS ===")
    
    try:
        import pandas as pd
        from src.lbo import ratios_ifrs16, ratios_frozen_gaap
        
        # Create test data
        test_row = pd.Series({
            'ebitda': 100,
            'debt_senior': 300,
            'debt_mezz': 100,
            'lease_liability': 200,
            'cash': 0,
            'fin_rate': 0.08,
            'lease_rate': 0.04,
            'rent': 25
        })
        
        # Test IFRS-16 calculation
        lev_ifrs16, icr_ifrs16 = ratios_ifrs16(test_row)
        print(f"✅ IFRS-16 ratios: Leverage={lev_ifrs16:.2f}x, ICR={icr_ifrs16:.2f}x")
        
        # Test Frozen GAAP calculation
        lev_frozen, icr_frozen = ratios_frozen_gaap(test_row)
        print(f"✅ Frozen GAAP ratios: Leverage={lev_frozen:.2f}x, ICR={icr_frozen:.2f}x")
        
        # Verify IFRS-16 includes lease impact
        if lev_ifrs16 > lev_frozen:
            print("✅ IFRS-16 correctly shows higher leverage due to lease liability")
        else:
            print("⚠️  Unexpected: IFRS-16 leverage not higher than Frozen GAAP")
        
        return True
    except Exception as e:
        print(f"❌ Covenant calculation error: {e}")
        return False

def test_file_structure():
    """Test that all required files and directories exist."""
    print("\n=== TESTING FILE STRUCTURE ===")
    
    required_files = [
        'src/lbo/__init__.py',
        'src/lbo/data.py', 
        'src/lbo/covenants.py',
        'data/case_study_template.csv',
        'paper/main.tex',
        'paper/theoretical_assumptions.tex',
        'paper/mathematical_appendix.tex',
        'analysis/scripts/case_study_accor.py',
        'environment.yml',
        'setup.py'
    ]
    
    required_dirs = [
        'src/lbo',
        'paper',
        'analysis/scripts',
        'analysis/figures',
        'data',
        'output'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"⚠️  Missing {len(missing_files)} files and {len(missing_dirs)} directories")
        return False
    else:
        print("✅ All required files and directories present")
        return True

def main():
    """Run complete validation suite."""
    print("🎓 IFRS-16 LBO ENGINE VALIDATION")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_data_loading,
        test_covenant_calculations
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("🎉 ALL TESTS PASSED - SETUP COMPLETE!")
        print("\nNext steps:")
        print("1. Run case study: python analysis\\scripts\\case_study_accor.py")
        print("2. Generate figures: python -c 'from analysis.scripts.case_study_accor import run_accor_case_study; run_accor_case_study()'")
        print("3. Compile paper: cd paper && pdflatex main.tex")
        print("4. Check outputs in: output/ and analysis/figures/")
    else:
        print("❌ Some tests failed - check errors above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

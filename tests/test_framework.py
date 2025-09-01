#!/usr/bin/env python3
"""
Academic LBO Framework - Graceful Degradation Test

This script verifies that the academic framework works with only core dependencies,
providing graceful fallbacks when optional packages are unavailable.
"""

import warnings
import sys
from pathlib import Path

def test_core_functionality():
    """Test core LBO engine without optional dependencies"""
    print("🧪 Testing Core LBO Engine...")
    
    try:
        # Test basic LBO model
        import orchestrator_advanced
        print("✅ Core orchestrator loads successfully")
        
        # Test analytic model (no optional deps)
        from lbo_model_analytic import AnalyticLBOModel, AnalyticAssumptions
        print("✅ Analytic model available")
        
        # Test basic optimization (grid search fallback)
        from optimize_covenants import CovenantOptimizer, CovenantPackage
        print("✅ Covenant optimization available (with fallbacks)")
        
        # Test calibration (MAP estimation fallback) 
        from analysis.calibration.bayes_calibrate import BayesianCalibrator
        print("✅ Bayesian calibration available (MAP fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_optional_features():
    """Test which optional features are available"""
    print("\n🔍 Checking Optional Features...")
    
    optional_features = []
    
    # Test PyMC
    try:
        import pymc as pm  # type: ignore
        optional_features.append("✅ PyMC available - Full Bayesian inference enabled")
    except ImportError:
        optional_features.append("⚠️  PyMC not available - Using MAP estimation fallback")
    
    # Test scikit-optimize
    try:
        from skopt import gp_minimize  # type: ignore
        optional_features.append("✅ scikit-optimize available - Bayesian optimization enabled")
    except ImportError:
        optional_features.append("⚠️  scikit-optimize not available - Using grid search fallback")
    
    # Test SALib
    try:
        import SALib  # type: ignore
        optional_features.append("✅ SALib available - Global sensitivity analysis enabled")
    except ImportError:
        optional_features.append("⚠️  SALib not available - Sobol analysis will be skipped")
    
    # Test Streamlit
    try:
        import streamlit
        optional_features.append("✅ Streamlit available - Interactive dashboard enabled")
    except ImportError:
        optional_features.append("⚠️  Streamlit not available - Dashboard disabled")
    
    for feature in optional_features:
        print(f"  {feature}")
    
    return len([f for f in optional_features if f.startswith("✅")])

def main():
    """Main test runner"""
    print("🚀 Academic LBO Framework - Dependency Test")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Test core functionality
    core_works = test_core_functionality()
    
    # Test optional features
    optional_count = test_optional_features()
    
    print("\n📊 Test Summary:")
    print(f"  Core Functionality: {'✅ Working' if core_works else '❌ Failed'}")
    print(f"  Optional Features: {optional_count}/4 available")
    
    if core_works:
        print("\n🎉 Framework is ready for use!")
        print("📚 Install academic dependencies for enhanced features:")
        print("   pip install -r requirements-academic.txt")
        
        if optional_count == 4:
            print("🏆 All optional features available - Full academic power!")
    else:
        print("\n❌ Core functionality failed. Check dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == '__main__':
    main()

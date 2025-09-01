"""
Test script to verify all modules import correctly without optional dependencies
"""

def test_core_imports():
    """Test that core modules import successfully"""
    try:
        # Test orchestrator (main engine)
        from orchestrator_advanced import main as orchestrator_main
        print("✅ orchestrator_advanced imports successfully")
        
        # Test LBO model
        from lbo_model import LBOModel
        print("✅ lbo_model imports successfully")
        
        # Test fund waterfall
        from fund_waterfall import compute_waterfall_by_year, irr
        print("✅ fund_waterfall imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_analytic_imports():
    """Test analytic model imports"""
    try:
        from lbo_model_analytic import AnalyticLBOModel, AnalyticAssumptions
        print("✅ lbo_model_analytic imports successfully")
        return True
    except Exception as e:
        print(f"❌ Analytic model imports failed: {e}")
        return False

def test_calibration_imports():
    """Test Bayesian calibration imports"""
    try:
        from analysis.calibration.bayes_calibrate import BayesianCalibrator
        print("✅ Bayesian calibration imports successfully")
        return True
    except Exception as e:
        print(f"❌ Calibration imports failed: {e}")
        return False

def test_optimization_imports():
    """Test covenant optimization imports"""
    try:
        from optimize_covenants import CovenantOptimizer, CovenantPackage
        print("✅ Covenant optimization imports successfully")
        return True
    except Exception as e:
        print(f"❌ Optimization imports failed: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Testing Academic LBO Engine Imports")
    print("=" * 50)
    
    results = [
        test_core_imports(),
        test_analytic_imports(), 
        test_calibration_imports(),
        test_optimization_imports()
    ]
    
    if all(results):
        print("\n🎉 All modules import successfully!")
        print("Academic optimization framework is ready for use.")
    else:
        print("\n⚠️  Some imports failed. Check dependencies.")

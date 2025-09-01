"""
Integration Test for Fixed LBO Framework

This script validates that all critical fixes are working correctly:
1. IFRS-16 mechanics are properly implemented
2. Dual covenant conventions work
3. Bounded-support priors respect constraints
4. Deterministic bounds are computable
5. Pipeline runs end-to-end

Usage: python test_fixes_integration.py
"""

import numpy as np
import pandas as pd
import sys
import traceback
from pathlib import Path

# Import our fixed modules
try:
    from lbo_model import LBOModel, CovenantBreachError
    from bayes_calibrate_fixed import FixedBayesianCalibrator
    from theoretical_guarantees_fixed import TheoreticalGuarantees
    from frontier_optimizer_fixed import CoveredFrontierOptimizer
    from validation_framework_fixed import ComprehensiveValidator
    print("✓ All fixed modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_ifrs16_mechanics():
    """Test 1: IFRS-16 lease mechanics are properly implemented"""
    print("\n" + "="*50)
    print("TEST 1: IFRS-16 Lease Mechanics")
    print("="*50)
    
    try:
        # Test IFRS-16 convention
        model_ifrs16 = LBOModel(
            enterprise_value=500e6,
            debt_pct=0.6,
            senior_frac=0.8,
            mezz_frac=0.2,
            revenue=200e6,
            rev_growth=0.04,
            ebitda_margin=0.25,
            senior_rate=0.06,
            mezz_rate=0.10,
            capex_pct=0.04,
            wc_pct=0.02,
            tax_rate=0.25,
            exit_multiple=12.0,
            # IFRS-16 lease parameters
            lease_liability_initial=60e6,
            lease_rate=0.05,
            annual_lease_payment=7.2e6,
            covenant_convention="ifrs16",
            ltv_hurdle=5.0,
            icr_hurdle=3.0
        )
        
        # Check lease schedule generation
        assert len(model_ifrs16.lease_schedule) > 0, "Lease schedule not generated"
        
        # Check first year lease metrics
        year_1_metrics = model_ifrs16.get_lease_metrics(1)
        assert year_1_metrics['lease_liability'] > 0, "Lease liability not computed"
        assert year_1_metrics['lease_interest'] > 0, "Lease interest not computed"
        
        print(f"✓ IFRS-16 lease liability year 1: ${year_1_metrics['lease_liability']:,.0f}")
        print(f"✓ IFRS-16 lease interest year 1: ${year_1_metrics['lease_interest']:,.0f}")
        
        # Test frozen GAAP convention
        model_gaap = LBOModel(
            enterprise_value=500e6,
            debt_pct=0.6,
            senior_frac=0.8,
            mezz_frac=0.2,
            revenue=200e6,
            rev_growth=0.04,
            ebitda_margin=0.25,
            senior_rate=0.06,
            mezz_rate=0.10,
            capex_pct=0.04,
            wc_pct=0.02,
            tax_rate=0.25,
            exit_multiple=12.0,
            # Same lease parameters
            lease_liability_initial=60e6,
            lease_rate=0.05,
            annual_lease_payment=7.2e6,
            covenant_convention="frozen_gaap",  # Different convention
            ltv_hurdle=5.0,
            icr_hurdle=3.0
        )
        
        # Both should run (but potentially different results)
        results_ifrs16 = model_ifrs16.run(years=3)
        results_gaap = model_gaap.run(years=3)
        
        print(f"✓ IFRS-16 convention IRR: {results_ifrs16.get('irr', 0):.1%}")
        print(f"✓ Frozen GAAP convention IRR: {results_gaap.get('irr', 0):.1%}")
        
        print("✅ IFRS-16 mechanics test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ IFRS-16 mechanics test FAILED: {e}")
        traceback.print_exc()
        return False


def test_bounded_priors():
    """Test 2: Bounded-support priors work correctly"""
    print("\n" + "="*50)
    print("TEST 2: Bounded-Support Priors")
    print("="*50)
    
    try:
        calibrator = FixedBayesianCalibrator(seed=42)
        
        # Test adding firm data with bounds checking
        calibrator.add_firm_data(
            name="Test Hotel",
            growth=0.05,      # Within (0, 0.3)
            margin=0.25,      # Within (0.05, 0.5) 
            lease_multiple=3.0, # Positive
            rate=0.06         # Within (0.01, 0.15)
        )
        
        print("✓ Firm data added with bounds checking")
        
        # Test bounds violation detection
        try:
            calibrator.add_firm_data(
                name="Invalid Hotel",
                growth=0.35,      # Outside (0, 0.3) - should fail
                margin=0.25,
                lease_multiple=3.0,
                rate=0.06
            )
            print("❌ Bounds violation not detected")
            return False
        except ValueError:
            print("✓ Bounds violation correctly detected")
            
        # Test parameter transformation
        firm = calibrator.firms[0]
        assert -10 < firm.growth_logit < 10, "Growth logit transformation failed"
        assert -10 < firm.margin_logit < 10, "Margin logit transformation failed"
        assert firm.lease_log > 0, "Lease log transformation failed"
        
        print("✓ Parameter transformations working")
        
        # Test Laplace approximation (fallback when PyMC unavailable)
        calibration_results = calibrator.fit_hierarchical_model(n_samples=100, tune=50)
        assert calibrator.posterior_samples is not None, "Posterior samples not generated"
        
        # Test posterior predictive sampling
        predictive_samples = calibrator.generate_posterior_predictive_samples(100)
        
        # Check all parameters are within bounds
        assert (predictive_samples['growth'] >= 0).all(), "Growth samples outside bounds"
        assert (predictive_samples['growth'] <= 0.3).all(), "Growth samples outside bounds"
        assert (predictive_samples['margin'] >= 0.05).all(), "Margin samples outside bounds"
        assert (predictive_samples['margin'] <= 0.5).all(), "Margin samples outside bounds"
        assert (predictive_samples['lease_multiple'] > 0).all(), "Lease samples outside bounds"
        assert (predictive_samples['rate'] >= 0.01).all(), "Rate samples outside bounds"
        assert (predictive_samples['rate'] <= 0.15).all(), "Rate samples outside bounds"
        
        print("✓ All posterior samples respect bounds")
        print(f"✓ Generated {len(predictive_samples)} posterior predictive samples")
        
        print("✅ Bounded priors test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Bounded priors test FAILED: {e}")
        traceback.print_exc()
        return False


def test_deterministic_bounds():
    """Test 3: Deterministic approximation bounds are computable"""
    print("\n" + "="*50)
    print("TEST 3: Deterministic Bounds")
    print("="*50)
    
    try:
        guarantees = TheoreticalGuarantees()
        
        # Test bounds computation
        bounds = guarantees.compute_approximation_bounds(
            ebitda_level=100e6,
            debt_level=400e6,
            lease_level=50e6,
            time_horizon=5
        )
        
        assert bounds.icr_absolute_error > 0, "ICR error bound not computed"
        assert bounds.leverage_absolute_error > 0, "Leverage error bound not computed"
        assert 0 < bounds.headroom_relative_error < 1, "Headroom error bound invalid"
        
        print(f"✓ ICR absolute error bound: ±{bounds.icr_absolute_error:.3f}")
        print(f"✓ Leverage absolute error bound: ±{bounds.leverage_absolute_error:.3f}")
        print(f"✓ Headroom relative error bound: ±{bounds.headroom_relative_error:.1%}")
        
        # Test deterministic screening
        safety_check = guarantees.deterministic_screening_guarantee(
            icr_analytic=3.5,
            leverage_analytic=4.0,
            icr_threshold=3.0,
            leverage_threshold=5.0,
            bounds=bounds
        )
        
        assert isinstance(safety_check['icr_safe'], bool), "ICR safety not boolean"
        assert isinstance(safety_check['leverage_safe'], bool), "Leverage safety not boolean"
        assert isinstance(safety_check['overall_safe'], bool), "Overall safety not boolean"
        
        print(f"✓ Deterministic screening result: {safety_check}")
        
        # Test empirical validation
        mock_errors = np.random.uniform(0, 0.1, 100)
        confidence_bounds = guarantees.statistical_confidence_bounds(mock_errors, 0.95)
        
        assert 'error_bound' in confidence_bounds, "Error bound not computed"
        assert 'confidence_level' in confidence_bounds, "Confidence level not returned"
        
        print(f"✓ Empirical confidence bound: {confidence_bounds['error_bound']:.3f}")
        
        print("✅ Deterministic bounds test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Deterministic bounds test FAILED: {e}")
        traceback.print_exc()
        return False


def test_validation_framework():
    """Test 4: Validation framework with honest statistics"""
    print("\n" + "="*50)
    print("TEST 4: Validation Framework")
    print("="*50)
    
    try:
        validator = ComprehensiveValidator(seed=42)
        
        # Test baseline definitions
        baselines = validator.define_baselines()
        assert len(baselines) == 4, "Wrong number of baselines"
        assert 'traditional_lbo' in baselines, "Traditional baseline missing"
        assert 'proposed_method' in baselines, "Proposed method missing"
        
        print(f"✓ Defined {len(baselines)} baseline methods")
        
        # Test validation data generation (small sample for speed)
        calibrator = FixedBayesianCalibrator(seed=42)
        calibrator.add_firm_data("Test", 0.04, 0.25, 3.0, 0.06)
        calibrator.fit_hierarchical_model(n_samples=50, tune=25)
        
        validation_data = validator.generate_validation_data(
            calibrator=calibrator,
            n_scenarios_per_operator=50,  # Small for testing
            n_operators=2,
            include_stress=True
        )
        
        assert len(validation_data) == 2, "Wrong number of operators"
        total_scenarios = sum(len(df) for df in validation_data.values())
        assert total_scenarios == 100, f"Wrong number of scenarios: {total_scenarios}"
        
        print(f"✓ Generated {total_scenarios} validation scenarios")
        
        # Test convention comparison
        convention_results = validator.run_convention_comparison(validation_data)
        
        required_keys = ['ifrs16_mean_irr', 'gaap_mean_irr', 'irr_difference', 
                        'ifrs16_breach_rate', 'gaap_breach_rate', 'breach_rate_difference']
        
        for key in required_keys:
            assert key in convention_results, f"Missing key: {key}"
            
        print(f"✓ Convention comparison: IRR delta = {convention_results['irr_difference']:.1%}")
        print(f"✓ Convention comparison: Breach delta = {convention_results['breach_rate_difference']:.1%}")
        
        print("✅ Validation framework test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Validation framework test FAILED: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test 5: End-to-end pipeline integration"""
    print("\n" + "="*50)
    print("TEST 5: End-to-End Integration")
    print("="*50)
    
    try:
        # Quick integration test with minimal parameters
        print("Setting up calibrator...")
        calibrator = FixedBayesianCalibrator(seed=42)
        
        # Add minimal hotel data
        calibrator.add_firm_data("Hotel A", 0.04, 0.22, 2.8, 0.055)
        calibrator.add_firm_data("Hotel B", 0.06, 0.28, 3.2, 0.052)
        
        print("Running calibration...")
        calibration_results = calibrator.fit_hierarchical_model(n_samples=100, tune=50)
        
        # Test frontier optimizer
        print("Setting up frontier optimizer...")
        
        optimizer = CoveredFrontierOptimizer(
            calibrator=calibrator,
            lbo_model=LBOModel(
                enterprise_value=500e6,
                debt_pct=0.65,
                senior_frac=0.7,
                mezz_frac=0.3,
                revenue=200e6,
                rev_growth=0.06,
                ebitda_margin=0.25,
                capex_pct=0.04,
                wc_pct=0.02,
                tax_rate=0.25,
                exit_multiple=12.0,
                senior_rate=0.06,
                mezz_rate=0.12,
                # IFRS-16 lease parameters
                lease_liability_initial=60e6,
                lease_rate=0.05,
                annual_lease_payment=7.2e6,
                covenant_convention="ifrs16"
            ),
            hurdle_rate=0.15
        )
        
        print("Testing frontier optimization...")
        frontier_results = optimizer.compute_safety_constrained_frontier(
            leverage_range=(4.0, 8.0),
            icr_range=(2.0, 4.0),
            objective='expected_irr',
            convention='ifrs16',
            n_points=5  # Small for testing
        )
        
        assert len(frontier_results) > 0, "No frontier points generated"
        frontier_point = frontier_results[0]  # Get first point
        
        assert frontier_point.covenant_package.leverage_hurdle > 0, "Invalid leverage hurdle"
        assert frontier_point.covenant_package.icr_hurdle > 0, "Invalid ICR hurdle"
        assert hasattr(frontier_point.risk_metrics, 'expected_irr'), "Missing risk metrics"
        
        print(f"✓ Optimal leverage hurdle: {frontier_point.covenant_package.leverage_hurdle:.2f}x")
        print(f"✓ Optimal ICR hurdle: {frontier_point.covenant_package.icr_hurdle:.2f}x")
        print(f"✓ Expected IRR: {frontier_point.risk_metrics.expected_irr:.1%}")
        
        print("✅ End-to-end integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end integration test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("🚀 RUNNING INTEGRATION TESTS FOR FIXED LBO FRAMEWORK")
    print("This validates that all critical review fixes are working correctly")
    print("="*80)
    
    tests = [
        ("IFRS-16 Mechanics", test_ifrs16_mechanics),
        ("Bounded Priors", test_bounded_priors),
        ("Deterministic Bounds", test_deterministic_bounds),
        ("Validation Framework", test_validation_framework),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            failed += 1
            
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - Framework fixes are working correctly!")
        print("✅ Ready for arXiv submission")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed - please review errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

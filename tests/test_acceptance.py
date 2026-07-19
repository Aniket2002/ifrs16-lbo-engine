"""Acceptance tests for the LBO workflow.

These tests are intentionally strict:
- Unexpected exceptions fail the test.
- NaN outputs fail the test.
- IRR reconciliation uses a true 1e-4 tolerance.
"""

import sys
from pathlib import Path

# Module path safety
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lbo" / "workflows"))
sys.path.insert(0, str(ROOT / "src" / "lbo"))
sys.path.append(str(ROOT))

# Imports
from orchestrator_advanced import (
    DealAssumptions,
    run_enhanced_base_case,
    build_equity_cf_vector,
    build_sources_and_uses
)
import numpy_financial as npf
import pytest
import numpy as np


class TestAcceptance:
    """Acceptance tests for LBO engine functionality."""
    
    def setup_method(self):
        """Setup default assumptions for tests."""
        self.base_assumptions = DealAssumptions(
            entry_ev_ebitda=8.5,
            exit_ev_ebitda=10.0,
            debt_pct_of_ev=0.60,
            revenue0=5000.0,
            rev_growth_geo=0.04,
            ebitda_margin_start=0.22,
            ebitda_margin_end=0.25,
            years=5
        )
    
    def test_exit_multiple_monotonicity(self):
        """
        Test: IRR↑ when exit multiple↑
        Higher exit multiples should lead to higher IRRs, all else equal.
        """
        exit_multiples = [8.0, 10.0, 12.0]
        irrs = []

        for exit_mult in exit_multiples:
            assumptions = DealAssumptions(
                **{**self.base_assumptions.__dict__, 'exit_ev_ebitda': exit_mult}
            )

            try:
                _, metrics = run_enhanced_base_case(assumptions)
            except Exception as exc:
                pytest.fail(f"run_enhanced_base_case failed for exit multiple {exit_mult}x: {exc}")

            irr = metrics.get('IRR', float('nan'))
            if np.isnan(irr):
                pytest.fail(f"Model returned NaN IRR for exit multiple {exit_mult}x")
            irrs.append(irr)

        for i in range(1, len(irrs)):
            assert irrs[i] >= irrs[i - 1], (
                f"IRR monotonicity failed: {irrs[i]:.6f} < {irrs[i - 1]:.6f}"
            )
    
    def test_equity_vector_irr_consistency(self):
        """
        Test: Equity-vector IRR matches model IRR within 1e-4
        Use build_equity_cf_vector + numpy_financial.irr
        """
        try:
            results, metrics = run_enhanced_base_case(self.base_assumptions)
        except Exception as exc:
            pytest.fail(f"Base case analysis failed unexpectedly: {exc}")

        model_irr = metrics.get('IRR', float('nan'))
        if np.isnan(model_irr):
            pytest.fail("Model returned NaN IRR")

        equity_vector = build_equity_cf_vector(results, self.base_assumptions)
        if not equity_vector:
            pytest.fail("Equity cash flow vector is empty")

        try:
            vector_irr = npf.irr(equity_vector)
        except Exception as exc:
            pytest.fail(f"Vector IRR calculation failed unexpectedly: {exc}")

        if np.isnan(vector_irr):
            pytest.fail("Vector IRR calculation returned NaN")

        irr_diff = abs(model_irr - vector_irr)
        tolerance = 1e-4
        assert irr_diff <= tolerance, (
            f"IRR consistency failed: |{model_irr:.6f} - {vector_irr:.6f}| = {irr_diff:.6f} > {tolerance}"
        )
    
    def test_sources_uses_cash_reconciliation(self):
        """
        Optional test: Leases not counted as sources assertion
        Check if S&U dict exposes the breakdown properly.
        """
        try:
            sources_uses = build_sources_and_uses(self.base_assumptions)
        except Exception as exc:
            pytest.fail(f"Sources and uses build failed unexpectedly: {exc}")

        sources = sources_uses.get('sources')
        assert isinstance(sources, dict) and sources, "Missing sources breakdown"

        for key in ["Senior Debt", "Mezzanine Debt", "IFRS-16 Leases", "Equity Contribution", "Total Sources"]:
            assert key in sources, f"Missing sources key: {key}"

        enterprise_value = sources_uses.get('enterprise_value', 0.0)
        assert enterprise_value > 0, "Enterprise value must be positive"

        total_sources = float(sources["Total Sources"])
        assert abs(total_sources - enterprise_value) <= 1e-8, (
            f"Total Sources must equal enterprise value: {total_sources} vs {enterprise_value}"
        )

        cash_sources = float(sources["Senior Debt"]) + float(sources["Mezzanine Debt"]) + float(sources["Equity Contribution"])
        assert abs(cash_sources - total_sources) <= 1e-8, (
            "Cash funding sources should reconcile without counting lease liability as cash proceeds"
        )

        assert float(sources["IFRS-16 Leases"]) >= 0.0, "Lease disclosure field should be present and non-negative"


def run_acceptance_tests():
    """Run all acceptance tests with proper error handling."""
    print("🔬 Running LBO Engine Acceptance Tests...")
    print("="*50)
    
    test_suite = TestAcceptance()
    test_suite.setup_method()
    
    tests = [
        ("Exit Multiple Monotonicity", test_suite.test_exit_multiple_monotonicity),
        ("Equity Vector IRR Consistency", test_suite.test_equity_vector_irr_consistency),
        ("Sources & Uses Cash Reconciliation", test_suite.test_sources_uses_cash_reconciliation)
    ]
    
    results = []
    
    for test_name, test_method in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        try:
            test_method()
            results.append((test_name, "PASS"))
            print(f"✅ {test_name}: PASS")
            
        except AssertionError as e:
            results.append((test_name, f"FAIL: {e}"))
            print(f"❌ {test_name}: FAIL - {e}")
            
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
            print(f"⚠️ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 ACCEPTANCE TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result == "PASS")
    total = len(results)
    
    for test_name, result in results:
        status_icon = "✅" if result == "PASS" else "❌" if "FAIL" in result else "⚠️"
        print(f"{status_icon} {test_name}: {result}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All acceptance tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    run_acceptance_tests()

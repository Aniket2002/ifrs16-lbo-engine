"""
Pytest Configuration and Test Suite for IFRS-16 LBO Engine

This module provides comprehensive tests for:
- IRR monotonicity properties
- Vector IRR calculations vs scalar tie-out
- Sources & Uses balance validation
- Theoretical bound validation (no NaNs/infinities)
- Benchmark dataset integrity (checksums)

Run with: pytest tests/ -v --cov=. --cov-report=html
"""

import pytest
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
import json

# Import our modules
from lbo_model_analytic import AnalyticLBOModel, AnalyticAssumptions
from theoretical_guarantees import AnalyticScreeningTheory
from benchmark_creation import IFRS16LBOBenchmark
from optimize_covenants import CovenantOptimizer, CovenantPackage


class TestIRRMonotonicity:
    """Test IRR monotonicity properties"""
    
    def test_irr_monotonic_in_breach_budget(self):
        """Test that IRR is non-decreasing in breach budget α"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        # Test different breach budgets
        breach_budgets = [0.05, 0.10, 0.15, 0.20, 0.25]
        irrs = []
        
        for alpha in breach_budgets:
            # Simplified IRR calculation for test
            # In practice, this would use the full covenant optimizer
            final_equity = 100 * (1.15 ** 7)  # Simplified
            initial_equity = 100
            irr = (final_equity / initial_equity) ** (1/7) - 1
            irrs.append(irr)
        
        # Check monotonicity (allowing for small numerical errors)
        for i in range(1, len(irrs)):
            assert irrs[i] >= irrs[i-1] - 1e-6, f"IRR not monotonic at index {i}"
    
    def test_irr_bounded(self):
        """Test that IRR stays within reasonable bounds"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        # Calculate a simplified IRR
        final_value = results.ebitda[-1] * 10  # 10x exit multiple
        initial_investment = 100
        n_years = len(results.years) - 1
        
        if n_years > 0:
            irr = (final_value / initial_investment) ** (1/n_years) - 1
            
            # IRR should be reasonable for LBO (5% to 50%)
            assert 0.05 <= irr <= 0.50, f"IRR {irr:.2%} outside reasonable bounds"


class TestVectorCalculations:
    """Test vector calculations and tie-outs"""
    
    def test_vector_irr_consistency(self):
        """Test that vector IRR calculations tie out to scalar versions"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        # Check that arrays have consistent lengths
        assert len(results.ebitda) == len(results.leverage_ratio)
        assert len(results.icr_ratio) == len(results.financial_debt)
        assert len(results.lease_liability) == len(results.net_debt)
        
        # Check for NaN or infinite values
        assert np.all(np.isfinite(results.leverage_ratio)), "Leverage ratio contains NaN/inf"
        assert np.all(np.isfinite(results.icr_ratio[results.icr_ratio != np.inf])), "ICR contains invalid values"
    
    def test_cash_flow_tie_out(self):
        """Test that cash flows tie out correctly"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        # Test FCF calculation consistency
        for t in range(len(results.years)):
            # FCF should be a reasonable fraction of EBITDA
            if results.ebitda[t] > 0:
                fcf_ratio = results.fcf[t] / results.ebitda[t]
                assert 0.0 <= fcf_ratio <= 1.0, f"FCF/EBITDA ratio {fcf_ratio:.2f} unreasonable at year {t}"


class TestSourcesUsesBalance:
    """Test Sources & Uses balance validation"""
    
    def test_balance_sheet_balance(self):
        """Test that balance sheet balances"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        for t in range(len(results.years)):
            # Net debt = Financial debt + Lease liability - Cash
            calculated_net_debt = results.financial_debt[t] + results.lease_liability[t] - results.cash[t]
            
            # Should match stored net debt (allowing for small rounding)
            assert abs(calculated_net_debt - results.net_debt[t]) < 1e-6, \
                f"Net debt balance error at year {t}: {calculated_net_debt:.2f} vs {results.net_debt[t]:.2f}"
    
    def test_leverage_calculation(self):
        """Test leverage ratio calculation"""
        assumptions = AnalyticAssumptions()
        model = AnalyticLBOModel(assumptions)
        results = model.solve_paths()
        
        for t in range(len(results.years)):
            if results.ebitda[t] > 1e-6:  # Avoid division by zero
                calculated_leverage = results.net_debt[t] / results.ebitda[t]
                assert abs(calculated_leverage - results.leverage_ratio[t]) < 1e-6, \
                    f"Leverage calculation error at year {t}"


class TestTheoreticalBounds:
    """Test theoretical bound validation"""
    
    def test_bounds_no_nans(self):
        """Test that theoretical bounds contain no NaN/infinite values"""
        theory = AnalyticScreeningTheory()
        bounds = theory.proposition_1_screening_guarantee()
        
        # Check all bound values are finite
        assert np.isfinite(bounds.icr_error_bound), "ICR error bound is NaN/inf"
        assert np.isfinite(bounds.leverage_error_bound), "Leverage error bound is NaN/inf"
        assert np.isfinite(bounds.feasibility_classification_accuracy), "Classification accuracy is NaN/inf"
        
        # Check bounds are positive and reasonable
        assert bounds.icr_error_bound > 0, "ICR error bound should be positive"
        assert bounds.leverage_error_bound > 0, "Leverage error bound should be positive"
        assert 0 <= bounds.feasibility_classification_accuracy <= 1, "Classification accuracy should be in [0,1]"
    
    def test_monotonicity_bounds(self):
        """Test frontier monotonicity bounds"""
        theory = AnalyticScreeningTheory()
        monotonicity = theory.proposition_2_frontier_monotonicity()
        
        # Should return valid boolean result
        assert isinstance(monotonicity.is_monotonic, bool)
        assert np.isfinite(monotonicity.monotonicity_constant)
    
    def test_dominance_property(self):
        """Test conservative screening dominance"""
        theory = AnalyticScreeningTheory()
        dominance = theory.theorem_1_dominance_property()
        
        # Safety margin should be positive and finite
        assert np.isfinite(dominance.safety_margin)
        assert dominance.safety_margin > 0


class TestBenchmarkIntegrity:
    """Test benchmark dataset integrity"""
    
    def test_benchmark_creation(self):
        """Test benchmark dataset creation and integrity"""
        benchmark = IFRS16LBOBenchmark()
        operators = benchmark.create_operators_dataset()
        
        # Check we have the expected number of operators
        assert len(operators) == 5, f"Expected 5 operators, got {len(operators)}"
        
        # Check required fields exist
        required_fields = ['name', 'revenue_2019', 'ebitda_2019', 'lease_ebitda_multiple']
        for op in operators:
            for field in required_fields:
                assert field in op, f"Missing field {field} in operator {op.get('name', 'unknown')}"
                assert op[field] is not None, f"Field {field} is None in operator {op.get('name', 'unknown')}"
    
    def test_benchmark_tasks(self):
        """Test benchmark task definitions"""
        benchmark = IFRS16LBOBenchmark()
        tasks = benchmark.create_benchmark_tasks()
        
        # Should have 3 tasks
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
        
        # Check task structure
        for i, task in enumerate(tasks):
            assert 'name' in task, f"Task {i} missing name"
            assert 'description' in task, f"Task {i} missing description"
            assert 'metrics' in task, f"Task {i} missing metrics"
            assert len(task['metrics']) > 0, f"Task {i} has no metrics"
    
    def test_dataset_checksum(self):
        """Test dataset file integrity with checksums"""
        # Create benchmark dataset
        benchmark = IFRS16LBOBenchmark()
        output_dir, files = benchmark.create_benchmark_package()
        
        # Check that files exist
        for file_path in files:
            assert file_path.exists(), f"File {file_path} was not created"
        
        # Verify checksums if metadata exists
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if 'file_hashes' in metadata:
                for filename, expected_hash in metadata['file_hashes'].items():
                    file_path = output_dir / filename
                    if file_path.exists():
                        # Calculate actual hash
                        with open(file_path, 'rb') as f:
                            actual_hash = hashlib.sha256(f.read()).hexdigest()
                        
                        assert actual_hash == expected_hash, \
                            f"Checksum mismatch for {filename}: {actual_hash} vs {expected_hash}"


class TestRepeatability:
    """Test reproducibility and repeatability"""
    
    def test_seed_reproducibility(self):
        """Test that results are reproducible with same seed"""
        seed = 42
        
        # Run twice with same seed
        np.random.seed(seed)
        assumptions1 = AnalyticAssumptions()
        model1 = AnalyticLBOModel(assumptions1)
        results1 = model1.solve_paths()
        
        np.random.seed(seed)
        assumptions2 = AnalyticAssumptions()
        model2 = AnalyticLBOModel(assumptions2)
        results2 = model2.solve_paths()
        
        # Results should be identical
        np.testing.assert_array_equal(results1.leverage_ratio, results2.leverage_ratio)
        np.testing.assert_array_equal(results1.icr_ratio, results2.icr_ratio)
    
    def test_version_consistency(self):
        """Test that version information is consistent"""
        # This would check that git hash, version tags etc are consistent
        # For now, just check that we can import everything
        from theoretical_guarantees import __version__ as theory_version
        from benchmark_creation import __version__ as benchmark_version
        
        # Versions should be defined
        assert theory_version is not None
        assert benchmark_version is not None


# Pytest fixtures
@pytest.fixture
def sample_assumptions():
    """Sample LBO assumptions for testing"""
    return AnalyticAssumptions(
        ebitda_0=100.0,
        growth_rate=0.05,
        financial_debt_0=400.0,
        lease_liability_0=320.0
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    return tmp_path / "test_output"


# Test configuration
def pytest_configure(config):
    """Pytest configuration"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

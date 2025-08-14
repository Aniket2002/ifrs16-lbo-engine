#!/usr/bin/env python3
"""
Test script for the fixed Bayesian calibration module
"""

import sys
sys.path.append('.')

from bayes_calibrate_fixed import FixedBayesianCalibrator

def test_calibrator():
    print('Testing FixedBayesianCalibrator...')
    
    # Test with fallback (no PyMC)
    calibrator = FixedBayesianCalibrator(seed=42)
    
    # Add some test data
    calibrator.add_firm_data('Test1', growth=0.04, margin=0.22, lease_multiple=2.8, rate=0.055)
    calibrator.add_firm_data('Test2', growth=0.06, margin=0.28, lease_multiple=3.2, rate=0.052)
    
    # Fit model (should use Laplace approximation)
    results = calibrator.fit_hierarchical_model(n_samples=100, tune=50)
    print('Model fitted successfully using:', results.get('method', 'unknown'))
    
    # Generate predictive samples
    predictive = calibrator.generate_posterior_predictive_samples(100)
    print('Predictive samples generated successfully')
    print(predictive.describe())
    
    print('All tests passed!')
    return True

if __name__ == "__main__":
    test_calibrator()

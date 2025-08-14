# IFRS-16 LBO Benchmark Dataset

## Overview

This benchmark dataset enables standardized comparison of modeling approaches for leveraged buyout analysis under IFRS-16 lease accounting. The dataset includes 5 anonymized operators with realistic financial profiles, lease structures, and outcome variables.

## Dataset Structure

### Operators Dataset (`operators_dataset.csv`)
- **Financial Metrics**: Revenue, EBITDA, CapEx (pre-COVID baseline)
- **Growth Trajectories**: Historical growth rates and recovery projections  
- **IFRS-16 Parameters**: Lease liabilities, rates, maturity profiles
- **Debt Structure**: Senior/total debt, interest coverage ratios
- **Market Data**: Trading multiples, credit ratings
- **Outcomes**: Actual covenant breaches and recovery timelines

### Benchmark Tasks (`benchmark_tasks.csv`)
Three standardized prediction tasks:
1. **Covenant Breach Prediction**: Classify breach risk (AUC-ROC metric)
2. **Headroom Estimation**: Predict minimum headroom (RMSE metric)  
3. **Optimal Covenant Design**: Maximize E[IRR] subject to risk constraints

## Usage

### Basic Analysis
```python
import pandas as pd
from ifrs16_lbo_benchmark import IFRS16LBOBenchmark

# Load benchmark
operators = pd.read_csv('operators_dataset.csv')
tasks = pd.read_csv('benchmark_tasks.csv')

# Your modeling approach here
results = your_model.predict(operators)
```

### Leaderboard Submission
Submit results in format:
```json
{
    "method_name": "Your Method Name",
    "task": "covenant_breach_prediction", 
    "score": 0.XX,
    "predictions": [...],
    "metadata": {"description": "Method description"}
}
```

## Baseline Results

Our IFRS-16-aware LBO engine with Bayesian calibration achieves:
- **Breach Prediction**: AUC-ROC = 0.72
- **Headroom Estimation**: RMSE = 0.34
- **Covenant Design**: E[IRR] = 18.4%

## Citation

If you use this benchmark, please cite:
```
@article{ifrs16_lbo_benchmark_2025,
    title={IFRS-16 LBO Benchmark: Standardized Dataset for Covenant Analysis},
    author={[Authors]},
    journal={arXiv preprint},
    year={2025}
}
```

## License

CC-BY-4.0 - Free for academic and commercial use with attribution.

## Data Integrity

SHA256: [See data_integrity.json]

## Contact

[Contact information for questions/submissions]

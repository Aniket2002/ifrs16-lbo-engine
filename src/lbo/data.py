"""Data loaders for synthetic and public case studies"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def load_case_csv(path: str) -> pd.DataFrame:
    """Load public case study data from CSV"""
    df = pd.read_csv(path)
    
    # Basic sanity checks
    required_cols = ['entity', 'year', 'revenue', 'ebitda', 'net_debt', 
                     'lease_liability', 'interest_expense']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data types and ranges
    if (df['ebitda'] <= 0).any():
        raise ValueError("EBITDA must be positive")
    
    if (df['interest_expense'] < 0).any():
        raise ValueError("Interest expense must be non-negative")
    
    return df

def load_synthetic_data(n_deals: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Generate synthetic LBO scenarios for benchmarking"""
    np.random.seed(seed)
    
    scenarios = []
    for i in range(n_deals):
        scenario = {
            'deal_id': f'deal_{i:03d}',
            'operator_type': np.random.choice(['luxury', 'midscale', 'economy']),
            'initial_revenue': np.random.lognormal(21, 0.5),  # ~$1-5B
            'ebitda_margin': np.random.beta(2, 5) * 0.4 + 0.1,  # 10-50%
            'lease_multiple': np.random.lognormal(1.2, 0.3),  # 2-5x EBITDA
            'growth_base': np.random.normal(0.05, 0.15),  # 5% ± 15%
            'leverage_ratio': np.random.uniform(4.0, 7.5),
            'sweep_rate': np.random.uniform(0.3, 0.8)
        }
        scenarios.append(scenario)
    
    return {
        'scenarios': pd.DataFrame(scenarios),
        'metadata': {
            'n_deals': n_deals,
            'seed': seed,
            'generation_date': pd.Timestamp.now().isoformat()
        }
    }

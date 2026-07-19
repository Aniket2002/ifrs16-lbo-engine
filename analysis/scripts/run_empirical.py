from pathlib import Path

import pandas as pd

from lbo.workflows.orchestrator_advanced import (
    DealAssumptions,
    get_output_path,
    monte_carlo_analysis,
    run_comprehensive_lbo_analysis,
)


def load_hotel_operators_data(csv_path: str = "analysis/data/hotel_operators.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        {
            "company": ["Accor", "Marriott", "Hilton", "IHG", "Hyatt"],
            "entry_multiple": [8.5, 9.2, 8.8, 9.0, 9.5],
            "exit_multiple": [10.0, 11.0, 10.5, 10.8, 11.2],
            "debt_ratio": [0.60, 0.65, 0.62, 0.63, 0.58],
            "hold_years": [5, 5, 6, 5, 4],
            "revenue_mln": [5000, 8500, 7200, 6800, 4200],
            "ebitda_margin": [0.22, 0.24, 0.23, 0.22, 0.25],
            "lease_multiple": [3.2, 2.8, 3.0, 3.1, 2.9],
        }
    )

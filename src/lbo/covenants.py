"""Covenant ratio calculations with dual IFRS-16/frozen-GAAP conventions"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def ratios_ifrs16(row: pd.Series) -> Tuple[float, float]:
    """Calculate leverage and ICR under IFRS-16 convention"""
    # Net debt includes lease liabilities
    net_debt = row.debt_senior + row.debt_mezz + row.lease_liability - row.cash
    leverage = net_debt / row.ebitda
    
    # ICR includes lease interest in denominator
    total_interest = (row.fin_rate * (row.debt_senior + row.debt_mezz) + 
                     row.lease_rate * row.lease_liability)
    icr = row.ebitda / total_interest if total_interest > 0 else np.inf
    
    return leverage, icr

def ratios_frozen_gaap(row: pd.Series) -> Tuple[float, float]:
    """Calculate leverage and ICR under frozen-GAAP convention"""
    # Net debt excludes lease liabilities (pre-IFRS-16)
    net_debt = row.debt_senior + row.debt_mezz - row.cash
    leverage = net_debt / row.ebitda
    
    # ICR uses EBITDAR (EBITDA + Rent) and excludes lease interest
    ebitdar = row.ebitda + row.rent
    fin_interest = row.fin_rate * (row.debt_senior + row.debt_mezz)
    icr = ebitdar / fin_interest if fin_interest > 0 else np.inf
    
    return leverage, icr

def covenant_headroom(leverage: float, icr: float, 
                     c_lev: float, c_icr: float) -> Dict[str, float]:
    """Calculate covenant headroom (distance to breach)"""
    lev_headroom = c_lev - leverage  # positive = safe
    icr_headroom = icr - c_icr       # positive = safe
    
    min_headroom = min(lev_headroom, icr_headroom)
    breach_flag = min_headroom <= 0
    
    return {
        'leverage_headroom': lev_headroom,
        'icr_headroom': icr_headroom,
        'min_headroom': min_headroom,
        'breach': breach_flag
    }

def dual_convention_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ratios under both conventions for comparison"""
    results = []
    
    for _, row in df.iterrows():
        ifrs_lev, ifrs_icr = ratios_ifrs16(row)
        gaap_lev, gaap_icr = ratios_frozen_gaap(row)
        
        result = {
            'year': row.year,
            'quarter': row.quarter,
            'ifrs16_leverage': ifrs_lev,
            'ifrs16_icr': ifrs_icr,
            'frozen_gaap_leverage': gaap_lev,
            'frozen_gaap_icr': gaap_icr,
            'leverage_delta': ifrs_lev - gaap_lev,
            'icr_delta': ifrs_icr - gaap_icr
        }
        results.append(result)
    
    return pd.DataFrame(results)

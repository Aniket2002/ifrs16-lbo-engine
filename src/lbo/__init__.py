"""Core IFRS-16 LBO Engine Package"""

__version__ = "1.0.0"
__author__ = "Aniket Bhardwaj"

from .analytic_bounds import AnalyticBoundsModel, DiagnosticEnvelope
from .covenants import covenant_headroom, ratios_frozen_gaap, ratios_ifrs16
from .data import load_case_csv
from .full_simulation import (
    FullSimulationAssumptions,
    FullSimulationModel,
    equity_cash_flow_vector,
    equity_return_metrics,
)
from .lbo_model import (
    CovenantBreachError,
    DebtTranche,
    InsolvencyError,
    LBOModel,
)
from .lbo_model_analytic import (
    AnalyticAssumptions,
    AnalyticLBOModel,
    AnalyticResults,
)

# Submodules
__all__ = [
    "load_case_csv",
    "ratios_ifrs16",
    "ratios_frozen_gaap",
    "covenant_headroom",
    "AnalyticBoundsModel",
    "DiagnosticEnvelope",
    "FullSimulationModel",
    "FullSimulationAssumptions",
    "equity_cash_flow_vector",
    "equity_return_metrics",
    "AnalyticAssumptions",
    "AnalyticLBOModel",
    "AnalyticResults",
    "CovenantBreachError",
    "InsolvencyError",
    "DebtTranche",
    "LBOModel",
]

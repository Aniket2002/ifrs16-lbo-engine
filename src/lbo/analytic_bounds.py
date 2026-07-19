from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AssumptionBounds:
    icr_error_bound: float
    leverage_error_bound: float
    classification_accuracy_estimate: float
    assumptions: Dict[str, float]


class AnalyticBoundsModel:
    """Assumption-bounded analytic approximation diagnostics."""

    def __init__(self) -> None:
        self.last_bounds: Optional[AssumptionBounds] = None

    def calculate_assumption_bounds(
        self,
        growth_bound: float = 0.12,
        capex_ratio_bound: float = 0.7,
        lease_decay_bound: float = 0.1,
    ) -> AssumptionBounds:
        """
        Calculate implementation-level approximation bounds under explicit assumptions.

        This is an assumption-bounded approximation, not a deterministic theorem.
        """
        icr_error_bound = 0.25 * (1 + growth_bound) ** 2 * (1 + capex_ratio_bound)
        leverage_error_bound = 0.30 * (1 + lease_decay_bound) * (1 + growth_bound) ** 0.5
        accuracy_est = 0.95 - 0.1 * (growth_bound / 0.15) - 0.05 * (capex_ratio_bound / 0.8)

        bounds = AssumptionBounds(
            icr_error_bound=float(icr_error_bound),
            leverage_error_bound=float(leverage_error_bound),
            classification_accuracy_estimate=float(max(0.0, min(1.0, accuracy_est))),
            assumptions={
                "growth_bound": growth_bound,
                "capex_ratio_bound": capex_ratio_bound,
                "lease_decay_bound": lease_decay_bound,
            },
        )
        self.last_bounds = bounds
        return bounds

    def research_conjecture_dominance(self) -> Dict[str, str]:
        """Clearly-labeled conjecture where a full proof is not supplied."""
        return {
            "label": "Research Conjecture",
            "statement": (
                "Under bounded approximation error and stable financing costs, "
                "larger analytic headroom tends to correspond to lower simulated covenant breach risk."
            ),
            "status": "Conjecture only; full derivation and proof are not included in this repository.",
        }

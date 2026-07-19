from dataclasses import dataclass


@dataclass
class DiagnosticEnvelope:
    icr_error_bound: float
    leverage_error_bound: float
    assumptions: dict[str, float]


class AnalyticBoundsModel:
    """Diagnostic envelopes for the analytic approximation."""

    def __init__(self) -> None:
        self.last_envelope: DiagnosticEnvelope | None = None

    def calculate_diagnostic_envelopes(
        self,
        growth_bound: float = 0.12,
        capex_ratio_bound: float = 0.7,
        lease_decay_bound: float = 0.1,
    ) -> DiagnosticEnvelope:
        """
        Calculate implementation-level diagnostic envelopes under explicit assumptions.
        """
        icr_error_bound = 0.25 * (1 + growth_bound) ** 2 * (1 + capex_ratio_bound)
        leverage_error_bound = 0.30 * (1 + lease_decay_bound) * (1 + growth_bound) ** 0.5

        envelope = DiagnosticEnvelope(
            icr_error_bound=float(icr_error_bound),
            leverage_error_bound=float(leverage_error_bound),
            assumptions={
                "growth_bound": growth_bound,
                "capex_ratio_bound": capex_ratio_bound,
                "lease_decay_bound": lease_decay_bound,
            },
        )
        self.last_envelope = envelope
        return envelope

    def calculate_assumption_bounds(
        self,
        growth_bound: float = 0.12,
        capex_ratio_bound: float = 0.7,
        lease_decay_bound: float = 0.1,
    ) -> DiagnosticEnvelope:
        return self.calculate_diagnostic_envelopes(
            growth_bound=growth_bound,
            capex_ratio_bound=capex_ratio_bound,
            lease_decay_bound=lease_decay_bound,
        )

    def research_conjecture_dominance(self) -> dict[str, str]:
        """Clearly-labeled conjecture where a full proof is not supplied."""
        return {
            "label": "Research Conjecture",
            "statement": (
                "Under bounded approximation error and stable financing costs, "
                "larger analytic headroom tends to correspond to lower simulated covenant breach risk."
            ),
            "status": "Conjecture only; full derivation and proof are not included in this repository.",
        }

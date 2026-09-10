"""
Bayesian Inference Module for CD-TTA Framework
==============================================

Symptom → Root Cause inference using Bayesian reasoning.
Pure lookup tables + Bayesian formula - NO training whatsoever.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Result of Bayesian root cause inference"""
    root_causes: List[Tuple[str, float]]  # [(cause, probability), ...]
    most_likely: Tuple[str, float]  # (cause, probability)
    confidence: float  # 0-1, how confident in the inference
    explanation: str


class BayesianInference:
    """
    Pure inference using Bayesian formula - completely lookup table based.

    P(cause | symptoms) ∝ P(symptoms | cause) × P(cause)

    All probabilities are predefined (hand-crafted from domain knowledge),
    NOT learned from data.
    """

    def __init__(self):
        """Initialize with predefined symptom→cause mappings"""

        # Predefined symptom-to-causes mappings (domain knowledge)
        # Key: tuple of symptoms (sorted for consistency)
        # Value: dict of {cause: probability}
        self.symptom_to_causes = {
            # Search-related failures
            ("no_results", "timeout"): {  # SORTED
                "max_results_too_large": 0.75,
                "query_too_complex": 0.20,
                "network_error": 0.05
            },
            ("timeout",): {
                "timeout_too_short": 0.80,
                "slow_network": 0.15,
                "search_engine_down": 0.05
            },
            ("no_results",): {
                "query_too_specific": 0.60,
                "database_empty": 0.25,
                "tool_not_suitable": 0.15
            },

            # Format-related failures
            ("format_error",): {
                "output_format_mismatch": 0.90,
                "encoding_error": 0.10
            },
            ("encoding_error",): {
                "unsupported_charset": 0.75,
                "api_response_malformed": 0.25
            },

            # Parameter-related failures
            ("low_precision",): {
                "confidence_threshold_too_high": 0.85,
                "top_k_too_small": 0.15
            },

            # Tool compatibility failures
            ("incompatible_output",): {
                "tool_output_format_wrong": 0.80,
                "missing_required_fields": 0.20
            },

            # Combined failures
            ("low_precision", "no_results", "timeout"): {  # SORTED
                "parameter_mismatch": 0.70,
                "tool_not_suitable": 0.30
            },

            # API failures
            ("api_error", "rate_limit"): {  # SORTED
                "api_rate_limit_exceeded": 0.95,
                "api_key_invalid": 0.05
            },
            ("api_error",): {
                "api_key_invalid": 0.40,
                "authentication_failed": 0.30,
                "endpoint_down": 0.20,
                "invalid_parameters": 0.10
            },

            # Network failures
            ("connection_error",): {
                "network_timeout": 0.60,
                "dns_resolution_failed": 0.25,
                "firewall_blocked": 0.15
            },

            # Data-related failures
            ("empty_result", "no_data"): {  # SORTED
                "source_empty": 0.70,
                "filter_too_strict": 0.30
            },

            # Tool execution failures
            ("execution_error",): {
                "invalid_tool_input": 0.50,
                "tool_crash": 0.30,
                "missing_dependency": 0.20
            },
        }

        # Prior probabilities (P(cause) - general likelihood of each cause)
        self.cause_priors = {
            "parameter_mismatch": 0.30,
            "tool_not_suitable": 0.20,
            "query_too_complex": 0.15,
            "network_error": 0.10,
            "api_error": 0.10,
            "format_error": 0.10,
            "invalid_tool_input": 0.05,
        }

    def infer_root_cause(
        self,
        symptoms: List[str],
        prior_override: Dict[str, float] = None
    ) -> InferenceResult:
        """
        Infer root causes from symptoms using Bayesian reasoning.

        Formula:
            P(cause | symptoms) ∝ P(symptoms | cause) × P(cause)

        Args:
            symptoms: List of observed symptoms
            prior_override: Optional dict to override priors for specific causes

        Returns:
            InferenceResult with sorted list of causes and probabilities
        """
        if not symptoms:
            return InferenceResult(
                root_causes=[],
                most_likely=("unknown", 0.0),
                confidence=0.0,
                explanation="No symptoms provided"
            )

        # Step 1: Normalize symptom key
        symptom_key = tuple(sorted(symptoms))

        # Step 2: Lookup symptom-to-causes (if exact match exists)
        likelihood_dict = self.symptom_to_causes.get(symptom_key)

        if likelihood_dict is None:
            # Try partial matching (any subset of symptoms)
            likelihood_dict = self._partial_match_symptoms(symptom_key)

            if not likelihood_dict:
                return InferenceResult(
                    root_causes=[],
                    most_likely=("unknown", 0.0),
                    confidence=0.0,
                    explanation=f"No known pattern for symptoms: {symptoms}"
                )

        # Step 3: Apply Bayesian formula
        posterior = {}

        for cause, likelihood in likelihood_dict.items():
            # P(cause)
            prior = self.cause_priors.get(cause, 0.1)

            # Override if provided
            if prior_override and cause in prior_override:
                prior = prior_override[cause]

            # P(cause | symptoms) ∝ P(symptoms | cause) × P(cause)
            posterior[cause] = likelihood * prior

        # Step 4: Normalize to get probabilities
        total = sum(posterior.values())
        if total == 0:
            return InferenceResult(
                root_causes=[],
                most_likely=("unknown", 0.0),
                confidence=0.0,
                explanation="No valid posterior distribution"
            )

        normalized = {
            cause: prob / total
            for cause, prob in posterior.items()
        }

        # Step 5: Sort by probability
        sorted_causes = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True
        )

        most_likely = sorted_causes[0] if sorted_causes else ("unknown", 0.0)
        confidence = most_likely[1]

        # Build explanation
        top_3 = sorted_causes[:3]
        explanation = "Bayesian inference: " + ", ".join(
            [f"{c[0]} ({c[1]:.2%})" for c in top_3]
        )

        return InferenceResult(
            root_causes=sorted_causes,
            most_likely=most_likely,
            confidence=confidence,
            explanation=explanation
        )

    # ==================== Learning-like Updates (No Training!) ====================

    def record_observation(
        self,
        symptoms: List[str],
        actual_root_cause: str,
        strength: float = 0.1
    ):
        """
        Record an observation to strengthen symptom→cause mapping.

        This looks like learning, but it's NOT:
        - We're just updating a lookup table, not training any model
        - Updates are observational (empirical counting)
        - No optimization or gradient descent

        Args:
            symptoms: Observed symptoms
            actual_root_cause: True root cause
            strength: How much to strengthen the mapping [0, 1]
        """
        symptom_key = tuple(sorted(symptoms))

        # Create entry if doesn't exist
        if symptom_key not in self.symptom_to_causes:
            self.symptom_to_causes[symptom_key] = {}

        current = self.symptom_to_causes[symptom_key]

        # Simple Bayesian-like update (no training!)
        if actual_root_cause not in current:
            current[actual_root_cause] = 0.5
        else:
            # Strengthen correct cause
            current[actual_root_cause] = min(
                1.0,
                current[actual_root_cause] + strength
            )

            # Slightly weaken other causes
            for cause in current:
                if cause != actual_root_cause:
                    current[cause] = max(
                        0.0,
                        current[cause] - (strength * 0.2)
                    )

    def add_custom_pattern(
        self,
        symptoms: List[str],
        causes_and_probabilities: Dict[str, float]
    ):
        """
        Manually add a new symptom-to-causes pattern.

        This is for domain knowledge encoding, not learned data.

        Args:
            symptoms: List of symptoms
            causes_and_probabilities: Dict of {cause: probability}
        """
        symptom_key = tuple(sorted(symptoms))
        self.symptom_to_causes[symptom_key] = causes_and_probabilities

    # ==================== Private Helper Methods ====================

    def _partial_match_symptoms(self, symptom_key: Tuple[str, ...]) -> Optional[Dict[str, float]]:
        """
        Try to find a partial match if exact match fails.

        For example, if we see ["timeout", "no_results", "network_error"],
        try to match against ["timeout", "no_results"].

        Returns the closest matching symptom pattern's causes.
        """
        # Try matching from longest to shortest
        for size in range(len(symptom_key), 0, -1):
            for subset in self._generate_subsets(symptom_key, size):
                if subset in self.symptom_to_causes:
                    return self.symptom_to_causes[subset]

        return None

    def _generate_subsets(self, items: Tuple, size: int) -> List[Tuple]:
        """Generate all subsets of given size"""
        from itertools import combinations
        return [tuple(sorted(combo)) for combo in combinations(items, size)]

    # ==================== Export and Statistics ====================

    def get_statistics(self) -> Dict[str, any]:
        """Return statistics about the inference system"""
        all_causes = set()
        for causes in self.symptom_to_causes.values():
            all_causes.update(causes.keys())

        return {
            "total_symptom_patterns": len(self.symptom_to_causes),
            "total_known_causes": len(all_causes),
            "total_priors": len(self.cause_priors),
            "known_causes": sorted(all_causes),
        }

    def visualize_patterns(self) -> str:
        """Return text visualization of symptom→cause patterns"""
        lines = []
        lines.append("=== Bayesian Inference Patterns ===\n")

        for symptoms, causes in sorted(self.symptom_to_causes.items()):
            lines.append(f"\nSymptoms: {' + '.join(symptoms)}")
            for cause, prob in sorted(causes.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  → {cause}: {prob:.2%}")

        return "\n".join(lines)


# ==================== Standalone Utility ====================

def create_default_bayesian_inference() -> BayesianInference:
    """Factory function to create initialized BayesianInference"""
    return BayesianInference()

"""
Causal Inference Module for CD-TTA Framework
=============================================

Core reasoning engine - pure inference, no training.
- Counterfactual analysis: predict "if...then..." scenarios
- Causal mediation analysis: decompose failures
- Constraint checking: validate parameters
- Parameter optimization: search for best params (no ML)
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import math


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis"""
    feasible: bool
    current_success_rate: float = 0.0
    predicted_success_rate: float = 0.0
    improvement: float = 0.0  # P_new - P_old
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class MediationResult:
    """Result of causal mediation analysis"""
    direct_effect: float  # Tool's inherent ability
    indirect_effect: float  # Parameter tuning potential
    total_effect: float  # DE + IE
    diagnosis: str  # "TOOL_PROBLEM" or "PARAMETER_PROBLEM"
    recommended_fix: Dict[str, Any] = None  # Recommended action


@dataclass
class ConstraintViolation:
    """Represents a parameter constraint"""
    parameter: str
    constraint_type: str  # "lower_bound", "upper_bound", "range", "relationship"
    value: Any
    message: str


class CausalInference:
    """
    Pure causal inference engine - no training whatsoever.
    All analysis uses predefined relationships and history queries.
    """

    def __init__(self, causal_graph=None, history_analyzer=None):
        """
        Args:
            causal_graph: CausalGraph instance
            history_analyzer: HistoryAnalyzer instance for historical queries
        """
        self.graph = causal_graph
        self.history = history_analyzer

        # Predefined parameter bounds (domain knowledge)
        self.parameter_bounds = {
            "max_results": {"min": 1, "max": 200, "default": 50},
            "timeout": {"min": 1, "max": 60, "default": 10},
            "confidence_threshold": {"min": 0.0, "max": 1.0, "default": 0.5},
        }

        # Predefined parameter relationships (domain knowledge)
        self.parameter_relations = {
            # max_results ↔ timeout relationship
            ("max_results", "timeout"): "timeout >= max_results * 0.1",
        }

    # ==================== Counterfactual Analysis ====================

    def counterfactual_analysis(
        self,
        current_plan: List[str],
        hypothetical_subgoal: str,
        context: Dict[str, Any] = None
    ) -> CounterfactualResult:
        """
        Predict: "If I add this subgoal to the plan, will success rate improve?"

        No training - just:
        1. Check if subgoal is feasible (has tools)
        2. Query historical success rates
        3. Calculate predicted improvement

        Args:
            current_plan: List of current subgoals
            hypothetical_subgoal: New subgoal to add
            context: Additional context (query_complexity, etc.)

        Returns:
            CounterfactualResult with feasibility and predicted improvement
        """
        result = CounterfactualResult(feasible=False)

        if not self.graph:
            result.reasoning = "No causal graph available"
            return result

        # Step 1: Check feasibility (has tools available?)
        available_tools = self.graph.find_tools_for_subgoal(hypothetical_subgoal)
        if not available_tools:
            result.feasible = False
            result.reasoning = f"No tools available for '{hypothetical_subgoal}'"
            return result

        result.feasible = True

        # Step 2: Query historical success rates
        # Use best tool's historical success rate
        best_tool = available_tools[0]
        historical_sr = self.graph.get_success_rate(best_tool, hypothetical_subgoal)

        # Step 3: Calculate plan success rates (assuming independence)
        # Current plan success = product of each step's success rate
        current_success = 0.8  # Prior
        if self.history:
            current_success = self._estimate_plan_success(current_plan)

        # New plan success = current × new_subgoal_success
        new_success = current_success * historical_sr
        improvement = new_success - current_success

        result.current_success_rate = current_success
        result.predicted_success_rate = new_success
        result.improvement = improvement
        result.confidence = 0.7  # Based on historical data
        result.reasoning = (
            f"Historical success rate for '{hypothetical_subgoal}' "
            f"using {best_tool}: {historical_sr:.2f}. "
            f"Predicted improvement: {improvement:.2f}"
        )

        return result

    # ==================== Causal Mediation Analysis ====================

    def mediation_analysis(
        self,
        subgoal: str,
        tool: str,
        params: Dict[str, Any],
        failure: Dict[str, Any]
    ) -> MediationResult:
        """
        Decompose failure: Is it a tool problem or a parameter problem?

        Uses:
        - Direct Effect (DE): Tool's inherent ability with default params
        - Indirect Effect (IE): Tool's improvement potential through param tuning

        If IE > DE: Problem is parameter tuning
        If DE > IE: Problem is wrong tool choice

        Args:
            subgoal: The subgoal that failed
            tool: The tool that was used
            params: Parameters used in failed execution
            failure: Failure information (error, symptoms, etc.)

        Returns:
            MediationResult with diagnosis and recommendation
        """
        result = MediationResult(
            direct_effect=0.0,
            indirect_effect=0.0,
            total_effect=0.0,
            diagnosis="UNKNOWN"
        )

        if not self.history:
            result.reasoning = "No history available for mediation analysis"
            return result

        # Step 1: Direct Effect - tool's base capability
        # Try default parameters
        default_params = self._get_default_params(tool)
        de = self.history.estimate_success_rate(tool, default_params)
        result.direct_effect = de

        # Step 2: Indirect Effect - parameter optimization potential
        # Get the best parameters from history
        best_params = self.history.get_best_parameters(tool)
        ie_best = self.history.estimate_success_rate(tool, best_params)

        # Indirect effect = improvement from tuning
        ie = ie_best - self.history.estimate_success_rate(tool, params)
        result.indirect_effect = ie

        result.total_effect = de + ie

        # Step 3: Diagnosis
        if ie > de:
            # Parameter problem dominates
            result.diagnosis = "PARAMETER_PROBLEM"
            result.recommended_fix = {
                "action": "tune_parameters",
                "suggested_params": best_params,
                "estimated_improvement": ie
            }
        else:
            # Tool problem dominates
            result.diagnosis = "TOOL_PROBLEM"
            result.recommended_fix = {
                "action": "switch_tool",
                "reason": "Tool's inherent ability is limited",
                "try_alternative": True
            }

        return result

    # ==================== Constraint Checking ====================

    def check_constraints(
        self,
        tool: str,
        parameters: Dict[str, Any],
        constraints: List[Dict[str, Any]] = None
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check if parameters satisfy all constraints.

        Supports:
        - Parameter bounds (e.g., max_results ≤ 50)
        - Parameter relationships (e.g., timeout ≥ max_results × 0.1)
        - Custom constraints

        Args:
            tool: Tool name
            parameters: Parameter dict
            constraints: List of constraint dicts (from learning)

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Check built-in bounds
        for param_name, param_value in parameters.items():
            if param_name in self.parameter_bounds:
                bounds = self.parameter_bounds[param_name]

                if param_value < bounds["min"]:
                    violations.append(ConstraintViolation(
                        parameter=param_name,
                        constraint_type="lower_bound",
                        value=param_value,
                        message=f"{param_name} must be ≥ {bounds['min']}"
                    ))

                if param_value > bounds["max"]:
                    violations.append(ConstraintViolation(
                        parameter=param_name,
                        constraint_type="upper_bound",
                        value=param_value,
                        message=f"{param_name} must be ≤ {bounds['max']}"
                    ))

        # Check parameter relationships
        if "max_results" in parameters and "timeout" in parameters:
            max_results = parameters["max_results"]
            timeout = parameters["timeout"]
            required_timeout = max_results * 0.1

            if timeout < required_timeout:
                violations.append(ConstraintViolation(
                    parameter="timeout",
                    constraint_type="relationship",
                    value=timeout,
                    message=f"timeout ({timeout}) must be ≥ max_results × 0.1 ({required_timeout})"
                ))

        # Check learned constraints
        if constraints:
            for constraint in constraints:
                if constraint["type"] == "parameter_bound":
                    param = constraint["parameter"]
                    if param in parameters:
                        # Parse bound (e.g., "≤ 50" or "[1, 50]")
                        bound_value = constraint["bound"]
                        if isinstance(bound_value, int):
                            if parameters[param] > bound_value:
                                violations.append(ConstraintViolation(
                                    parameter=param,
                                    constraint_type="learned_bound",
                                    value=parameters[param],
                                    message=f"{param} must be ≤ {bound_value}"
                                ))

        is_valid = len(violations) == 0
        return is_valid, violations

    # ==================== Parameter Optimization ====================

    def optimize_parameters(
        self,
        tool: str,
        subgoal: str,
        constraints: List[Dict[str, Any]] = None,
        search_strategy: str = "grid"
    ) -> Dict[str, Any]:
        """
        Search for best parameters within constraints.

        No ML - just grid search or greedy search.

        Args:
            tool: Tool name
            subgoal: Subgoal to achieve
            constraints: Parameter constraints
            search_strategy: "grid" or "greedy"

        Returns:
            Best parameter dict
        """
        if not self.history:
            return self._get_default_params(tool)

        # Step 1: Get candidate parameter combinations
        candidates = self._generate_parameter_candidates(tool)

        # Step 2: Filter by constraints
        valid_candidates = []
        for params in candidates:
            is_valid, _ = self.check_constraints(tool, params, constraints)
            if is_valid:
                valid_candidates.append(params)

        if not valid_candidates:
            return self._get_default_params(tool)

        # Step 3: Score each candidate (no training, just history queries)
        best_params = valid_candidates[0]
        best_score = 0.0

        for params in valid_candidates:
            # Query historical success rate for these params
            score = self.history.estimate_success_rate(tool, params)

            if score > best_score:
                best_score = score
                best_params = params

        return best_params

    # ==================== Private Helper Methods ====================

    def _estimate_plan_success(self, plan: List[str]) -> float:
        """Estimate success rate of a plan (product of step success rates)"""
        if not plan:
            return 1.0

        if not self.graph:
            return 0.5

        success_rate = 1.0
        for subgoal in plan:
            # Find best tool for this subgoal
            tools = self.graph.find_tools_for_subgoal(subgoal)
            if tools:
                tool = tools[0]
                step_sr = self.graph.get_success_rate(tool, subgoal)
                success_rate *= step_sr
            else:
                success_rate *= 0.5  # No tool found

        return success_rate

    def _get_default_params(self, tool: str) -> Dict[str, Any]:
        """Get default parameters for a tool"""
        return {
            "max_results": self.parameter_bounds["max_results"]["default"],
            "timeout": self.parameter_bounds["timeout"]["default"],
        }

    def _generate_parameter_candidates(self, tool: str) -> List[Dict[str, Any]]:
        """Generate candidate parameter combinations"""
        candidates = []

        # Grid of common values
        max_results_vals = [10, 20, 30, 50, 75, 100]
        timeout_vals = [5, 10, 15, 30]

        for mr in max_results_vals:
            for t in timeout_vals:
                candidates.append({
                    "max_results": mr,
                    "timeout": t
                })

        return candidates


# ==================== Integration Points ====================

class CausalInferenceWithHistory:
    """
    Wrapper combining CausalInference with HistoryAnalyzer.
    Used for complete reasoning.
    """

    def __init__(self, causal_graph, history_analyzer, bayesian_inference=None):
        self.inference = CausalInference(causal_graph, history_analyzer)
        self.graph = causal_graph
        self.history = history_analyzer
        self.bayesian = bayesian_inference

    def diagnose_and_recommend(
        self,
        failure: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Full diagnosis: symptoms → root cause → recommended fix

        Pipeline:
        1. Parse symptoms from failure
        2. Use Bayesian inference to find likely root causes
        3. Use mediation analysis to recommend fix
        4. Check constraints
        5. Return recommendation
        """
        symptoms = failure.get("symptoms", [])
        tool = failure.get("tool", "")
        params = failure.get("parameters", {})

        diagnosis = {
            "symptoms": symptoms,
            "root_causes": [],
            "recommended_action": None,
            "confidence": 0.0
        }

        # Step 1: Bayesian inference if available
        if self.bayesian and symptoms:
            root_causes = self.bayesian.infer_root_cause(symptoms)
            diagnosis["root_causes"] = root_causes
            diagnosis["confidence"] = root_causes[0][1] if root_causes else 0.0

            # Step 2: Recommend fix based on root cause
            primary_cause = root_causes[0][0] if root_causes else "unknown"

            if "parameter" in primary_cause.lower():
                # Parameter problem - optimize
                best_params = self.inference.optimize_parameters(tool, context.get("subgoal", ""))
                diagnosis["recommended_action"] = {
                    "type": "parameter_tuning",
                    "suggested_params": best_params
                }
            elif "tool" in primary_cause.lower():
                # Tool problem - switch
                diagnosis["recommended_action"] = {
                    "type": "tool_switch",
                    "reason": primary_cause
                }

        return diagnosis

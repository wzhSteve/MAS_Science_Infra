"""
History Analyzer Module for CD-TTA Framework
============================================

Query execution history for parameter optimization and constraint extraction.
Pure queries and simple arithmetic - NO training whatsoever.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class ExecutionRecord:
    """Record of a single tool execution"""
    tool: str
    subgoal: str
    parameters: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    timestamp: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)  # e.g., latency, result_count


@dataclass
class ParameterConstraint:
    """Extracted constraint on a parameter"""
    parameter: str
    constraint_type: str  # "lower_bound", "upper_bound", "range", "relationship"
    value: Any
    confidence: float  # 0-1, based on observation count
    supporting_evidence_count: int


class HistoryAnalyzer:
    """
    Analyze execution history to extract:
    1. Optimal parameters (simple averaging)
    2. Parameter constraints (simple comparison)
    3. Success rates (frequency analysis)

    Pure queries - NO training algorithms whatsoever.
    """

    def __init__(self):
        """Initialize empty history"""
        # {tool: [ExecutionRecord, ...]}
        self.history: Dict[str, List[ExecutionRecord]] = defaultdict(list)

        # Cache for efficiency
        self._best_params_cache: Dict[str, Dict[str, Any]] = {}
        self._success_rate_cache: Dict[str, float] = {}

    # ==================== Recording ====================

    def record_execution(
        self,
        tool: str,
        subgoal: str,
        parameters: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        metrics: Dict[str, Any] = None
    ) -> ExecutionRecord:
        """
        Record a tool execution attempt.

        This is NOT training - just appending to history.

        Args:
            tool: Tool name
            subgoal: Subgoal being pursued
            parameters: Parameter dict used
            success: Whether execution succeeded
            error: Error message if failed
            metrics: Additional metrics (latency, result_count, etc.)

        Returns:
            ExecutionRecord that was recorded
        """
        record = ExecutionRecord(
            tool=tool,
            subgoal=subgoal,
            parameters=parameters.copy(),
            success=success,
            error=error,
            metrics=metrics or {}
        )

        self.history[tool].append(record)

        # Invalidate caches
        self._best_params_cache.pop(tool, None)
        self._success_rate_cache.pop(tool, None)

        return record

    def get_history_for_tool(self, tool: str) -> List[ExecutionRecord]:
        """Get all execution records for a tool"""
        return self.history.get(tool, [])

    def get_history_size(self) -> Dict[str, int]:
        """Get number of records per tool"""
        return {tool: len(records) for tool, records in self.history.items()}

    # ==================== Simple Queries (No Training!) ====================

    def get_best_parameters(self, tool: str) -> Dict[str, Any]:
        """
        Query: What parameters work best for this tool?

        Method: Simple averaging of successful executions.
        No machine learning, no optimization.

        Returns:
            Dict of parameter → value (averaged over successful executions)
        """
        # Check cache
        if tool in self._best_params_cache:
            return self._best_params_cache[tool].copy()

        records = self.history.get(tool, [])
        if not records:
            return {}

        # Filter successful executions
        successful = [r for r in records if r.success]
        if not successful:
            return {}

        # Simple averaging (NO optimization!)
        best_params = {}
        all_param_names = set()

        for record in successful:
            all_param_names.update(record.parameters.keys())

        for param_name in all_param_names:
            values = [
                r.parameters[param_name]
                for r in successful
                if param_name in r.parameters
            ]

            if values:
                # Average numeric values
                if all(isinstance(v, (int, float)) for v in values):
                    best_params[param_name] = statistics.mean(values)
                else:
                    # For non-numeric, use most common
                    best_params[param_name] = max(
                        set(values),
                        key=values.count
                    )

        # Cache result
        self._best_params_cache[tool] = best_params.copy()

        return best_params

    def get_success_rate(self, tool: str) -> float:
        """
        Query: What's the success rate for this tool?

        Simple frequency: successes / total
        """
        # Check cache
        cache_key = f"{tool}_overall"
        if cache_key in self._success_rate_cache:
            return self._success_rate_cache[cache_key]

        records = self.history.get(tool, [])
        if not records:
            return 0.5  # Prior

        successes = sum(1 for r in records if r.success)
        rate = successes / len(records)

        self._success_rate_cache[cache_key] = rate
        return rate

    def get_success_rate_for_params(self, tool: str, params: Dict[str, Any]) -> float:
        """
        Query: Success rate with these specific parameters?

        Simple frequency matching.
        """
        records = self.history.get(tool, [])
        if not records:
            return 0.5

        # Find records with matching parameters
        matching = [
            r for r in records
            if self._params_match(r.parameters, params)
        ]

        if not matching:
            return 0.5  # Not enough data

        successes = sum(1 for r in matching if r.success)
        return successes / len(matching)

    def estimate_success_rate(self, tool: str, params: Dict[str, Any]) -> float:
        """
        Estimate success rate for parameters.

        If exact match exists in history, use it.
        Otherwise, interpolate from similar successful executions.
        """
        # Try exact match first
        exact_rate = self.get_success_rate_for_params(tool, params)
        if exact_rate > 0.5:
            return exact_rate

        # Fall back to general tool success rate
        return self.get_success_rate(tool)

    # ==================== Constraint Extraction ====================

    def extract_parameter_bounds(self, tool: str) -> Dict[str, ParameterConstraint]:
        """
        Infer parameter bounds from history.

        Heuristic:
        - If all successes have X ≤ 50 but some failures have X > 50
          → infer constraint X ≤ 50

        This is empirical analysis, NOT training.

        Returns:
            Dict of parameter → ParameterConstraint
        """
        records = self.history.get(tool, [])
        if not records:
            return {}

        constraints = {}

        # Identify all parameters
        all_param_names = set()
        for record in records:
            all_param_names.update(record.parameters.keys())

        # Analyze each parameter
        for param_name in all_param_names:
            successful_values = []
            failed_values = []

            for record in records:
                if param_name in record.parameters:
                    value = record.parameters[param_name]

                    # Only numeric values can have bounds
                    if isinstance(value, (int, float)):
                        if record.success:
                            successful_values.append(value)
                        else:
                            failed_values.append(value)

            if not successful_values or not failed_values:
                continue

            # Check for upper bound
            max_successful = max(successful_values)
            min_failed = min(failed_values)

            if min_failed > max_successful:
                # Found upper bound
                confidence = len(successful_values) / (len(successful_values) + len(failed_values))
                constraints[f"{param_name}_max"] = ParameterConstraint(
                    parameter=param_name,
                    constraint_type="upper_bound",
                    value=max_successful,
                    confidence=confidence,
                    supporting_evidence_count=len(successful_values)
                )

            # Check for lower bound
            min_successful = min(successful_values)
            max_failed = max(failed_values)

            if max_failed < min_successful:
                # Found lower bound
                confidence = len(successful_values) / (len(successful_values) + len(failed_values))
                constraints[f"{param_name}_min"] = ParameterConstraint(
                    parameter=param_name,
                    constraint_type="lower_bound",
                    value=min_successful,
                    confidence=confidence,
                    supporting_evidence_count=len(successful_values)
                )

        return constraints

    def extract_parameter_relationships(self, tool: str) -> List[Dict[str, Any]]:
        """
        Infer relationships between parameters.

        Example: if successful executions all have timeout ≥ max_results * 0.1
                 → infer relationship

        This is heuristic analysis, NOT training.

        Returns:
            List of relationship dicts
        """
        records = self.history.get(tool, [])
        if not records:
            return []

        successful = [r for r in records if r.success]
        if not successful:
            return []

        relationships = []

        # Check max_results ↔ timeout relationship
        if any("max_results" in r.parameters for r in successful):
            max_results_vals = [
                r.parameters["max_results"]
                for r in successful
                if "max_results" in r.parameters
            ]
            timeout_vals = [
                r.parameters["timeout"]
                for r in successful
                if "timeout" in r.parameters and "max_results" in r.parameters
            ]

            if len(max_results_vals) == len(timeout_vals) and max_results_vals:
                # Simple regression heuristic
                ratio = max(
                    t / m for t, m in zip(timeout_vals, max_results_vals)
                    if m > 0
                )

                relationships.append({
                    "param1": "max_results",
                    "param2": "timeout",
                    "relation": f"timeout >= max_results * {ratio:.2f}",
                    "confidence": 0.7
                })

        return relationships

    # ==================== Private Helper Methods ====================

    def _params_match(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Check if two parameter dicts match (all values equal)"""
        if set(params1.keys()) != set(params2.keys()):
            return False

        for key in params1:
            if params1[key] != params2[key]:
                return False

        return True

    # ==================== Statistics and Export ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about recorded history"""
        stats = {
            "total_tools": len(self.history),
            "total_records": sum(len(r) for r in self.history.values()),
            "tools": {}
        }

        for tool, records in self.history.items():
            successes = sum(1 for r in records if r.success)
            stats["tools"][tool] = {
                "total_executions": len(records),
                "successes": successes,
                "failures": len(records) - successes,
                "success_rate": successes / len(records) if records else 0.0
            }

        return stats

    def export_to_dict(self) -> Dict[str, Any]:
        """Export history as dictionary (for serialization)"""
        return {
            "history": {
                tool: [
                    {
                        "subgoal": r.subgoal,
                        "parameters": r.parameters,
                        "success": r.success,
                        "error": r.error,
                        "metrics": r.metrics
                    }
                    for r in records
                ]
                for tool, records in self.history.items()
            }
        }

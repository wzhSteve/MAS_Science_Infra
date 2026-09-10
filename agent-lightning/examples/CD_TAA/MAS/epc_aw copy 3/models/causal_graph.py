"""
Causal Graph Module for CD-TTA Framework
=========================================

Stores and queries causal relationships in the multi-agent system.
- No training, pure data structures and inference
- Supports task→subgoal→tool→parameter relationships
- Tracks failure patterns (symptoms→root causes)
- Records observations to update causal strengths
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
import json


class NodeType(Enum):
    """Types of nodes in the causal graph"""
    TASK = "task"
    STATE = "state"
    SUBGOAL = "subgoal"
    TOOL = "tool"
    PARAMETER = "parameter"
    OUTCOME = "outcome"
    EVIDENCE = "evidence"
    SYMPTOM = "symptom"
    ROOT_CAUSE = "root_cause"


class EdgeType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"                    # X → Y (direct causation)
    CONDITIONAL = "conditional"          # X → Y if Z (conditional causation)
    BIDIRECTIONAL = "bidirectional"      # X ↔ Y (mutual influence)
    FAILURE_PATTERN = "failure_pattern"  # Symptom → Root Cause


@dataclass
class CausalNode:
    """Represents a node in the causal graph"""
    node_id: str
    node_type: NodeType
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.node_id == other.node_id
        return self.node_id == other


@dataclass
class CausalEdge:
    """Represents a causal relationship between two nodes"""
    source: str          # source node_id
    target: str          # target node_id
    edge_type: EdgeType
    strength: float = 0.5  # [0, 1] - confidence in this causal relationship
    condition: str = ""    # optional condition (e.g., "when query_complexity is high")
    confidence: float = 0.5  # [0, 1] - confidence based on observations
    observation_count: int = 0  # Number of observations supporting this edge

    def __hash__(self):
        return hash((self.source, self.target))

    def __eq__(self, other):
        if isinstance(other, CausalEdge):
            return self.source == other.source and self.target == other.target
        return False


class CausalGraph:
    """
    Directed acyclic graph representing causal relationships.

    Pure inference system - no training whatsoever.
    All updates are observational (recording successes/failures).
    """

    def __init__(self):
        """Initialize empty causal graph"""
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}

        # Failure patterns: {(symptom_tuple): {root_cause: probability}}
        self.failure_patterns: Dict[Tuple[str, ...], Dict[str, float]] = {}

        # Historical success rates: {("tool", "subgoal"): success_rate}
        self.success_rates: Dict[Tuple[str, str], float] = {}

    # ==================== Node Management ====================

    def add_node(self, node_id: str, node_type: NodeType, description: str = "",
                 metadata: Dict[str, Any] = None) -> CausalNode:
        """Add a node to the graph"""
        if node_id not in self.nodes:
            self.nodes[node_id] = CausalNode(
                node_id=node_id,
                node_type=node_type,
                description=description,
                metadata=metadata or {}
            )
        return self.nodes[node_id]

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[CausalNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    # ==================== Edge Management ====================

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType,
                 condition: str = "", strength: float = 0.5,
                 confidence: float = 0.5) -> CausalEdge:
        """Add a causal edge (or update if exists)"""
        key = (source_id, target_id)

        if key not in self.edges:
            self.edges[key] = CausalEdge(
                source=source_id,
                target=target_id,
                edge_type=edge_type,
                condition=condition,
                strength=strength,
                confidence=confidence
            )
        else:
            # Update existing edge
            edge = self.edges[key]
            edge.condition = condition or edge.condition
            edge.strength = strength
            edge.confidence = confidence

        return self.edges[key]

    def get_edge(self, source_id: str, target_id: str) -> Optional[CausalEdge]:
        """Get a causal edge"""
        return self.edges.get((source_id, target_id))

    def get_outgoing_edges(self, node_id: str) -> List[CausalEdge]:
        """Get all edges originating from a node"""
        return [e for e in self.edges.values() if e.source == node_id]

    def get_incoming_edges(self, node_id: str) -> List[CausalEdge]:
        """Get all edges pointing to a node"""
        return [e for e in self.edges.values() if e.target == node_id]

    # ==================== Query: Find Tools for Subgoal ====================

    def find_tools_for_subgoal(self, subgoal_id: str, strength_threshold: float = 0.3) -> List[str]:
        """
        Query: Which tools can achieve this subgoal?

        Returns list of tool IDs sorted by causal strength.
        """
        tools = []

        for edge in self.get_incoming_edges(subgoal_id):
            if edge.edge_type in [EdgeType.DIRECT, EdgeType.CONDITIONAL]:
                if edge.strength >= strength_threshold:
                    source_node = self.get_node(edge.source)
                    if source_node and source_node.node_type == NodeType.TOOL:
                        tools.append((edge.source, edge.strength))

        # Sort by strength (descending)
        tools.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in tools]

    # ==================== Query: Find Subgoals for Task ====================

    def find_subgoals_for_task(self, task_id: str, strength_threshold: float = 0.3) -> List[str]:
        """
        Query: What subgoals are needed to complete this task?

        Returns list of subgoal IDs sorted by causal strength.
        """
        subgoals = []

        for edge in self.get_outgoing_edges(task_id):
            if edge.edge_type in [EdgeType.DIRECT, EdgeType.CONDITIONAL]:
                if edge.strength >= strength_threshold:
                    target_node = self.get_node(edge.target)
                    if target_node and target_node.node_type == NodeType.SUBGOAL:
                        subgoals.append((edge.target, edge.strength))

        subgoals.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in subgoals]

    # ==================== Failure Pattern Management ====================

    def add_failure_pattern(self, symptoms: List[str], root_causes: Dict[str, float]):
        """
        Add a failure pattern: symptoms → root causes with probabilities.

        Args:
            symptoms: List of symptom strings (e.g., ["timeout", "no_results"])
            root_causes: Dict of {cause: probability}
        """
        symptom_key = tuple(sorted(symptoms))
        self.failure_patterns[symptom_key] = root_causes

    def get_failure_pattern(self, symptoms: List[str]) -> Optional[Dict[str, float]]:
        """Get root causes for a set of symptoms (without specific order)"""
        symptom_key = tuple(sorted(symptoms))
        return self.failure_patterns.get(symptom_key)

    # ==================== Observation Recording (No Training!) ====================

    def add_observation(self, source_id: str, target_id: str, success: bool,
                       weight: float = 0.1):
        """
        Record an observation to update causal strength.

        This is NOT training - just simple observational updates:
        - Success: increase strength
        - Failure: decrease strength

        Args:
            source_id: Source node ID (e.g., tool)
            target_id: Target node ID (e.g., subgoal)
            success: Whether the causal relationship led to success
            weight: Step size for strength update [0, 1]
        """
        edge = self.get_edge(source_id, target_id)

        if edge is None:
            # Create edge if doesn't exist
            edge = self.add_edge(
                source_id, target_id,
                EdgeType.DIRECT,
                strength=0.5
            )

        # Observational update (simple Bayesian-like)
        if success:
            # Increase strength when successful
            edge.strength = min(1.0, edge.strength + weight)
        else:
            # Decrease strength when failed
            edge.strength = max(0.0, edge.strength - weight)

        edge.observation_count += 1

        # Update confidence (more observations = higher confidence)
        edge.confidence = min(1.0, 0.5 + (edge.observation_count * 0.1))

    # ==================== Success Rate Tracking ====================

    def record_tool_subgoal_success(self, tool_id: str, subgoal_id: str, success: bool):
        """
        Record a tool execution attempt for a subgoal.
        Used to build historical success rates.
        """
        key = (tool_id, subgoal_id)

        if key not in self.success_rates:
            self.success_rates[key] = 0.5  # Prior

        # Simple exponential moving average (no training!)
        alpha = 0.7  # Forget older observations
        self.success_rates[key] = (
            alpha * self.success_rates[key] +
            (1 - alpha) * (1.0 if success else 0.0)
        )

    def get_success_rate(self, tool_id: str, subgoal_id: str) -> float:
        """Get historical success rate for a tool-subgoal pair"""
        return self.success_rates.get((tool_id, subgoal_id), 0.5)

    # ==================== Export and Visualization ====================

    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary (for serialization)"""
        return {
            "nodes": {
                nid: {
                    "type": node.node_type.value,
                    "description": node.description,
                    "metadata": node.metadata
                }
                for nid, node in self.nodes.items()
            },
            "edges": {
                f"{e.source}->{e.target}": {
                    "type": e.edge_type.value,
                    "strength": e.strength,
                    "confidence": e.confidence,
                    "observation_count": e.observation_count,
                    "condition": e.condition
                }
                for e in self.edges.values()
            },
            "failure_patterns": {
                str(k): v for k, v in self.failure_patterns.items()
            },
            "success_rates": {
                str(k): v for k, v in self.success_rates.items()
            }
        }

    def visualize_structure(self) -> str:
        """Return a text representation of the graph structure"""
        lines = []
        lines.append("=== Causal Graph Structure ===\n")

        # Group nodes by type
        by_type = {}
        for node in self.nodes.values():
            if node.node_type not in by_type:
                by_type[node.node_type] = []
            by_type[node.node_type].append(node)

        for node_type, nodes in by_type.items():
            lines.append(f"\n{node_type.value.upper()} Nodes:")
            for node in nodes:
                lines.append(f"  - {node.node_id}: {node.description}")

        # Show edges
        lines.append("\n\nCausal Edges:")
        for edge in self.edges.values():
            condition_str = f" (if {edge.condition})" if edge.condition else ""
            lines.append(
                f"  {edge.source} --[{edge.edge_type.value}, "
                f"strength={edge.strength:.2f}]--> {edge.target}{condition_str}"
            )

        # Show failure patterns
        if self.failure_patterns:
            lines.append("\n\nFailure Patterns:")
            for symptoms, causes in self.failure_patterns.items():
                lines.append(f"  {symptoms} →")
                for cause, prob in sorted(causes.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"    - {cause} ({prob:.2f})")

        return "\n".join(lines)

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the graph"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "failure_patterns": len(self.failure_patterns),
            "tracked_success_rates": len(self.success_rates),
            "nodes_by_type": {
                t.value: len(self.get_nodes_by_type(t))
                for t in NodeType
            }
        }

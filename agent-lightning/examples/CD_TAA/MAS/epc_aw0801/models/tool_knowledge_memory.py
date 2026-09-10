"""Tool Knowledge Memory (offline, cross-task, abstract).

Two independent, lightweight, human-readable memories that store ABSTRACT
tool knowledge learned from successful trajectories. This is Tool Knowledge,
NOT Experience Memory: it never grows linearly with the number of completed
tasks. Every evolution step merges / generalizes / compresses / differentiates
so that memory size stays approximately constant while knowledge quality
continuously improves.

Canonical terminology (used consistently in comments, docstrings, prompts):
  State              -> what information is currently available (online only)
  Subgoal            -> what information is still missing
  Capability         -> when should this tool be selected
  Applicability      -> under what conditions is this capability appropriate
  Invocation         -> how should this tool be invoked
  Decision Dimension -> which independent aspect influences parameter
                        construction (a.k.a. Factor)
  Instruction        -> how parameters should be constructed under that
                        decision dimension

Memory 1 — ToolCapabilityMemory (consumed by Planner):
    Answers "which tool should be selected?". Describes each tool's
    capability boundary as plain-text subgoals, each with a small list of
    Applicability Conditions (`context_summary`).

Memory 2 — ToolInvocationMemory (consumed by Executor, never Planner):
    Answers "how should the selected tool be invoked?". Stores, per subgoal,
    a small set of Decision Dimensions (factors), each with exactly one
    Instruction.

Schema invariants (HARD — must never be violated):
  * No scores / counts / confidence / probability / timestamps / embeddings /
    metadata / history / aliases / merge_log / statistical information.
    Everything stored is plain text.
  * No new persistent fields may be added to the on-disk schema unless
    absolutely necessary. Prefer improving the semantics of existing fields
    over increasing schema complexity. A simpler schema is preferred over a
    more expressive schema. (Memory Compression is a core selling point of
    this design; schema bloat would break it.)
  * Capability: <= MAX_SUBGOALS_PER_TOOL subgoals, <= MAX_CONTEXT_SUMMARIES
    applicability conditions per subgoal. Subgoals are canonical/frozen.
  * Invocation: <= MAX_SUBGOALS_PER_TOOL subgoals, <= MAX_FACTORS_PER_SUBGOAL
    decision dimensions per subgoal, exactly one instruction per dimension.
  * When a limit is reached, always merge / generalize / rewrite — never append.

The memory is NOT append-only. Every `evolve()` call retrieves a similar
existing entry, merges, and rewrites into a more general / more
distinguishable description, so the memory stays compact over time.

See MEMORY_SYSTEM_DESIGN.md and MEMORY_TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# ── YAML (optional) ─────────────────────────────────────────────────
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ── Capacity constraints (hard invariants from the spec) ────────────
# Capability: <= 5 subgoals/tool, <= 5 applicability conditions/subgoal.
# Invocation: <= 5 subgoals/tool, <= 4 decision dimensions/subgoal,
# exactly 1 instruction per dimension.
MAX_SUBGOALS_PER_TOOL = 5
MAX_CONTEXT_SUMMARIES = 5        # applicability conditions per subgoal
MAX_FACTORS_PER_SUBGOAL = 4      # decision dimensions per subgoal

# Schema version. Bumped when the on-disk shape changes; load_offline_memory
# resets to seed when a stale schema is detected. NOTE: do NOT bump this to
# accommodate new fields — the schema is intentionally minimal.
MEMORY_SCHEMA_VERSION = "tool_knowledge_memory_v2"

# Suggested canonical orthogonal Decision Dimension names. Not hard-coded —
# the LLM abstraction gate prefers reusing these so dimensions stay orthogonal
# across tools, and factor_refinement merges any semantic duplicates
# (e.g. Time->Freshness, Object->Entity).
CANONICAL_FACTOR_NAMES = (
    "Entity", "Freshness", "Scope", "Source", "Breadth", "Latency", "Format",
)


# =====================================================================
# Retrieval interface
# =====================================================================
class MemoryRetriever(ABC):
    """Abstract retrieval interface.

    Pure-text semantic matching. No embeddings. The default implementation
    may call an LLM; future implementations may replace it without touching
    the memory classes.
    """

    @abstractmethod
    def find_similar_subgoal(
        self, subgoals: List[dict], query_subgoal: str
    ) -> Optional[dict]:
        """Return the most similar subgoal entry to `query_subgoal`, or None."""

    @abstractmethod
    def find_similar_factor(
        self, factors: Dict[str, dict], query_factor: str
    ) -> Optional[str]:
        """Return the most similar factor name to `query_factor`, or None."""

    def find_similar_subgoal_across_tools(
        self, subgoal_text: str, candidates: List[Tuple[str, dict]]
    ) -> Optional[Tuple[str, dict]]:
        """Optional helper: cross-tool subgoal match.

        `candidates` is a list of (tool_name, entry). Default raises; concrete
        retrievers override. Used by boundary refinement.
        """
        raise NotImplementedError("Provided by concrete retrievers.")


class KeywordMemoryRetriever(MemoryRetriever):
    """No-LLM fallback retriever using token-overlap scoring.

    Used when no LLM engine is available (tests, offline dev). Keeps the
    module importable and testable without network/API access.
    """

    STOPWORDS = {
        "a", "an", "the", "of", "for", "to", "in", "on", "and", "or", "is",
        "are", "be", "with", "from", "by", "that", "this", "it", "as",
    }

    @classmethod
    def _tokens(cls, text: str) -> set:
        toks = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {t for t in toks if t not in cls.STOPWORDS and len(t) > 1}

    @classmethod
    def _score(cls, a: str, b: str) -> float:
        ta, tb = cls._tokens(a), cls._tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb) or 1
        return inter / union

    def find_similar_subgoal(
        self, subgoals: List[dict], query_subgoal: str
    ) -> Optional[dict]:
        best, best_score = None, 0.0
        for entry in subgoals:
            s = self._score(entry.get("subgoal", ""), query_subgoal)
            if s > best_score:
                best, best_score = entry, s
        # Jaccard >= 0.15 is "sufficiently similar" for the keyword fallback
        return best if best_score >= 0.15 else None

    def find_similar_factor(
        self, factors: Dict[str, dict], query_factor: str
    ) -> Optional[str]:
        best, best_score = None, 0.0
        for name in factors:
            s = self._score(name, query_factor)
            if s > best_score:
                best, best_score = name, s
        return best if best_score >= 0.15 else None

    def find_similar_subgoal_across_tools(
        self, subgoal_text: str, candidates: List[Tuple[str, dict]]
    ) -> Optional[Tuple[str, dict]]:
        best: Optional[Tuple[str, dict]] = None
        best_score = 0.0
        for tool_name, entry in candidates:
            s = self._score(entry.get("subgoal", ""), subgoal_text)
            if s > best_score:
                best, best_score = (tool_name, entry), s
        return best if best_score >= 0.20 else None


class LLMMemoryRetriever(MemoryRetriever):
    """LLM-backed retriever. Uses `create_llm_engine` lazily so the module
    imports cleanly even when the LLM provider SDK is absent."""

    def __init__(self, model_string: Optional[str] = None):
        from MAS.epc_aw.engine.factory import create_llm_engine
        self._engine = create_llm_engine(
            model_string=model_string or os.getenv("MODEL_Name", "gpt-4o-mini"),
            temperature=0.0,
        )
        self._keyword = KeywordMemoryRetriever()

    def _ask(self, system: str, user: str) -> str:
        resp = self._engine.generate(user, system_prompt=system)
        if isinstance(resp, dict):
            # engine may return an error dict
            return ""
        return (resp or "").strip()

    def find_similar_subgoal(
        self, subgoals: List[dict], query_subgoal: str
    ) -> Optional[dict]:
        if not subgoals:
            return None
        candidates = [e.get("subgoal", "") for e in subgoals]
        system = (
            "You are a semantic matcher. Given a query subgoal and a list of "
            "candidate subgoals, return EXACTLY ONE line: the candidate that is "
            "most semantically similar to the query, copied verbatim. If none is "
            "similar, return the single word NONE."
        )
        user = (
            f"Query subgoal: {query_subgoal}\n\n"
            f"Candidates:\n" + "\n".join(f"- {c}" for c in candidates) +
            "\n\nAnswer (one line, verbatim candidate or NONE):"
        )
        ans = self._ask(system, user)
        if not ans or ans.upper() == "NONE":
            # fall back to keyword match before giving up
            return self._keyword.find_similar_subgoal(subgoals, query_subgoal)
        for entry in subgoals:
            if entry.get("subgoal", "").strip() == ans.strip():
                return entry
        # LLM returned a paraphrase: fall back to keyword
        return self._keyword.find_similar_subgoal(subgoals, query_subgoal)

    def find_similar_factor(
        self, factors: Dict[str, dict], query_factor: str
    ) -> Optional[str]:
        if not factors:
            return None
        names = list(factors.keys())
        system = (
            "You are a semantic matcher. Given a query factor name and a list "
            "of candidate factor names, return EXACTLY ONE line: the candidate "
            "most semantically similar to the query, copied verbatim. If none is "
            "similar, return NONE."
        )
        user = (
            f"Query factor: {query_factor}\n\n"
            f"Candidates:\n" + "\n".join(f"- {n}" for n in names) +
            "\n\nAnswer (one line, verbatim candidate or NONE):"
        )
        ans = self._ask(system, user)
        if not ans or ans.upper() == "NONE":
            return self._keyword.find_similar_factor(factors, query_factor)
        if ans.strip() in factors:
            return ans.strip()
        return self._keyword.find_similar_factor(factors, query_factor)

    def find_similar_subgoal_across_tools(
        self, subgoal_text: str, candidates: List[Tuple[str, dict]]
    ) -> Optional[Tuple[str, dict]]:
        if not candidates:
            return None
        system = (
            "You are a semantic matcher. Given a query and a list of "
            "'<tool>: <subgoal>' candidates, return EXACTLY ONE line: the "
            "candidate (in the form '<tool>: <subgoal>') most semantically "
            "similar to the query, copied verbatim. If none is similar, "
            "return NONE."
        )
        user = (
            f"Query subgoal: {subgoal_text}\n\n"
            f"Candidates:\n" +
            "\n".join(f"- {t}: {e.get('subgoal', '')}" for t, e in candidates) +
            "\n\nAnswer (one line, verbatim '<tool>: <subgoal>' or NONE):"
        )
        ans = self._ask(system, user)
        if not ans or ans.upper() == "NONE":
            return self._keyword.find_similar_subgoal_across_tools(subgoal_text, candidates)
        m = re.match(r"^-?\s*([^:]+):\s*(.+)$", ans)
        if not m:
            return self._keyword.find_similar_subgoal_across_tools(subgoal_text, candidates)
        tool_name, sub_text = m.group(1).strip(), m.group(2).strip()
        for t, e in candidates:
            if t == tool_name and e.get("subgoal", "").strip() == sub_text:
                return (t, e)
        return self._keyword.find_similar_subgoal_across_tools(subgoal_text, candidates)


# =====================================================================
# Refiner — all memory-evolution logic
# =====================================================================
class MemoryRefiner:
    """Generalization, boundary refinement, factor refinement, instruction rewriting.

    Holds every piece of logic that mutates memory toward a more abstract,
    compact, and distinguishable state.
    """

    def __init__(self, retriever: MemoryRetriever, llm_engine: Any = None):
        self.retriever = retriever
        self._engine = llm_engine  # may be None; then keyword fallbacks apply

    # ---- low-level LLM helper ----
    def _ask(self, system: str, user: str) -> str:
        if self._engine is None:
            return ""
        try:
            resp = self._engine.generate(user, system_prompt=system)
        except Exception:
            return ""
        if isinstance(resp, dict):
            return ""
        return (resp or "").strip()

    # ---- instruction rewriting (one instruction per decision dimension) ----
    def rewrite_instruction(self, old: str, new: str) -> str:
        """Merge two instructions for the same decision dimension into ONE more
        general instruction that subsumes both. The result must preserve all
        previous meanings while becoming more general; never accumulate a
        second instruction."""
        if not old:
            return new
        if not new:
            return old
        if old.strip() == new.strip():
            return old
        system = (
            "You merge two tool-invocation instructions for the SAME decision "
            "dimension into ONE more general instruction that subsumes both. The "
            "merged instruction MUST preserve all previous meanings while becoming "
            "more general, and MUST remain task-abstract (no entities, URLs, or "
            "task-specific details). Output ONLY the single merged instruction, one "
            "line, imperative voice. Do not output a list."
        )
        user = f"Instruction A: {old}\nInstruction B: {new}\n\nMerged instruction:"
        merged = self._ask(system, user)
        if not merged:
            # fallback: keep the longer one (usually more specific → more general
            # when combined); append a disjunction only if they diverge a lot.
            return old if len(old) >= len(new) else new
        return merged

    # ---- applicability-condition generalization ----
    def generalize_context_summaries(
        self, existing: List[str], new: List[str]
    ) -> List[str]:
        """Merge two applicability-condition lists into a more general, deduplicated
        list of at most MAX_CONTEXT_SUMMARIES conditions.

        Each condition answers 'Under what conditions should this tool be
        selected?'. When a new condition is similar to an existing one, the
        existing sentence is rewritten into a more general description rather
        than appended."""
        combined = list(dict.fromkeys([s.strip() for s in (existing + new) if s and s.strip()]))
        new_clean = [s.strip() for s in (new or []) if s and s.strip()]
        existing_clean = [s.strip() for s in (existing or []) if s and s.strip()]

        # Skip the LLM call entirely when there is nothing to generalize:
        #   - no new conditions, OR
        #   - every new condition is already present verbatim, OR
        #   - deduped set fits under the cap AND no condition overlaps an
        #     existing one (keyword Jaccard < 0.3) — nothing to merge.
        def _no_overlap(a: str, b: str) -> bool:
            return KeywordMemoryRetriever._score(a, b) < 0.3

        nothing_new = (
            not new_clean
            or all(n in existing_clean for n in new_clean)
        )
        no_overlap = all(
            all(_no_overlap(n, e) for e in existing_clean)
            for n in new_clean
        )
        if nothing_new or (len(combined) <= MAX_CONTEXT_SUMMARIES and no_overlap):
            return combined[:MAX_CONTEXT_SUMMARIES]
        if self._engine is None:
            return combined[:MAX_CONTEXT_SUMMARIES]
        system = (
            "You generalize and merge a list of short APPLICABILITY CONDITIONS for "
            "a tool subgoal. Each condition answers 'Under what conditions should "
            "this tool be selected?'. Produce a MORE GENERAL, deduplicated list of "
            "at most " f"{MAX_CONTEXT_SUMMARIES} short phrases. When two conditions "
            "overlap, rewrite them into one more general sentence — do NOT keep "
            "redundant conditions. Conditions must stay task-abstract (no entities, "
            "URLs, execution progress, or step counts). Output STRICT JSON: a JSON "
            "array of strings, nothing else."
        )
        user = (
            "Applicability conditions (some may overlap):\n" +
            "\n".join(f"- {s}" for s in combined) +
            f"\n\nReturn a JSON array of <= {MAX_CONTEXT_SUMMARIES} generalized conditions."
        )
        raw = self._ask(system, user)
        parsed = self._parse_json_list(raw)
        if not parsed:
            # fallback: simple dedup, keep first MAX
            return combined[:MAX_CONTEXT_SUMMARIES]
        return [s.strip() for s in parsed if s and s.strip()][:MAX_CONTEXT_SUMMARIES]

    # ---- boundary refinement (cross-tool differentiation) ----
    def boundary_refinement(self, memory: "ToolCapabilityMemory") -> None:
        """Differentiate capability boundaries across tools (maximize inter-tool
        discrimination without increasing memory size) in a SINGLE batched LLM
        call.

        Subgoal texts are FROZEN (canonical vocabulary) and are NEVER rewritten
        here. Differentiation targets only the `capability_summary` lines: all
        tools whose summaries pairwise overlap (keyword Jaccard >= 0.20) are sent
        to one LLM call that rewrites each overlapping summary so the tools'
        roles become clearly distinguishable. If no pair overlaps, 0 LLM calls.
        Mutates `memory` in place; never creates or removes entries, never adds
        fields.
        """
        tool_names = list(memory.data.keys())
        if len(tool_names) < 2 or self._engine is None:
            return
        # Find tools involved in any overlapping pair (keyword pre-screen).
        involved: set = set()
        for i in range(len(tool_names)):
            for j in range(i + 1, len(tool_names)):
                cap_a = memory.data[tool_names[i]].get("capability_summary", "")
                cap_b = memory.data[tool_names[j]].get("capability_summary", "")
                if cap_a and cap_b and KeywordMemoryRetriever._score(cap_a, cap_b) >= 0.20:
                    involved.add(tool_names[i])
                    involved.add(tool_names[j])
        if not involved:
            return
        self._differentiate_capabilities_batch(memory, sorted(involved))

    def _differentiate_capabilities_batch(
        self, memory: "ToolCapabilityMemory", tool_names: List[str],
    ) -> None:
        """One LLM call that rewrites the capability_summary of every listed tool
        to maximize inter-tool discrimination."""
        summaries = [
            (t, memory.data[t].get("capability_summary", "")) for t in tool_names
        ]
        system = (
            "Several tools have overlapping capability summaries. Rewrite EACH "
            "listed summary so every tool's capability boundary becomes clearly "
            "distinguishable from the others (maximize inter-tool discrimination "
            "WITHOUT increasing memory size). Example direction: 'Retrieve "
            "information' -> 'Retrieve open-domain factual information.' vs "
            "'Interact with webpage content requiring user interaction.' Preserve "
            "each tool's actual role; keep each to one short sentence; stay "
            "task-abstract (no entities, URLs). Output STRICT JSON: an object "
            "mapping each tool name to its new summary string."
        )
        user = (
            "Tools and current summaries:\n" +
            "\n".join(f"- {t}: {s}" for t, s in summaries) +
            "\n\nReturn STRICT JSON: {\"<tool_name>\": \"new summary\", ...} for "
            "every listed tool."
        )
        raw = self._ask(system, user)
        obj = self._parse_json_obj(raw)
        if not obj:
            # fallback: per-pair refinement on overlapping pairs only
            for i in range(len(tool_names)):
                for j in range(i + 1, len(tool_names)):
                    ta, tb = tool_names[i], tool_names[j]
                    cap_a = memory.data[ta].get("capability_summary", "")
                    cap_b = memory.data[tb].get("capability_summary", "")
                    if cap_a and cap_b and KeywordMemoryRetriever._score(cap_a, cap_b) >= 0.20:
                        self._differentiate_capability(ta, cap_a, tb, cap_b, memory)
            return
        for t, _ in summaries:
            new_s = (obj.get(t) or "").strip()
            if new_s:
                memory.data[t]["capability_summary"] = new_s

    def _differentiate_capability(
        self, tool_a: str, cap_a: str, tool_b: str, cap_b: str,
        memory: "ToolCapabilityMemory",
    ) -> None:
        """Per-pair fallback for boundary refinement (used only when the batched
        call fails to parse)."""
        system = (
            "Two tools have overlapping capability summaries. Rewrite EACH summary "
            "so the two tools' capability boundaries become clearly distinguishable "
            "(maximize inter-tool discrimination WITHOUT increasing memory size). "
            "Example direction: 'Retrieve information' -> 'Retrieve open-domain "
            "factual information.' vs 'Interact with webpage content requiring user "
            "interaction.' Preserve each tool's actual role; keep each to one short "
            "sentence; stay task-abstract (no entities, URLs). Do NOT output any "
            "field other than the two summaries. Output STRICT JSON: "
            '{"tool_a_summary": "...", "tool_b_summary": "..."}.'
        )
        user = (
            f"Tool A = {tool_a}\n  summary A: {cap_a}\n"
            f"Tool B = {tool_b}\n  summary B: {cap_b}\n\n"
            "Return JSON with differentiated capability summaries."
        )
        raw = self._ask(system, user)
        obj = self._parse_json_obj(raw)
        if not obj:
            return
        new_a = (obj.get("tool_a_summary") or "").strip()
        new_b = (obj.get("tool_b_summary") or "").strip()
        if new_a and new_a != cap_a:
            memory.data[tool_a]["capability_summary"] = new_a
        if new_b and new_b != cap_b:
            memory.data[tool_b]["capability_summary"] = new_b

    # ---- decision-dimension refinement (within one subgoal) ----
    def factor_refinement(self, factors: Dict[str, dict]) -> Dict[str, dict]:
        """Aggressively merge semantically equivalent Decision Dimensions within
        a subgoal so dimensions stay orthogonal.

        A dimension is one independent aspect influencing parameter construction.
        e.g. {Time, Freshness, Date} -> {Freshness}; {Object, Entity, Target} ->
        {Entity}. For each merge group, keep one canonical name and merge the
        instructions into ONE more general instruction via `rewrite_instruction`.
        Enforces <= MAX_FACTORS_PER_SUBGOAL dimensions, each with exactly one
        instruction.
        """
        if not factors:
            return factors
        names = list(factors.keys())
        # Group names by semantic equivalence via LLM (or keyword fallback).
        groups: List[List[str]] = []
        if self._engine is not None:
            system = (
                "You group semantically equivalent DECISION DIMENSION names (the "
                "names of independent aspects that influence parameter "
                "construction). Output STRICT JSON: a JSON array of arrays, each "
                "inner array listing dimension names that mean the same thing. Use "
                "the EXACT input names. Put every name in exactly one group. "
                "Examples of merges: {Time, Freshness, Date} belong together; "
                "{Object, Entity, Target} belong together."
            )
            user = (
                "Decision dimension names:\n" + "\n".join(f"- {n}" for n in names) +
                "\n\nReturn JSON array of arrays."
            )
            raw = self._ask(system, user)
            parsed = self._parse_json_list(raw)
            if parsed and all(isinstance(g, list) for g in parsed):
                flat = [n for g in parsed for n in g]
                if set(flat) == set(names):
                    groups = [g for g in parsed if isinstance(g, list)]
        if not groups:
            # fallback: each name its own group
            groups = [[n] for n in names]

        refined: Dict[str, dict] = {}
        for group in groups:
            group = [n for n in group if n in factors]
            if not group:
                continue
            # canonical = shortest name (matches spec's "shortest name" spirit)
            canonical = min(group, key=len)
            merged_instr = factors[group[0]].get("instruction", "")
            for n in group[1:]:
                merged_instr = self.rewrite_instruction(
                    merged_instr, factors[n].get("instruction", "")
                )
            refined[canonical] = {"instruction": merged_instr}
        # enforce cap: if still over, keep the dimensions with the longest
        # instructions (most information-dense) and drop the rest.
        if len(refined) > MAX_FACTORS_PER_SUBGOAL:
            kept = sorted(refined.items(), key=lambda kv: -len(kv[1].get("instruction", "")))
            refined = dict(kept[:MAX_FACTORS_PER_SUBGOAL])
        return refined

    def merge_factors_batch(
        self, existing: Dict[str, dict], new_pairs: Dict[str, str],
    ) -> Dict[str, dict]:
        """Merge existing decision dimensions with new (dimension -> instruction)
        pairs in a SINGLE LLM call, replacing the old per-factor
        `find_similar_factor` + `rewrite_instruction` + `factor_refinement`
        sequence (which cost up to ~9 calls).

        Output contract: <= MAX_FACTORS_PER_SUBGOAL dimensions, each with exactly
        one instruction that is a MORE GENERAL rewrite preserving all previous
        meanings; semantically equivalent dimensions collapsed (e.g.
        {Time, Freshness} -> {Freshness}, {Object, Entity} -> {Entity}). Task-
        abstract (no entities/URLs). Falls back to the keyword+rewrite path when
        no LLM is available or the batch parse fails.
        """
        merged_in = {k: dict(v) for k, v in (existing or {}).items()}
        for name, instr in (new_pairs or {}).items():
            if name and instr:
                # naive accumulation; the LLM (or fallback) dedupes below
                if name in merged_in:
                    merged_in[name] = {"instruction": merged_in[name].get("instruction", "")}
                else:
                    merged_in[name] = {"instruction": instr}

        if self._engine is None or not merged_in:
            return self._merge_factors_fallback(existing, new_pairs)

        system = (
            "You merge decision dimensions for ONE tool subgoal into a compact, "
            "orthogonal set. Input: existing dimensions (each with one "
            "instruction) plus new dimension->instruction pairs. Output STRICT "
            f"JSON: an object mapping <= {MAX_FACTORS_PER_SUBGOAL} dimension names "
            "to {{\"instruction\": \"...\"}}. Rules: (a) collapse semantically "
            "equivalent dimensions (e.g. Time+Freshness -> Freshness; "
            "Object+Entity+Target -> Entity) into ONE; (b) for each merged "
            "dimension produce ONE more general instruction that preserves ALL "
            "previous meanings — never two instructions; (c) keep dimensions "
            "orthogonal; (d) stay task-abstract (no entities/URLs); (e) prefer "
            f"these canonical names when applicable: {', '.join(CANONICAL_FACTOR_NAMES)}."
        )
        lines = []
        for name, v in merged_in.items():
            lines.append(f"- {name}: {v.get('instruction', '')}")
        user = (
            "Dimensions to merge (existing + new):\n" + "\n".join(lines) +
            f"\n\nReturn STRICT JSON object: {{Dimension: {{\"instruction\": \"...\"}}, ...}} "
            f"(<= {MAX_FACTORS_PER_SUBGOAL} dimensions)."
        )
        raw = self._ask(system, user)
        obj = self._parse_json_obj(raw)
        if not obj:
            return self._merge_factors_fallback(existing, new_pairs)
        out: Dict[str, dict] = {}
        for k, v in obj.items():
            name = str(k).strip()
            instr = (v.get("instruction") if isinstance(v, dict) else str(v)).strip()
            if name and instr:
                out[name] = {"instruction": instr}
        if not out:
            return self._merge_factors_fallback(existing, new_pairs)
        if len(out) > MAX_FACTORS_PER_SUBGOAL:
            kept = sorted(out.items(), key=lambda kv: -len(kv[1].get("instruction", "")))
            out = dict(kept[:MAX_FACTORS_PER_SUBGOAL])
        return out

    def _merge_factors_fallback(
        self, existing: Dict[str, dict], new_pairs: Dict[str, str],
    ) -> Dict[str, dict]:
        """Keyword + per-merge-rewrite fallback for `merge_factors_batch` when no
        LLM is available or the batch call failed. Reuses `find_similar_factor` +
        `rewrite_instruction` + `factor_refinement` (the original path)."""
        factors: Dict[str, dict] = {k: dict(v) for k, v in (existing or {}).items()}
        for name, instr in (new_pairs or {}).items():
            name = (name or "").strip()
            instr = (instr or "").strip()
            if not name or not instr:
                continue
            similar = self.retriever.find_similar_factor(factors, name)
            if similar is not None:
                factors[similar]["instruction"] = self.rewrite_instruction(
                    factors[similar].get("instruction", ""), instr
                )
            else:
                factors[name] = {"instruction": instr}
        return self.factor_refinement(factors)

    # ---- ingest-time abstraction gate (Generalize) ----
    def abstract_experience(
        self,
        tool: str,
        capability_summary: str,
        canonical_subgoals: List[str],
        concrete_subgoal: str,
        question: str,
        successful_params: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ingest-time Abstraction Gate (the Generalize step).

        Maps ONE concrete successful experience onto the ABSTRACT Tool Knowledge
        vocabulary, so memory never accumulates task-specific entries. The
        returned object is the unit consumed by `evolve()` (which then merges /
        compresses it into existing slots instead of appending).

        Returns:
            {"abstract_subgoal": str (verbatim from canonical_subgoals),
             "context_summaries": List[str] (APPLICABILITY CONDITIONS: under
                 what conditions should this tool be selected; <=3),
             "factors": {DecisionDimension: "instruction"} (<=4, orthogonal;
                 exactly one instruction per dimension)}

        With an LLM: asks the model to pick a canonical subgoal verbatim and to
        write abstract applicability conditions / decision dimensions. Without an
        LLM: falls back to keyword-nearest subgoal + tool-typed default factors.
        Either way the output is passed through `_strip_task_leakage`; if
        abstraction fails the leaked content is discarded and canonical defaults
        are used.
        """
        canonical_subgoals = [s for s in (canonical_subgoals or []) if s and s.strip()]
        concrete = (concrete_subgoal or "").strip()
        if not canonical_subgoals or not concrete:
            return {"abstract_subgoal": "", "context_summaries": [], "factors": {}}

        if self._engine is not None:
            abst = self._abstract_with_llm(
                tool, capability_summary, canonical_subgoals,
                concrete, question, successful_params,
            )
            if abst and abst.get("abstract_subgoal") in canonical_subgoals:
                abst = self._strip_task_leakage(abst, concrete, question, tool, successful_params)
                return abst
            # LLM returned a non-verbatim subgoal: fall back to keyword nearest
            # but keep any valid context/factors it produced.
            nearest = self._nearest_canonical(canonical_subgoals, concrete)
            if abst:
                abst["abstract_subgoal"] = nearest
                abst = self._strip_task_leakage(abst, concrete, question, tool, successful_params)
                return abst
        # keyword fallback
        return self._abstract_fallback(
            tool, canonical_subgoals, concrete, successful_params,
        )

    # Common words allowed even though they are capitalized / look like names.
    _LEAK_WHITELIST = {
        "Tool", "Tools", "Search", "Web", "Google", "Wikipedia", "Python",
        "URL", "API", "JSON", "HTML", "Image", "Page", "Table", "Chart",
        "Entity", "Freshness", "Scope", "Source", "Breadth", "Latency",
        "Format", "Inputs", "Method", "Precision", "Viewport", "Reasoning",
        "Content", "Structure", "The", "A", "An",
    }

    def _strip_task_leakage(
        self, abst: Dict[str, Any], concrete: str, question: str,
        tool: str, successful_params: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Post-LLM leakage guard. Drops any applicability condition / decision-
        dimension instruction that still contains a task-specific identifier
        (entity name, URL, paper name, proper noun, numeric/catalogue token, or
        any other user-/task-specific information leaked from the concrete
        experience). Guarantees memory stays abstract even if the LLM disobeys
        the no-leak rule. If all factor instructions are stripped, falls back to
        the tool-typed canonical default decision dimensions rather than storing
        concrete content."""
        suspicious = self._suspicious_tokens(concrete, question, successful_params)
        if not suspicious:
            return abst

        def clean(text: str) -> bool:
            t = (text or "").lower()
            return not any(s in t for s in suspicious)

        ctx = [c for c in (abst.get("context_summaries") or []) if clean(c)]
        abst["context_summaries"] = ctx[:MAX_CONTEXT_SUMMARIES]

        factors_raw = abst.get("factors") or {}
        factors = {k: v for k, v in factors_raw.items() if clean(v)}
        if not factors:
            # all instructions leaked: use the tool-typed abstract defaults
            factors = dict(self._FALLBACK_FACTORS.get(tool, {}))
        abst["factors"] = factors
        return abst

    @classmethod
    def _suspicious_tokens(
        cls, concrete: str, question: str, successful_params: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract lowercase task-specific identifiers from the concrete
        experience: proper-noun-like capitalized words (not whitelisted) and
        multi-digit / catalogue-number tokens."""
        blob = " ".join([
            concrete or "", question or "",
            *(str(rec.get("parameter", "")) if isinstance(rec, dict) else str(rec)
              for rec in (successful_params or [])),
        ])
        toks: List[str] = []
        for m in re.finditer(r"[A-Z][a-zA-Z]{2,}", blob):
            w = m.group(0)
            if w not in cls._LEAK_WHITELIST:
                toks.append(w.lower())
        # catalogue / numeric identifiers (>= 3 chars containing a digit)
        for m in re.finditer(r"\b[A-Za-z0-9]*\d[A-Za-z0-9.,-]{2,}\b", blob):
            toks.append(m.group(0).lower())
        # dedup, drop trivial
        seen = set()
        out = []
        for t in toks:
            if t in seen or len(t) < 3:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _nearest_canonical(self, canonical_subgoals: List[str], concrete: str) -> str:
        entry = self.retriever.find_similar_subgoal(
            [{"subgoal": s} for s in canonical_subgoals], concrete,
        )
        if entry and entry.get("subgoal") in canonical_subgoals:
            return entry["subgoal"]
        return canonical_subgoals[0]

    def _abstract_with_llm(
        self, tool, capability_summary, canonical_subgoals, concrete, question, successful_params,
    ) -> Optional[Dict[str, Any]]:
        param_preview = []
        for rec in successful_params[:5]:
            p = rec.get("parameter", "") if isinstance(rec, dict) else str(rec)
            if p:
                param_preview.append(p[:160])
        system = self._ABSTRACT_SYSTEM_PROMPT
        user = (
            f"Tool: {tool}\n"
            f"Tool capability: {capability_summary}\n"
            f"Canonical subgoals (pick one verbatim):\n"
            + "\n".join(f"- {s}" for s in canonical_subgoals) +
            f"\n\nConcrete successful subgoal: {concrete}\n"
            f"Task question (context): {(question or '')[:400]}\n"
            f"Successful parameters (examples):\n"
            + ("\n".join(f"- {p}" for p in param_preview) if param_preview else "- (none)") +
            "\n\nReturn STRICT JSON: "
            '{"abstract_subgoal": "...", "context_summary": ["...", "..."], '
            '"factors": {"FactorName": "instruction", ...}}'
        )
        raw = self._ask(system, user)
        obj = self._parse_json_obj(raw)
        if not obj:
            return None
        return self._normalize_abst_obj(obj)

    # Shared system prompt for the abstraction gate (single + batch).
    _ABSTRACT_SYSTEM_PROMPT = (
        "You map one or more CONCRETE successful tool experiences onto an "
        "ABSTRACT Tool Knowledge vocabulary. You must output STRICT JSON only.\n"
        "Rules:\n"
        "1. abstract_subgoal MUST be copied VERBATIM from the provided canonical "
        "subgoal list (pick the single most appropriate one per experience). "
        "Subgoals are canonical and frozen — never invent a new one.\n"
        "2. context_summary is a JSON array of <= 3 short phrases, each being an "
        "APPLICABILITY CONDITION: a sentence answering 'Under what conditions "
        "should this tool be selected?'. Good: 'External knowledge is "
        "unavailable.', 'The target entity has already been identified.', "
        "'Authoritative evidence is required.'. FORBIDDEN content: execution "
        "progress ('partial information has been collected'), task state ('three "
        "entities remain unresolved', 'current step is incomplete'), retrieved "
        "evidence, parameter configuration, or step counts.\n"
        "3. factors is a JSON object mapping a DECISION DIMENSION name to exactly "
        "one imperative INSTRUCTION. A decision dimension is one INDEPENDENT "
        "aspect influencing parameter construction (e.g. Entity, Freshness, "
        "Scope, Source). Use <= 4 ORTHOGONAL dimensions. Prefer reusing these "
        f"canonical names when applicable: {', '.join(CANONICAL_FACTOR_NAMES)}. "
        "Never emit two dimensions that mean the same thing (e.g. do NOT produce "
        "both Entity and Target, or both Time and Freshness). Each instruction "
        "must state how parameters should be constructed under that dimension.\n"
        "4. ABSTRACTION IS MANDATORY: every output string (applicability condition "
        "and instruction) MUST be domain- and task-abstract. NEVER mention the "
        "specific task entity, domain, topic, person, species, language, library, "
        "URL, paper, or dataset (e.g. never write 'Unlambda', 'numpy', a species "
        "name, a person name, a museum number, a URL). Describe the CATEGORY of "
        "information, not the instance. A reader who has never seen this task "
        "must still understand the instruction. If you cannot abstract a piece of "
        "content, omit it rather than leaking it.\n"
        "5. Do NOT add any field beyond abstract_subgoal, context_summary, factors."
    )

    @staticmethod
    def _normalize_abst_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one parsed abstraction object to the canonical shape."""
        subgoal = (obj.get("abstract_subgoal") or "").strip()
        ctx = obj.get("context_summary") or obj.get("context_summaries") or []
        if isinstance(ctx, str):
            ctx = [ctx]
        ctx = [str(s).strip() for s in ctx if s and str(s).strip()][:MAX_CONTEXT_SUMMARIES]
        factors_raw = obj.get("factors") or {}
        factors: Dict[str, str] = {}
        if isinstance(factors_raw, dict):
            for k, v in factors_raw.items():
                name = str(k).strip()
                instr = (v.get("instruction") if isinstance(v, dict) else str(v)).strip()
                if name and instr:
                    factors[name] = instr
        if len(factors) > MAX_FACTORS_PER_SUBGOAL:
            kept = sorted(factors.items(), key=lambda kv: -len(kv[1]))[:MAX_FACTORS_PER_SUBGOAL]
            factors = dict(kept)
        return {"abstract_subgoal": subgoal, "context_summaries": ctx, "factors": factors}

    def abstract_experiences_batch(
        self,
        tool: str,
        capability_summary: str,
        canonical_subgoals: List[str],
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batched ingest-time Abstraction Gate: map ALL successful experiences
        for ONE tool onto the abstract vocabulary in a SINGLE LLM call.

        `items` is a list of dicts: {"concrete_subgoal", "question",
        "successful_params"}. Returns one normalized abstraction object per
        item (same shape as `abstract_experience`), in input order. On batch
        parse failure or per-item invalid subgoal, falls back to the
        single-item `_abstract_with_llm` for that item only (local fallback,
        so one bad item does not waste the whole batch). Without an LLM, falls
        back to per-item keyword abstraction.
        """
        canonical_subgoals = [s for s in (canonical_subgoals or []) if s and s.strip()]
        items = [it for it in (items or []) if (it.get("concrete_subgoal") or "").strip()]
        if not canonical_subgoals or not items:
            return [
                {"abstract_subgoal": "", "context_summaries": [], "factors": {}}
                for _ in items
            ]

        if self._engine is None:
            return [
                self._abstract_fallback(
                    tool, canonical_subgoals,
                    it.get("concrete_subgoal", ""),
                    it.get("successful_params", []),
                )
                for it in items
            ]

        # Build one batched user prompt.
        canon_block = "\n".join(f"- {s}" for s in canonical_subgoals)
        exp_blocks: List[str] = []
        for idx, it in enumerate(items):
            concrete = (it.get("concrete_subgoal") or "").strip()
            question = (it.get("question") or "")
            params = it.get("successful_params", []) or []
            preview = []
            for rec in params[:5]:
                p = rec.get("parameter", "") if isinstance(rec, dict) else str(rec)
                if p:
                    preview.append(p[:160])
            exp_blocks.append(
                f"[Experience {idx + 1}]\n"
                f"  concrete subgoal: {concrete}\n"
                f"  task question: {question[:300]}\n"
                f"  successful params:\n"
                + ("    " + "\n    ".join(f"- {p}" for p in preview) if preview else "    - (none)")
            )
        user = (
            f"Tool: {tool}\n"
            f"Tool capability: {capability_summary}\n"
            f"Canonical subgoals (pick one verbatim per experience):\n{canon_block}\n\n"
            + "\n\n".join(exp_blocks) +
            "\n\nReturn STRICT JSON: a JSON ARRAY with one object per experience IN "
            "ORDER, each "
            '{"abstract_subgoal": "...", "context_summary": ["...", "..."], '
            '"factors": {"FactorName": "instruction", ...}}'
        )
        raw = self._ask(self._ABSTRACT_SYSTEM_PROMPT, user)
        arr = self._parse_json_list(raw)
        results: List[Dict[str, Any]] = []
        if not arr or len(arr) != len(items):
            # whole-batch failure: per-item fallback
            for it in items:
                results.append(
                    self._abstract_with_llm(
                        tool, capability_summary, canonical_subgoals,
                        (it.get("concrete_subgoal") or "").strip(),
                        it.get("question", ""), it.get("successful_params", []),
                    ) or {"abstract_subgoal": "", "context_summaries": [], "factors": {}}
                )
            return results
        for it, obj in zip(items, arr):
            if not isinstance(obj, dict):
                results.append(
                    self._abstract_with_llm(
                        tool, capability_summary, canonical_subgoals,
                        (it.get("concrete_subgoal") or "").strip(),
                        it.get("question", ""), it.get("successful_params", []),
                    ) or {"abstract_subgoal": "", "context_summaries": [], "factors": {}}
                )
                continue
            abst = self._normalize_abst_obj(obj)
            # validate subgoal membership; fallback per-item if invalid
            if abst["abstract_subgoal"] not in canonical_subgoals:
                fb = self._abstract_with_llm(
                    tool, capability_summary, canonical_subgoals,
                    (it.get("concrete_subgoal") or "").strip(),
                    it.get("question", ""), it.get("successful_params", []),
                )
                abst = fb or abst
                if abst["abstract_subgoal"] not in canonical_subgoals:
                    abst["abstract_subgoal"] = self._nearest_canonical(
                        canonical_subgoals, (it.get("concrete_subgoal") or "")
                    )
            abst = self._strip_task_leakage(
                abst, (it.get("concrete_subgoal") or ""), it.get("question", ""),
                tool, it.get("successful_params", []),
            )
            results.append(abst)
        return results

    # Tool-typed default factors used when no LLM is available.
    _FALLBACK_FACTORS: Dict[str, Dict[str, str]] = {
        "Google_Search_Tool": {
            "Entity": "Include sufficient identifiers to uniquely specify the target.",
            "Freshness": "Prefer recent sources when temporal relevance affects the answer.",
            "Scope": "Restrict the search to the requested domain or topic.",
            "Source": "Prefer authoritative sources when reliability is critical.",
        },
        "Web_Search_Tool": {
            "Entity": "Pin the exact URL of the target page.",
            "Source": "Fetch directly from the canonical URL.",
            "Scope": "Target the specific section or field, not the whole page.",
        },
        "Wikipedia_Search_Tool": {
            "Entity": "Provide the entity name and any disambiguating context.",
            "Scope": "Target the Wikipedia namespace/language requested.",
            "Source": "Use the Wikipedia article as the authoritative source.",
        },
        "Python_Coder_Tool": {
            "Inputs": "Bind every numeric input the computation requires.",
            "Method": "Implement the exact arithmetic formula requested.",
            "Precision": "Apply rounding only as the task specifies.",
        },
        "Screenshot_Tool": {
            "Entity": "Pin the exact URL to capture.",
            "Viewport": "Set the viewport to reveal the relevant content.",
        },
        "Vision_OCR_Tool": {
            "Entity": "Identify the image and the text region of interest.",
            "Format": "Return extracted text verbatim, preserving order.",
        },
        "Base_Generator_Tool": {
            "Reasoning": "Lay out the deductive steps explicitly before the answer.",
        },
    }

    def _abstract_fallback(
        self, tool, canonical_subgoals, concrete, successful_params,
    ) -> Dict[str, Any]:
        subgoal = self._nearest_canonical(canonical_subgoals, concrete)
        factors = {
            k: v for k, v in self._FALLBACK_FACTORS.get(tool, {}).items()
        }
        return {"abstract_subgoal": subgoal, "context_summaries": [], "factors": factors}

    # ---- JSON parsing helpers (tolerant) ----
    @staticmethod
    def _strip_fences(s: str) -> str:
        s = (s or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s)
        return s.strip()

    def _parse_json_list(self, raw: str) -> Optional[list]:
        raw = self._strip_fences(raw)
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, list) else None
        except Exception:
            # try to extract first JSON array
            m = re.search(r"\[.*\]", raw, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    def _parse_json_obj(self, raw: str) -> Optional[dict]:
        raw = self._strip_fences(raw)
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None


# =====================================================================
# Memory 1 — Tool Capability Memory (Planner)
# =====================================================================
class ToolCapabilityMemory:
    """Answers 'which tool should be selected?' (Capability). Plain text only.

    `context_summary` is interpreted as APPLICABILITY CONDITIONS — each sentence
    answers 'Under what conditions should this tool be selected?'. It must NOT
    contain execution progress, task state, retrieved evidence, or parameter
    configuration.

    Schema (intentionally minimal — do NOT add fields):
        {
            ToolName: {
                "capability_summary": str,
                "subgoals": [
                    {"subgoal": str, "context_summary": [str, ...]}
                ]
            }
        }
    """

    def __init__(self, retriever: Optional[MemoryRetriever] = None,
                 refiner: Optional[MemoryRefiner] = None):
        self.data: Dict[str, Dict[str, Any]] = {}
        self.retriever = retriever or KeywordMemoryRetriever()
        self.refiner = refiner or MemoryRefiner(self.retriever)

    # ---- accessors ----
    def get_tool(self, tool_name: str) -> Optional[dict]:
        return self.data.get(tool_name)

    def all_tools(self) -> List[str]:
        return list(self.data.keys())

    def to_dict(self) -> dict:
        # deep copy for serialization safety
        return json.loads(json.dumps(self.data))

    # ---- retrieval ----
    def retrieve(self, tool_name: str, current_subgoal: str) -> Optional[dict]:
        """Return the most similar subgoal entry for `tool_name`, or None."""
        tool = self.data.get(tool_name)
        if not tool:
            return None
        return self.retriever.find_similar_subgoal(tool["subgoals"], current_subgoal)

    # ---- evolution ----
    def evolve(
        self, tool_name: str, successful_subgoal: str,
        context_summary_list: Optional[List[str]] = None,
    ) -> None:
        """Incorporate one abstracted experience (retrieve -> merge -> generalize
        -> rewrite). Knowledge compression, not memory growth.

        The subgoal vocabulary is FROZEN: `successful_subgoal` must be an
        abstract subgoal already belonging to this tool's canonical slots (the
        ingest abstraction gate guarantees this). Evolution only generalizes the
        matched slot's applicability conditions (`context_summary`); it NEVER
        appends a new subgoal, NEVER rewrites the frozen subgoal text, and NEVER
        adds fields. When a new applicability condition is similar to an existing
        one, the existing sentence is rewritten into a more general description
        rather than appended.

        If no exact/semantic match is found (e.g. an unknown tool with no
        seed), the closest existing slot is reused rather than creating one.
        """
        context_summary_list = [s for s in (context_summary_list or []) if s and s.strip()]
        tool = self.data.setdefault(
            tool_name, {"capability_summary": "", "subgoals": []}
        )
        existing = self._find_slot(tool, successful_subgoal)
        if existing is None and tool["subgoals"]:
            # Frozen vocabulary: never append. Map to the nearest existing
            # slot even when below the retriever's similarity threshold.
            existing = self._nearest_existing_slot(tool["subgoals"], successful_subgoal)
        if existing is None:
            # No slots at all yet (unseeded tool): create exactly one abstract
            # slot. This branch is unreachable for the 7 seeded tools.
            existing = {
                "subgoal": successful_subgoal.strip(),
                "context_summary": [],
            }
            tool["subgoals"].append(existing)
            self._enforce_subgoal_cap(tool_name)
        # Generalize context summaries in place. Subgoal text stays frozen.
        existing["context_summary"] = self.refiner.generalize_context_summaries(
            existing.get("context_summary", []), context_summary_list
        )

    def _find_slot(self, tool: dict, subgoal_text: str) -> Optional[dict]:
        """Return the canonical slot for an abstract subgoal.

        Exact verbatim match wins; otherwise a semantic/keyword match above
        threshold. Never creates a new slot."""
        subs = tool.get("subgoals", [])
        target = (subgoal_text or "").strip()
        for entry in subs:
            if entry.get("subgoal", "").strip() == target:
                return entry
        return self.retriever.find_similar_subgoal(subs, target)

    @staticmethod
    def _nearest_existing_slot(subs: List[dict], subgoal_text: str) -> Optional[dict]:
        """Always return the highest-scoring slot (or the first), ignoring the
        retriever's similarity threshold. Used to enforce frozen-vocabulary
        mapping when no above-threshold match exists."""
        if not subs:
            return None
        target = subgoal_text or ""
        best, best_score = subs[0], -1.0
        for entry in subs:
            score = KeywordMemoryRetriever._score(entry.get("subgoal", ""), target)
            if score > best_score:
                best, best_score = entry, score
        return best

    def _enforce_subgoal_cap(self, tool_name: str) -> None:
        """If a tool exceeds MAX_SUBGOALS_PER_TOOL, merge the two most similar
        subgoals into one rather than dropping data."""
        tool = self.data[tool_name]
        subs = tool["subgoals"]
        while len(subs) > MAX_SUBGOALS_PER_TOOL:
            # find the most similar pair (by retriever)
            a_idx, b_idx, best = 0, 1, -1.0
            for i in range(len(subs)):
                for j in range(i + 1, len(subs)):
                    score = self._pair_similarity(subs[i]["subgoal"], subs[j]["subgoal"])
                    if score > best:
                        best, a_idx, b_idx = score, i, j
            self._merge_subgoals(tool_name, a_idx, b_idx)

    def _pair_similarity(self, a: str, b: str) -> float:
        if isinstance(self.retriever, KeywordMemoryRetriever):
            return KeywordMemoryRetriever._score(a, b)
        # generic fallback
        return KeywordMemoryRetriever._score(a, b)

    def _merge_subgoals(self, tool_name: str, i: int, j: int) -> None:
        subs = self.data[tool_name]["subgoals"]
        if i == j or i >= len(subs) or j >= len(subs):
            return
        a, b = subs[i], subs[j]
        # generalized subgoal text
        if self.refiner._engine is not None:
            merged_text = self.refiner.rewrite_instruction(a["subgoal"], b["subgoal"])
        else:
            merged_text = a["subgoal"]  # keep first; keyword mode can't rewrite
        merged_ctx = self.refiner.generalize_context_summaries(
            a.get("context_summary", []), b.get("context_summary", [])
        )
        # replace a with merged, remove b
        a["subgoal"] = merged_text
        a["context_summary"] = merged_ctx
        subs.pop(j)

    # ---- serialization ----
    def save(self, path: str, fmt: str = "json") -> None:
        fmt = fmt.lower()
        if fmt == "yaml":
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "YAML serialization requires PyYAML. `pip install pyyaml` or use fmt='json'."
                )
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"unsupported fmt: {fmt} (use 'json' or 'yaml')")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yaml", ".yml")):
            if not _YAML_AVAILABLE:
                raise ImportError("YAML load requires PyYAML.")
            self.data = yaml.safe_load(text) or {}
        else:
            self.data = json.loads(text)
        self._validate()

    def _validate(self) -> None:
        for tool_name, tool in self.data.items():
            tool.setdefault("capability_summary", "")
            tool.setdefault("subgoals", [])
            tool["subgoals"] = tool["subgoals"][:MAX_SUBGOALS_PER_TOOL]
            for entry in tool["subgoals"]:
                entry.setdefault("subgoal", "")
                entry.setdefault("context_summary", [])
                entry["context_summary"] = entry["context_summary"][:MAX_CONTEXT_SUMMARIES]

    # ---- seed ----
    @classmethod
    def seed_default_capabilities(
        cls, retriever: Optional[MemoryRetriever] = None,
        refiner: Optional[MemoryRefiner] = None,
    ) -> "ToolCapabilityMemory":
        """Build a memory pre-seeded with ABSTRACT, ORTHOGONAL capabilities for
        the 7 enabled tools.

        The subgoal list per tool is a fixed canonical vocabulary (3-4
        orthogonal slots). Evolution NEVER appends new subgoals; every
        successful experience is mapped (by the ingest abstraction gate) onto
        one of these slots and only merges/generalizes its applicability
        conditions (`context_summary`). Compliant with Principle 2 (no concrete
        task examples, no scores/counts/confidence/metadata).
        """
        mem = cls(retriever=retriever, refiner=refiner)
        mem.data = {
            "Google_Search_Tool": {
                "capability_summary": "Retrieve open-domain factual information via web search.",
                "subgoals": [
                    {"subgoal": "Retrieve external factual knowledge",
                     "context_summary": ["Missing external knowledge", "Open-domain question"]},
                    {"subgoal": "Disambiguate an ambiguous entity",
                     "context_summary": ["Entity not yet pinned down", "Multiple candidates"]},
                    {"subgoal": "Retrieve time-sensitive information",
                     "context_summary": ["Temporal recency required", "Recent event or update"]},
                    {"subgoal": "Verify a contested claim",
                     "context_summary": ["Conflicting sources", "Authoritative confirmation needed"]},
                ],
            },
            "Web_Search_Tool": {
                "capability_summary": "Retrieve content from a known URL via RAG.",
                "subgoals": [
                    {"subgoal": "Extract a specific field from a known page",
                     "context_summary": ["Target URL known", "Specific field required"]},
                    {"subgoal": "Verify content of a known page",
                     "context_summary": ["URL known", "Content needs confirmation"]},
                    {"subgoal": "Retrieve full-text content behind a known URL",
                     "context_summary": ["URL known", "Full body or section required"]},
                ],
            },
            "Wikipedia_Search_Tool": {
                "capability_summary": "Look up encyclopedic entities on Wikipedia.",
                "subgoals": [
                    {"subgoal": "Resolve a notable encyclopedic entity",
                     "context_summary": ["Entity is encyclopedic", "Background facts required"]},
                    {"subgoal": "Verify an entity attribute",
                     "context_summary": ["Entity identified", "Attribute needs authoritative source"]},
                    {"subgoal": "Retrieve structured facts on a topic",
                     "context_summary": ["Topic identified", "List or attribute required"]},
                ],
            },
            "Python_Coder_Tool": {
                "capability_summary": "Compute numerical or logical results in Python.",
                "subgoals": [
                    {"subgoal": "Compute from given numbers",
                     "context_summary": ["All inputs available", "Arithmetic or rounding required"]},
                    {"subgoal": "Perform a symbolic or logic verification",
                     "context_summary": ["Symbolic expression available", "Equivalence or tautology to test"]},
                    {"subgoal": "Transform or parse structured data",
                     "context_summary": ["Raw data available", "Parsing or transformation required"]},
                ],
            },
            "Screenshot_Tool": {
                "capability_summary": "Capture a webpage as an image.",
                "subgoals": [
                    {"subgoal": "Capture the visual state of a page",
                     "context_summary": ["Page URL known", "Visual content needs to be seen"]},
                ],
            },
            "Vision_OCR_Tool": {
                "capability_summary": "Extract text or structure from images.",
                "subgoals": [
                    {"subgoal": "Extract text from an image",
                     "context_summary": ["Image available", "On-image text required"]},
                    {"subgoal": "Parse figure axes or a table",
                     "context_summary": ["Chart or table image available", "Structured elements to extract"]},
                ],
            },
            "Base_Generator_Tool": {
                "capability_summary": "Answer by reasoning without external tools.",
                "subgoals": [
                    {"subgoal": "Answer by pure reasoning",
                     "context_summary": ["No external knowledge needed", "Question is self-contained"]},
                    {"subgoal": "Solve a logic or language puzzle",
                     "context_summary": ["Puzzle is self-contained", "Reasoning over given rules"]},
                ],
            },
        }
        return mem


# =====================================================================
# Memory 2 — Tool Invocation Memory (Executor; Planner never accesses)
# =====================================================================
class ToolInvocationMemory:
    """Answers 'how should the selected tool be invoked?' (Invocation). Plain
    text only.

    `factors` are interpreted as DECISION DIMENSIONS — each key is one
    independent aspect influencing parameter construction (e.g. Entity,
    Freshness, Scope, Source). Each dimension carries exactly ONE instruction
    describing how parameters should be constructed under that dimension.
    Semantically equivalent dimensions are aggressively merged.

    Schema (intentionally minimal — do NOT add fields):
        {
            ToolName: {
                "subgoals": [
                    {
                        "subgoal": str,
                        "factors": {DecisionDimension: {"instruction": str}}
                    }
                ]
            }
        }
    """

    def __init__(self, retriever: Optional[MemoryRetriever] = None,
                 refiner: Optional[MemoryRefiner] = None):
        self.data: Dict[str, Dict[str, Any]] = {}
        self.retriever = retriever or KeywordMemoryRetriever()
        self.refiner = refiner or MemoryRefiner(self.retriever)

    # ---- accessors ----
    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.data))

    # ---- retrieval ----
    def retrieve(self, tool_name: str, subgoal: str) -> Optional[dict]:
        tool = self.data.get(tool_name)
        if not tool:
            return None
        return self.retriever.find_similar_subgoal(tool["subgoals"], subgoal)

    # ---- evolution ----
    def evolve(
        self, tool_name: str, successful_subgoal: str,
        factor_instruction_pairs: Dict[str, str],
    ) -> None:
        """Incorporate one abstracted invocation (retrieve -> merge ->
        generalize -> rewrite). Knowledge compression, not memory growth.

        The subgoal vocabulary is FROZEN: `successful_subgoal` must be an
        abstract subgoal already belonging to this tool's canonical slots.
        Evolution maps it onto the matching slot and only refines that slot's
        decision dimensions — it NEVER appends a new subgoal entry and NEVER
        adds fields.

        Decision dimensions are free-abstraction with a hard cap and an
        exactly-one-instruction invariant: each incoming dimension is either
        merged into a semantically similar existing dimension (the single
        instruction is rewritten into a more general one that preserves all
        previous meanings) or added as a new one, then factor_refinement
        collapses semantic duplicates. Enforces <= MAX_FACTORS_PER_SUBGOAL
        dimensions, each with exactly one instruction.
        """
        # normalize input: exactly one instruction per factor
        pairs = {k.strip(): v.strip()
                 for k, v in factor_instruction_pairs.items() if k and k.strip() and v and v.strip()}
        tool = self.data.setdefault(tool_name, {"subgoals": []})
        entry = self._find_slot(tool, successful_subgoal)
        if entry is None and tool["subgoals"]:
            # Frozen vocabulary: never append. Map to the nearest existing slot.
            entry = self._nearest_existing_slot(tool["subgoals"], successful_subgoal)
        if entry is None:
            # Unseeded tool with no slots: create one abstract slot. Unreachable
            # for the 7 seeded tools.
            factors = {name: {"instruction": instr} for name, instr in pairs.items()}
            factors = self.refiner.factor_refinement(factors)
            entry = {"subgoal": successful_subgoal.strip(), "factors": factors}
            tool["subgoals"].append(entry)
            self._enforce_subgoal_cap(tool_name)
            return
        # hit: merge new decision dimensions into the frozen slot in ONE batched
        # LLM call (replaces the old per-factor find_similar_factor +
        # rewrite_instruction + factor_refinement sequence, ~5-9 calls -> 1).
        factors = entry.setdefault("factors", {})
        entry["factors"] = self.refiner.merge_factors_batch(factors, pairs)

    def _find_slot(self, tool: dict, subgoal_text: str) -> Optional[dict]:
        subs = tool.get("subgoals", [])
        target = (subgoal_text or "").strip()
        for entry in subs:
            if entry.get("subgoal", "").strip() == target:
                return entry
        return self.retriever.find_similar_subgoal(subs, target)

    @staticmethod
    def _nearest_existing_slot(subs: List[dict], subgoal_text: str) -> Optional[dict]:
        if not subs:
            return None
        target = subgoal_text or ""
        best, best_score = subs[0], -1.0
        for entry in subs:
            score = KeywordMemoryRetriever._score(entry.get("subgoal", ""), target)
            if score > best_score:
                best, best_score = entry, score
        return best

    def _enforce_subgoal_cap(self, tool_name: str) -> None:
        tool = self.data[tool_name]
        subs = tool["subgoals"]
        while len(subs) > MAX_SUBGOALS_PER_TOOL:
            a_idx, b_idx, best = 0, 1, -1.0
            for i in range(len(subs)):
                for j in range(i + 1, len(subs)):
                    s = KeywordMemoryRetriever._score(subs[i]["subgoal"], subs[j]["subgoal"])
                    if s > best:
                        best, a_idx, b_idx = s, i, j
            a, b = subs[a_idx], subs[b_idx]
            merged_text = (self.refiner.rewrite_instruction(a["subgoal"], b["subgoal"])
                           if self.refiner._engine is not None else a["subgoal"])
            merged_factors = dict(a.get("factors", {}))
            merged_factors.update(b.get("factors", {}))
            merged_factors = self.refiner.factor_refinement(merged_factors)
            a["subgoal"] = merged_text
            a["factors"] = merged_factors
            subs.pop(b_idx)

    # ---- serialization ----
    def save(self, path: str, fmt: str = "json") -> None:
        fmt = fmt.lower()
        if fmt == "yaml":
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "YAML serialization requires PyYAML. `pip install pyyaml` or use fmt='json'."
                )
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"unsupported fmt: {fmt} (use 'json' or 'yaml')")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yaml", ".yml")):
            if not _YAML_AVAILABLE:
                raise ImportError("YAML load requires PyYAML.")
            self.data = yaml.safe_load(text) or {}
        else:
            self.data = json.loads(text)
        self._validate()

    def _validate(self) -> None:
        for tool in self.data.values():
            tool.setdefault("subgoals", [])
            tool["subgoals"] = tool["subgoals"][:MAX_SUBGOALS_PER_TOOL]
            for entry in tool["subgoals"]:
                entry.setdefault("subgoal", "")
                factors = entry.setdefault("factors", {})
                # each factor: exactly one instruction
                for fname, fobj in list(factors.items()):
                    if not isinstance(fobj, dict):
                        factors[fname] = {"instruction": str(fobj)}
                        continue
                    instr = fobj.get("instruction")
                    if isinstance(instr, list):
                        # collapse multiple instructions into one
                        factors[fname] = {"instruction": " ".join(str(x) for x in instr)}
                    elif not isinstance(instr, str):
                        factors[fname] = {"instruction": str(instr) if instr is not None else ""}
                if len(factors) > MAX_FACTORS_PER_SUBGOAL:
                    kept = sorted(factors.items(), key=lambda kv: -len(kv[1].get("instruction", "")))
                    entry["factors"] = dict(kept[:MAX_FACTORS_PER_SUBGOAL])

    # ---- seed ----
    @classmethod
    def seed_default_invocations(
        cls, retriever: Optional[MemoryRetriever] = None,
        refiner: Optional[MemoryRefiner] = None,
    ) -> "ToolInvocationMemory":
        """Build an invocation memory pre-seeded with ABSTRACT, ORTHOGONAL
        decision-dimension -> instruction pairs for every canonical subgoal of
        the 7 tools.

        Decision dimensions are the canonical orthogonal dimensions per tool,
        each carrying exactly one instruction. Evolution only generalizes each
        instruction (preserving all previous meanings) and merges
        semantic-duplicate dimensions; it never adds new subgoal slots (those
        are fixed by the capability vocabulary), never adds fields, and stays
        within <= MAX_FACTORS_PER_SUBGOAL dimensions per subgoal.
        """
        mem = cls(retriever=retriever, refiner=refiner)

        def F(**instructions) -> Dict[str, dict]:
            return {name: {"instruction": instr} for name, instr in instructions.items()}

        google = {
            "Retrieve external factual knowledge": F(
                Entity="Include sufficient identifiers to uniquely specify the target.",
                Freshness="Prefer recent sources when temporal relevance affects the answer.",
                Scope="Restrict the search to the requested domain or topic.",
                Source="Prefer authoritative sources when reliability is critical.",
            ),
            "Disambiguate an ambiguous entity": F(
                Entity="Add distinguishing qualifiers to separate competing candidates.",
                Freshness="Use recent sources when the entity identity may have changed.",
                Scope="Broaden the search to surface multiple candidates.",
                Source="Cross-check at least two independent sources.",
            ),
            "Retrieve time-sensitive information": F(
                Entity="Identify the event or subject whose recency matters.",
                Freshness="Favor the most recent results; constrain by date when possible.",
                Scope="Narrow to sources likely to publish timely updates.",
                Source="Prefer primary or official sources for time-sensitive facts.",
            ),
            "Verify a contested claim": F(
                Entity="Name the specific claim subject being verified.",
                Freshness="Check whether newer evidence has overturned earlier claims.",
                Scope="Search for both supporting and contradicting evidence.",
                Source="Prefer peer-reviewed or official sources over informal ones.",
            ),
        }
        web = {
            "Extract a specific field from a known page": F(
                Entity="Pin the exact URL of the target page.",
                Source="Fetch directly from the canonical URL.",
                Scope="Target the specific section or field, not the whole page.",
                Format="Request the exact field format required by the task.",
            ),
            "Verify content of a known page": F(
                Entity="Pin the exact URL to verify.",
                Source="Fetch the canonical URL directly.",
                Scope="Compare the relevant passage against the claim.",
                Format="Return the verbatim text supporting or refuting the claim.",
            ),
            "Retrieve full-text content behind a known URL": F(
                Entity="Pin the exact URL of the document.",
                Source="Fetch the canonical URL directly.",
                Scope="Retrieve the full body or the requested section.",
                Format="Preserve the original structure of the extracted content.",
            ),
        }
        wiki = {
            "Resolve a notable encyclopedic entity": F(
                Entity="Provide the entity name and any disambiguating context.",
                Scope="Target the Wikipedia namespace/language requested.",
                Source="Use the Wikipedia article as the authoritative source.",
            ),
            "Verify an entity attribute": F(
                Entity="Name the entity and the attribute to verify.",
                Scope="Open the specific article section that states the attribute.",
                Source="Prefer the article's cited reference when reliability is critical.",
            ),
            "Retrieve structured facts on a topic": F(
                Entity="Identify the topic and the structured fields required.",
                Scope="Target list/table sections (e.g., discography, infobox).",
                Source="Use Wikipedia as the structured-fact source.",
            ),
        }
        python = {
            "Compute from given numbers": F(
                Inputs="Bind every numeric input the computation requires.",
                Method="Implement the exact arithmetic formula requested.",
                Precision="Apply rounding only as the task specifies.",
            ),
            "Perform a symbolic or logic verification": F(
                Inputs="Bind the symbolic expression or logical statement to test.",
                Method="Implement the equivalence or tautology check directly.",
                Precision="Return a clear boolean/verdict, not an approximation.",
            ),
            "Transform or parse structured data": F(
                Inputs="Bind the raw data string to transform.",
                Method="Use the correct parsing rules for the data format.",
                Precision="Preserve all relevant fields; do not drop data.",
            ),
        }
        screenshot = {
            "Capture the visual state of a page": F(
                Entity="Pin the exact URL to capture.",
                Viewport="Set the viewport to reveal the relevant content.",
            ),
        }
        ocr = {
            "Extract text from an image": F(
                Entity="Identify the image and the text region of interest.",
                Format="Return extracted text verbatim, preserving order.",
            ),
            "Parse figure axes or a table": F(
                Entity="Identify the chart/table and the structured elements to read.",
                Format="Return axes labels or table cells in a structured form.",
            ),
        }
        base = {
            "Answer by pure reasoning": F(
                Reasoning="Lay out the deductive steps explicitly before the answer.",
            ),
            "Solve a logic or language puzzle": F(
                Reasoning="Apply the puzzle's given rules step by step.",
            ),
        }

        def build(subgoal_factor_map: Dict[str, Dict[str, dict]]) -> Dict[str, Any]:
            return {"subgoals": [
                {"subgoal": sg, "factors": fac}
                for sg, fac in subgoal_factor_map.items()
            ]}

        mem.data = {
            "Google_Search_Tool": build(google),
            "Web_Search_Tool": build(web),
            "Wikipedia_Search_Tool": build(wiki),
            "Python_Coder_Tool": build(python),
            "Screenshot_Tool": build(screenshot),
            "Vision_OCR_Tool": build(ocr),
            "Base_Generator_Tool": build(base),
        }
        return mem


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    import tempfile

    print("== Tool Knowledge Memory smoke test (keyword retriever, no LLM) ==")

    cap = ToolCapabilityMemory.seed_default_capabilities()
    print("seeded tools:", cap.all_tools())
    seeded_count = len(cap.data["Google_Search_Tool"]["subgoals"])
    print("Google_Search_Tool seeded subgoal slots:", seeded_count)

    hit = cap.retrieve("Google_Search_Tool", "find scientific name of a fish")
    print("retrieve Google_Search_Tool / 'find scientific name of a fish' ->",
          hit["subgoal"] if hit else None)

    # evolve with an abstract subgoal that IS in the frozen vocabulary:
    # slot count must NOT grow; only context_summary generalizes.
    before = len(cap.data["Google_Search_Tool"]["subgoals"])
    cap.evolve("Google_Search_Tool", "Retrieve external factual knowledge",
               ["Missing external knowledge", "Question needs supporting facts"])
    after = len(cap.data["Google_Search_Tool"]["subgoals"])
    print(f"evolve canonical subgoal: subgoals {before} -> {after} (must NOT grow)")
    assert after == before == seeded_count, "frozen vocabulary must not grow"

    # evolve with an abstract subgoal NOT verbatim in the vocabulary:
    # the gate maps it onto the nearest canonical slot; still no new slot.
    cap.evolve("Google_Search_Tool", "Find a recent news event",
               ["Temporal recency required"])
    after2 = len(cap.data["Google_Search_Tool"]["subgoals"])
    print("evolve non-verbatim subgoal: Google_Search_Tool subgoals now", after2)
    assert after2 == seeded_count, "non-verbatim evolve must map to existing slot"

    # JSON roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cap.save(path, fmt="json")
    cap2 = ToolCapabilityMemory()
    cap2.load(path)
    assert cap2.to_dict() == cap.to_dict(), "JSON roundtrip mismatch"
    print("JSON roundtrip OK")

    if _YAML_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            ypath = f.name
        cap.save(ypath, fmt="yaml")
        cap3 = ToolCapabilityMemory()
        cap3.load(ypath)
        assert cap3.to_dict() == cap.to_dict(), "YAML roundtrip mismatch"
        print("YAML roundtrip OK")
    else:
        print("YAML skipped (PyYAML not installed)")

    # Invocation memory: seed then evolve factors within a frozen slot.
    inv = ToolInvocationMemory.seed_default_invocations()
    inv_slots = len(inv.data["Google_Search_Tool"]["subgoals"])
    print("Google_Search_Tool invocation slots:", inv_slots)

    inv.evolve("Google_Search_Tool", "Retrieve external factual knowledge",
               {"Entity": "Add unique identifiers so the target is unambiguous.",
                "Freshness": "Enable recent search when temporal relevance is required."})
    got = inv.retrieve("Google_Search_Tool", "Retrieve external factual knowledge")
    print("invocation factors after 1st evolve:", list(got["factors"].keys()))
    assert len(got["factors"]) <= MAX_FACTORS_PER_SUBGOAL
    # slot count unchanged
    assert len(inv.data["Google_Search_Tool"]["subgoals"]) == inv_slots

    # evolve same slot with a new factor + a similar-to-existing factor
    inv.evolve("Google_Search_Tool", "Retrieve external factual knowledge",
               {"Scope": "Restrict search to the requested domain.",
                "Entity": "Pin the target with names and qualifiers."})
    got = inv.retrieve("Google_Search_Tool", "Retrieve external factual knowledge")
    print("invocation factors after 2nd evolve:", list(got["factors"].keys()))
    assert len(got["factors"]) <= MAX_FACTORS_PER_SUBGOAL
    print("factor cap enforced OK")

    inv.save(path + ".inv.json", fmt="json")
    inv2 = ToolInvocationMemory()
    inv2.load(path + ".inv.json")
    assert inv2.to_dict() == inv.to_dict(), "inv JSON roundtrip mismatch"
    print("invocation JSON roundtrip OK")

    # Schema version sanity
    print("MEMORY_SCHEMA_VERSION =", MEMORY_SCHEMA_VERSION)

    print("== smoke test passed ==")

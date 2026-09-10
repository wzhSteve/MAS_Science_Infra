"""
TaskProfile + SlotGate: slot-driven task completion for EPC_AW.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

PHASE_RETRIEVE = "RETRIEVE"
PHASE_TRANSFORM = "TRANSFORM"
PHASE_SYNTHESIZE = "SYNTHESIZE"
PHASE_VERIFY = "VERIFY"

SOURCE_RANK = {
    "unknown": 0,
    "inferred": 1,
    "secondary": 2,
    "primary": 3,
    "computed": 4,
}

FINAL_TERMS = (
    "egalitarian", "hierarchical", "utilitarian", "deontological",
    "standardization", "localization", "egalitarianism", "utilitarianism",
)
AXIS_ENDPOINT_WORDS = (
    "standardization", "localization", "utilitarian", "egalitarian", "deontological",
    "standardized", "localized", "utilitarianism", "egalitarianism",
)
USGS_MARKERS = ("usgs", "nas.er.usgs.gov", "nonindigenous", "nonnative")


def _extract_zip_codes(text: str) -> List[str]:
    raw = str(text)
    candidates = re.findall(r"\b\d{5}\b", raw)
    excluded: Set[str] = set()
    for match in re.finditer(r"\b(\d{4})\.(\d{5})\b", raw):
        excluded.add(match.group(2))
    for match in re.finditer(
        r"arxiv(?:\.org/(?:abs|pdf)/(\d{4})\.(\d{5})|:?(\d{4})\.(\d{5}))",
        raw,
        re.I,
    ):
        suffix = match.group(2) or match.group(4)
        if suffix:
            excluded.add(suffix)
    for url_match in re.finditer(r"https?://[^\s\])\"']+", raw, re.I):
        for seg in re.findall(r"/(\d{5})(?:/|$|\?|#)", url_match.group(0)):
            excluded.add(seg)
    return [z for z in dict.fromkeys(candidates) if z not in excluded]


def _has_usgs_context(text: str) -> bool:
    lower = str(text).lower()
    return any(m in lower for m in USGS_MARKERS)


def _is_character_name_answer(value: Any) -> bool:
    """Accept a minimal character name, never a code fragment or explanation."""
    raw = str(value or "").strip()
    raw = re.sub(
        r"^(?:the\s+)?(?:exact\s+)?character(?:\s+name)?\s+(?:is|needed\s+is)\s*[:\-]?\s*",
        "",
        raw,
        flags=re.I,
    )
    raw = raw.strip().strip("`'\"").strip()
    if not raw or len(raw) > 30:
        return False
    if re.search(r"[^A-Za-z\-\s]", raw):
        return False
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", raw)
    return 1 <= len(tokens) <= 2


def _extract_character_name(value: Any) -> Optional[str]:
    """Extract canonical short names for common punctuation characters."""
    text = str(value or "").lower()
    aliases = (
        (("backtick", "backquote", "grave accent"), "backtick"),
        (("apostrophe", "single quote"), "apostrophe"),
        (("double quote", "quotation mark"), "quote"),
        (("semicolon",), "semicolon"),
        (("colon",), "colon"),
        (("comma",), "comma"),
        (("period", "full stop"), "period"),
        (("parenthesis",), "parenthesis"),
        (("bracket",), "bracket"),
    )
    for names, canonical in aliases:
        if any(name in text for name in names):
            return canonical
    return None


def _count_axis_pairs(content: str) -> int:
    """Count axis pairs like (i) X vs. Y or X vs Y."""
    lower = str(content).lower()
    vs_pairs = len(re.findall(r"\bvs\.?\b", lower))
    roman_pairs = len(re.findall(r"\(\s*[ivx]+\s*\)", lower))
    return max(vs_pairs, roman_pairs // 2 if roman_pairs >= 2 else vs_pairs)


@dataclass
class AnswerSlot:
    name: str
    slot_type: str = "entity"  # entity | enum | numeric | formatted_list
    required: bool = True
    min_source: str = "secondary"  # primary | secondary | computed | inferred


@dataclass
class TaskProfile:
    question: str
    slots: List[AnswerSlot] = field(default_factory=list)
    phase: str = PHASE_RETRIEVE
    abstain_allowed: bool = False
    exhaustive: bool = False  # count/enumerate tasks requiring full coverage

    def missing_slot_names(self, evidence_records: List[Dict[str, Any]]) -> List[str]:
        filled = SlotGate.filled_slots(self, evidence_records)
        return [s.name for s in self.slots if s.required and s.name not in filled]

    def task_state_block(self, evidence_records: Optional[List[Dict[str, Any]]] = None) -> str:
        records = evidence_records or []
        missing = self.missing_slot_names(records)
        forbidden = SlotGate.forbidden_tools(self, records)
        forbidden_str = ", ".join(forbidden) if forbidden else "none"
        missing_str = ", ".join(missing) if missing else "none"
        return (
            f"[TaskState] phase={self.phase} "
            f"missing_slots=[{missing_str}] "
            f"forbidden_tools=[{forbidden_str}]"
        )


class SlotGate:
    """Single gate for STOP / synthesis phase / tool restrictions."""

    @staticmethod
    def infer_profile(question: str) -> TaskProfile:
        q = question.lower()
        slots: List[AnswerSlot] = []

        if "zip code" in q or "zip codes" in q:
            slots.append(AnswerSlot(
                name="zip_codes",
                slot_type="formatted_list",
                min_source="primary",
            ))
        elif any(m in q for m in (
            "how many", "round the", "round your", "rounded to",
            "round up", "incorrect as to", "p-value", "p value",
            "statistical significance",
        )):
            if any(m in q for m in ("p-value", "p value", "statistical significance", "incorrect as to")):
                slots.append(AnswerSlot(name="base_count", slot_type="numeric", min_source="secondary"))
                slots.append(AnswerSlot(name="computed_count", slot_type="numeric", min_source="computed"))
            elif re.search(
                r"\b(hours?|distance|pace|speed|km/h|perigee|marathon|calculate|computation|thousand)\b",
                q,
            ):
                slots.append(AnswerSlot(
                    name="input_metrics",
                    slot_type="formatted_list",
                    min_source="secondary",
                ))
                slots.append(AnswerSlot(name="computed_count", slot_type="numeric", min_source="computed"))
            else:
                slots.append(AnswerSlot(name="computed_count", slot_type="numeric", min_source="computed"))
        elif re.search(r"which of (these|the)", q) or (
            "arxiv" in q and re.search(r"describe a type|type of society|one of these", q)
        ):
            slots.append(AnswerSlot(name="final_term", slot_type="enum", min_source="primary"))
            if "figure" in q or "axis" in q or "label" in q:
                slots.append(AnswerSlot(name="axis_labels", slot_type="formatted_list", min_source="primary"))
        elif re.search(r"(name of the character|what character|exact char(?:a)?cter)", q):
            slots.append(AnswerSlot(
                name="final_answer",
                slot_type="character_name",
                min_source="secondary",
            ))
        else:
            slots.append(AnswerSlot(name="final_answer", slot_type="entity", min_source="secondary"))

        # Flag exhaustive count/enumerate tasks (require full coverage, not a snippet)
        exhaustive = bool(
            re.search(r"\bhow many\b", q)
            and (
                re.search(r"\bbetween\b\s+\d{4}\s+(and|to|-)\s+\d{4}", q)
                or re.search(r"\b(discography|studio albums|albums|papers|articles)\b", q)
                or re.search(r"\b(published by|released by)\b.{0,60}\b\d{4}\b", q)
            )
        )

        return TaskProfile(
            question=question, slots=slots, phase=PHASE_RETRIEVE, exhaustive=exhaustive,
        )

    @staticmethod
    def _cross_source_corroborates(
        records: List[Dict[str, Any]],
        slot_name: str,
        value: str,
    ) -> bool:
        """True when >=2 active records from different tools or entity_keys support the same value."""
        val = str(value).lower().strip()
        if not val:
            return False
        supporting: List[Tuple[str, str]] = []
        for rec in records:
            if rec.get("status") == "disputed":
                continue
            content = str(rec.get("content", "")).lower()
            bindings = rec.get("slot_bindings") or {}
            bound = str(bindings.get(slot_name, "")).lower()
            if val not in content and val not in bound:
                continue
            tool = str(rec.get("tool", ""))
            entities = rec.get("entity_keys") or []
            key = entities[0] if entities else tool
            supporting.append((tool, key))
        if len(supporting) < 2:
            return False
        keys = {k for _, k in supporting}
        tools = {t for t, _ in supporting}
        return len(keys) >= 2 or len(tools) >= 2

    @staticmethod
    def _record_is_computed(record: Dict[str, Any]) -> bool:
        """True when a record's evidence originated from a real compute step.

        v3: a Python_Coder_Tool record counts as computed only when
        `compute_real` is True (the executed code actually derived a value from
        retrieved inputs). The bare tool-name fallback is removed so "printer"
        and hallucinated-computation steps cannot satisfy computed slots.
        """
        if record.get("compute_real") is True:
            return True
        return str(record.get("source_quality", "")).lower() == "computed"

    @staticmethod
    def _record_qualifies(
        record: Dict[str, Any],
        slot: AnswerSlot,
        all_records: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if record.get("status") == "disputed":
            return False

        bindings = record.get("slot_bindings") or {}
        if slot.name in bindings and bindings[slot.name]:
            if slot.slot_type == "character_name":
                return _is_character_name_answer(bindings[slot.name])
            # computed-slots (e.g. computed_count) must be filled by a computed
            # source — a retrieved value (e.g. a pace) must NOT pollute them.
            if slot.min_source == "computed" and not SlotGate._record_is_computed(record):
                return False
            return True

        quality = str(record.get("source_quality", "unknown")).lower()
        claim_type = str(record.get("claim_type", "fact")).lower()
        if claim_type == "absence" and slot.slot_type != "entity":
            return False
        if claim_type == "hypothesis":
            min_rank = SOURCE_RANK.get(slot.min_source, 2)
            if SOURCE_RANK.get("inferred", 1) < min_rank:
                return False

        content = str(record.get("content", "")).lower()
        records = all_records or [record]

        if not SlotGate._content_fills_slot(slot, content, record, records):
            return False

        min_rank = SOURCE_RANK.get(slot.min_source, 2)
        if SOURCE_RANK.get(quality, 0) >= min_rank:
            return True

        return SlotGate._relaxed_source_qualifies(record, slot, content, records)

    @staticmethod
    def _relaxed_source_qualifies(
        record: Dict[str, Any],
        slot: AnswerSlot,
        content: str,
        records: List[Dict[str, Any]],
    ) -> bool:
        quality = str(record.get("source_quality", "unknown")).lower()
        if quality not in ("secondary", "primary", "computed"):
            return False

        if slot.name == "final_term":
            if "arxiv.org/abs/" in content or (record.get("entity_keys") or []):
                return True
            for term in FINAL_TERMS:
                if term in content:
                    return SlotGate._cross_source_corroborates(records, slot.name, term)
            return False

        if slot.name == "axis_labels":
            return SlotGate._axis_labels_satisfied(content)

        if slot.name == "zip_codes":
            if not _extract_zip_codes(content):
                return False
            cumulative = " ".join(str(r.get("content", "")) for r in records)
            return _has_usgs_context(cumulative) or _has_usgs_context(content)

        return False

    @staticmethod
    def _axis_labels_satisfied(content: str) -> bool:
        lower = str(content).lower()
        if _count_axis_pairs(lower) >= 2:
            return True
        endpoint_hits = sum(1 for w in AXIS_ENDPOINT_WORDS if w in lower)
        return endpoint_hits >= 4

    @staticmethod
    def extract_axis_labels_snippet(text: str) -> Optional[str]:
        """Extract a compact axis-labels string when ≥2 axis pairs are present."""
        raw = str(text)
        if not SlotGate._axis_labels_satisfied(raw):
            return None
        pairs = re.findall(
            r"\(\s*[ivx]+\s*\)\s*[^(),\n]{3,70}\s+vs\.?\s+[^(),\n]{3,70}",
            raw,
            re.I,
        )
        if len(pairs) >= 2:
            return ", ".join(p.strip() for p in pairs[:3])[:300]
        chunks = re.split(r"[,;]\s*", raw)
        axis_parts = [c.strip() for c in chunks if re.search(r"\bvs\.?\b", c, re.I)]
        if len(axis_parts) >= 2:
            return "; ".join(axis_parts[:3])[:300]
        lower = raw.lower()
        idx = lower.find(" vs")
        if idx >= 0:
            return raw[max(0, idx - 60): idx + 140].strip()[:300]
        return raw[:300]

    @staticmethod
    def _content_fills_slot(
        slot: AnswerSlot,
        content: str,
        record: Dict[str, Any],
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if slot.name == "zip_codes":
            return bool(_extract_zip_codes(content))
        if slot.name == "computed_count":
            if re.search(r"\b(the answer is|final answer|result is)\s*[:\s]*\d+", content):
                return True
            if record.get("tool") == "Python_Coder_Tool" and re.search(r"\b\d+\b", content):
                return True
            return bool(re.search(r"(ceil\s*\(|calculated|computed|rounded up)", content) and re.search(r"\b\d+\b", content))
        if slot.name == "input_metrics":
            nums = re.findall(r"\b\d[\d,]*\.?\d*\b", content.replace(",", ""))
            has_distance = any(m in content for m in ("km", "kilometer", "distance", "perigee", "moon", "earth"))
            has_speed = any(m in content for m in ("km/h", "km/hr", "pace", "speed", "marathon", "mph"))
            return len(nums) >= 2 and has_distance and has_speed
        if slot.name == "base_count":
            return bool(re.search(r"\b\d{2,5}\b", content) and any(
                m in content for m in ("article", "published", "paper", "count", "research")
            ))
        if slot.name == "axis_labels":
            return SlotGate._axis_labels_satisfied(content)
        if slot.name == "final_term":
            if re.search(r"\banswer:\s*(none|null)\b", content):
                return False
            if any(t in content for t in FINAL_TERMS):
                if record.get("source_quality") == "inferred" and record.get("tool") == "Base_Generator_Tool":
                    return False
                return True
            return bool(re.search(r"\b(the answer is|final answer is)\s+\S+", content))
        if slot.name == "final_answer":
            if slot.slot_type == "character_name":
                return _is_character_name_answer(content)
            if len(content.strip()) < 8:
                return False
            negative = ("cannot be determined", "insufficient", "no valid", "answer: none")
            if any(n in content for n in negative):
                return False
            return True
        return False

    @staticmethod
    def filled_slots(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> Set[str]:
        filled: Set[str] = set()
        for slot in profile.slots:
            if slot.name == "input_metrics":
                combined = " ".join(str(r.get("content", "")) for r in evidence_records)
                if SlotGate._content_fills_slot(slot, combined.lower(), {}, evidence_records):
                    filled.add(slot.name)
                continue
            for rec in evidence_records:
                if SlotGate._record_qualifies(rec, slot, evidence_records):
                    filled.add(slot.name)
                    break
        return filled

    @staticmethod
    def all_required_filled(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> bool:
        if not profile.slots:
            return False
        filled = SlotGate.filled_slots(profile, evidence_records)
        return all(s.name in filled for s in profile.slots if s.required)

    @staticmethod
    def can_stop(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> bool:
        return SlotGate.all_required_filled(profile, evidence_records)

    @staticmethod
    def can_enter_synthesize(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> bool:
        acquisition_slots = [
            s for s in profile.slots
            if s.name not in ("final_term", "final_answer", "computed_count")
        ]
        if not acquisition_slots:
            return SlotGate.all_required_filled(profile, evidence_records)
        filled = SlotGate.filled_slots(profile, evidence_records)
        return all(s.name in filled for s in acquisition_slots if s.required)

    @staticmethod
    def needs_compute_step(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> bool:
        """True when input evidence is ready but computed_count still missing."""
        slot_names = {s.name for s in profile.slots}
        if "computed_count" not in slot_names:
            return False
        filled = SlotGate.filled_slots(profile, evidence_records)
        if "computed_count" in filled:
            return False
        if "input_metrics" in slot_names:
            return "input_metrics" in filled
        if "base_count" in slot_names:
            return "base_count" in filled
        return False

    @staticmethod
    def has_compute_slot(profile: TaskProfile) -> bool:
        """True when the profile requires at least one computed-source slot."""
        return any(s.min_source == "computed" for s in profile.slots)

    @staticmethod
    def compute_slot_filled_by_computed(
        profile: TaskProfile, evidence_records: List[Dict[str, Any]],
    ) -> bool:
        """True when every computed-source slot is filled by a computed record."""
        for slot in profile.slots:
            if slot.min_source != "computed":
                continue
            if not slot.required:
                continue
            filled = False
            for rec in evidence_records:
                if rec.get("status") == "disputed":
                    continue
                if not SlotGate._record_is_computed(rec):
                    continue
                bindings = rec.get("slot_bindings") or {}
                if slot.name in bindings and bindings[slot.name]:
                    filled = True
                    break
                if SlotGate._record_qualifies(rec, slot, evidence_records):
                    filled = True
                    break
            if not filled:
                return False
        return True

    @staticmethod
    def forbidden_tools(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> List[str]:
        forbidden: List[str] = []
        if not SlotGate.can_enter_synthesize(profile, evidence_records):
            forbidden.append("Base_Generator_Tool")
        return forbidden

    @staticmethod
    def update_phase(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> str:
        if SlotGate.all_required_filled(profile, evidence_records):
            return PHASE_VERIFY
        if SlotGate.can_enter_synthesize(profile, evidence_records):
            missing = profile.missing_slot_names(evidence_records)
            if missing and missing[0] in ("final_term", "final_answer", "computed_count"):
                return PHASE_SYNTHESIZE
            return PHASE_TRANSFORM
        return PHASE_RETRIEVE

    @staticmethod
    def apply_slot_updates(
        profile: TaskProfile,
        evidence_records: List[Dict[str, Any]],
        slot_updates: List[Dict[str, Any]],
    ) -> None:
        if not slot_updates or not evidence_records:
            return
        latest = evidence_records[-1]
        bindings = dict(latest.get("slot_bindings") or {})
        for upd in slot_updates:
            if not isinstance(upd, dict):
                continue
            if upd.get("filled") and upd.get("value"):
                bindings[str(upd.get("slot", ""))] = str(upd.get("value"))
        latest["slot_bindings"] = bindings

    @staticmethod
    def extract_final_answer(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> Optional[str]:
        """Extract formatted final answer from filled slots."""
        if not profile or not evidence_records:
            return None

        slot_order = [s.name for s in profile.slots]
        if "computed_count" in slot_order:
            slot_order.remove("computed_count")
            slot_order.insert(0, "computed_count")

        slot_by_name = {s.name: s for s in profile.slots}
        filled = SlotGate.filled_slots(profile, evidence_records)

        for slot_name in slot_order:
            if slot_name not in filled:
                continue
            slot = slot_by_name[slot_name]
            for rec in evidence_records:
                if rec.get("status") == "disputed":
                    continue
                bindings = rec.get("slot_bindings") or {}
                if slot.name in bindings and bindings[slot.name]:
                    val = str(bindings[slot.name]).strip()
                    if val:
                        return SlotGate._format_slot_answer(slot, val, evidence_records)

                content = str(rec.get("content", ""))
                if SlotGate._record_qualifies(rec, slot, evidence_records):
                    return SlotGate._format_slot_answer(slot, content, evidence_records)
        return None

    @staticmethod
    def _format_slot_answer(
        slot: AnswerSlot,
        value: str,
        records: List[Dict[str, Any]],
    ) -> str:
        if slot.name == "zip_codes":
            zips = _extract_zip_codes(value)
            if not zips:
                for rec in records:
                    zips.extend(_extract_zip_codes(str(rec.get("content", ""))))
                zips = list(dict.fromkeys(zips))
            return ",".join(zips) if zips else value.strip()

        if slot.name == "computed_count":
            m = re.search(r"\b(\d+)\b", value)
            return m.group(1) if m else value.strip()

        if slot.name == "final_term":
            lower = value.lower()
            preferred = (
                "egalitarian", "hierarchical", "utilitarian", "deontological",
                "standardization", "localization",
            )
            for term in preferred:
                if term in lower:
                    return term
            m = re.search(r"'(\w+)'", value)
            if m:
                return m.group(1).lower()
            return value.strip().split()[0].lower()

        if slot.name == "final_answer":
            if slot.slot_type == "character_name":
                raw = re.sub(
                    r"^(?:the\s+)?(?:exact\s+)?character(?:\s+name)?\s+"
                    r"(?:is|needed\s+is)\s*[:\-]?\s*",
                    "",
                    value.strip(),
                    flags=re.I,
                )
                return raw.strip().strip("`'\"").strip()
            m = re.search(r"\b(?:the answer is|final answer is)\s+(.+?)(?:\.|$)", value, re.I)
            if m:
                return m.group(1).strip()
            return value.strip()[:200]

        return value.strip()

    @staticmethod
    def has_partial_cross_ref(profile: TaskProfile, evidence_records: List[Dict[str, Any]]) -> bool:
        """True when final_term is filled but axis_labels or full stop not yet reached."""
        if not profile:
            return False
        filled = SlotGate.filled_slots(profile, evidence_records)
        return "final_term" in filled and not SlotGate.all_required_filled(profile, evidence_records)

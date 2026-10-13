"""Extractive structured facts for search tools (no TaskProfile slots).

Output contract appended to tool strings:

    <human text>

    ---STRUCTURED---
    {"facts":[...], "refusal": false}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

STRUCTURED_MARKER = "---STRUCTURED---"

_CHARACTER_ALIASES = (
    (("backtick", "backquote", "grave accent"), "backtick"),
    (("apostrophe", "single quote"), "apostrophe"),
    (("double quote", "quotation mark"), "quote"),
    (("semicolon",), "semicolon"),
    (("colon",), "colon"),
    (("comma",), "comma"),
    (("period", "full stop"), "period"),
    (("parenthesis", "parentheses"), "parenthesis"),
    (("bracket",), "bracket"),
)


def infer_query_intent(query: str) -> str:
    """Return numeric | character_name | entity from query text."""
    q = str(query or "").lower()
    if re.search(
        r"(name of the char(?:a)?cter|what char(?:a)?cter|exact char(?:a)?cter)",
        q,
    ):
        return "character_name"
    if re.search(
        r"\b(how many|how much|p[- ]?value|km\b|kilometer|thousand hours|"
        r"round (?:the|your)|perigee|marathon pace|statistical significance|"
        r"\d+(?:\.\d+)?\s*%|count of|number of)\b",
        q,
    ):
        return "numeric"
    return "entity"


def _context_window(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def extract_numeric_candidates(
    text: str,
    url: str = "",
    *,
    max_candidates: int = 12,
) -> List[Dict[str, Any]]:
    """Pull number+unit candidates with surrounding quotes from raw text."""
    blob = str(text or "")
    if not blob.strip():
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def _emit(raw_val: str, unit: Optional[str], start: int, end: int, confidence: str) -> None:
        nonlocal out
        if len(out) >= max_candidates:
            return
        quote = _context_window(blob, start, end)
        key = (raw_val.replace(",", ""), unit or "", quote[:80])
        if key in seen:
            return
        seen.add(key)
        try:
            if ":" in raw_val:
                value: Any = raw_val
            else:
                value = float(raw_val.replace(",", ""))
                if value == int(value):
                    value = int(value)
        except ValueError:
            value = raw_val
        if unit is None and isinstance(value, (int, float)) and value < 10:
            return
        out.append({
            "type": "number",
            "value": value,
            "unit": unit,
            "quote": quote,
            "url": url,
            "confidence": confidence,
        })

    # Range endpoints: "356,355 to 370,399 km" → both bounds (unit on the pair).
    for m in re.finditer(
        r"(\d{1,3}(?:,\d{3})+|\d{5,7})\s*(?:to|-|–|—)\s*"
        r"(\d{1,3}(?:,\d{3})+|\d{5,7})\s*(km|kilometers?)\b",
        blob,
        re.I,
    ):
        unit = m.group(3).lower()
        unit = unit.rstrip("s") if unit.endswith("s") else unit
        _emit(m.group(1), unit, m.start(), m.end(), "high")
        _emit(m.group(2), unit, m.start(), m.end(), "high")
        if len(out) >= max_candidates:
            return out

    patterns = (
        # 363,300 km / 363300 kilometers
        (
            r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
            r"(km|kilometers?|miles?|hours?|articles?|papers?|%|percent)\b",
            True,
        ),
        # bare mid-size integers near count words (handled via quote later)
        (r"\b(\d{2,7})(?:\.\d+)?\b", False),
        # H:MM:SS marathon times as string values
        (r"\b(\d:\d{2}:\d{2})\b", False),
    )
    for pattern, has_unit in patterns:
        for m in re.finditer(pattern, blob, re.I):
            raw_val = m.group(1)
            unit = m.group(2).lower() if has_unit and m.lastindex and m.lastindex >= 2 else None
            if unit:
                unit = unit.rstrip("s") if unit.endswith("s") and unit not in ("hours",) else unit
            _emit(raw_val, unit, m.start(), m.end(), "high" if unit else "medium")
            if len(out) >= max_candidates:
                return out
    return out


def extract_character_candidates(
    text: str,
    url: str = "",
) -> List[Dict[str, Any]]:
    """Find canonical punctuation character names grounded in text."""
    blob = str(text or "")
    lower = blob.lower()
    out: List[Dict[str, Any]] = []
    seen = set()
    for names, canonical in _CHARACTER_ALIASES:
        for name in names:
            for m in re.finditer(
                rf"(?<![\w-]){re.escape(name)}(?![\w-])",
                lower,
            ):
                if canonical in seen:
                    break
                quote = _context_window(blob, m.start(), m.end())
                seen.add(canonical)
                out.append({
                    "type": "character_name",
                    "value": canonical,
                    "unit": None,
                    "quote": quote,
                    "url": url,
                    "confidence": "high",
                })
                break
    return out


def ground_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only facts whose value appears inside their quote (quote-grounded)."""
    kept: List[Dict[str, Any]] = []
    for fact in facts or []:
        quote = str(fact.get("quote") or "")
        qlow = quote.lower()
        ftype = str(fact.get("type") or "")
        value = fact.get("value")
        if ftype == "character_name":
            val = str(value or "").lower()
            # Only canonical punctuation names — refuse bare letters like "g".
            aliases = set()
            for names, canonical in _CHARACTER_ALIASES:
                if canonical == val or val in names:
                    aliases.update(names)
                    aliases.add(canonical)
            if not aliases:
                continue
            if not any(re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", qlow) for a in aliases):
                continue
            kept.append(fact)
            continue
        if ftype == "number":
            if value is None:
                continue
            if isinstance(value, str) and ":" in value:
                if value not in quote:
                    continue
            else:
                # Accept 363300 vs 363,300
                raw = str(value)
                variants = {raw, raw.replace(",", "")}
                if isinstance(value, float) and value == int(value):
                    variants.add(str(int(value)))
                if isinstance(value, (int, float)):
                    # also comma-formatted
                    try:
                        variants.add(f"{int(value):,}")
                    except Exception:
                        pass
                quote_compact = quote.replace(" ", "")
                if not any(v and (v in quote or v in quote_compact) for v in variants):
                    # looser: digit sequence in quote
                    digits = re.sub(r"[^\d.]", "", raw)
                    quote_digits = re.sub(r"[^\d.]", "", quote)
                    if not digits or digits not in quote_digits:
                        continue
            kept.append(fact)
            continue
        kept.append(fact)
    return kept


def format_tool_output(
    human: str,
    facts: Optional[List[Dict[str, Any]]] = None,
    *,
    refusal: bool = False,
) -> str:
    """Append STRUCTURED block; grounds facts before emit."""
    grounded = ground_facts(list(facts or []))
    if refusal or not grounded:
        payload = {"facts": [], "refusal": True}
        body = (human or "").strip() or (
            "The search results do not provide information that answers this query."
        )
        # Keep an explicit human-visible marker for diagnoser paywalled/refusal.
        if "[refusal:true]" not in body.lower():
            body = f"{body}\n[REFUSAL:true]"
    else:
        payload = {"facts": grounded, "refusal": False}
        body = (human or "").strip()
        if not body:
            # Build a short extractive human line from quotes
            parts = []
            for f in grounded[:3]:
                parts.append(str(f.get("quote") or f.get("value")))
            body = " ".join(parts)
    return f"{body}\n\n{STRUCTURED_MARKER}\n{json.dumps(payload, ensure_ascii=False)}"


def parse_structured_block(text: str) -> Optional[Dict[str, Any]]:
    """Parse ---STRUCTURED--- JSON from a tool result string."""
    raw = str(text or "")
    if STRUCTURED_MARKER not in raw:
        return None
    _, _, rest = raw.partition(STRUCTURED_MARKER)
    rest = rest.strip()
    if not rest:
        return None
    # Take first JSON object
    try:
        return json.loads(rest)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", rest, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def structured_numbers(text: str) -> List[Any]:
    """Convenience: list numeric values from STRUCTURED facts."""
    payload = parse_structured_block(text)
    if not payload or payload.get("refusal"):
        return []
    vals = []
    for f in payload.get("facts") or []:
        if f.get("type") == "number" and f.get("value") is not None:
            vals.append(f["value"])
    return vals


def structured_character(text: str) -> Optional[str]:
    payload = parse_structured_block(text)
    if not payload or payload.get("refusal"):
        return None
    for f in payload.get("facts") or []:
        if f.get("type") == "character_name" and f.get("value"):
            return str(f["value"])
    return None


def human_section(text: str) -> str:
    """Strip STRUCTURED block for display / soft prompts."""
    raw = str(text or "")
    if STRUCTURED_MARKER not in raw:
        return raw
    return raw.split(STRUCTURED_MARKER, 1)[0].strip()


def select_best_numeric_fact(
    facts: List[Dict[str, Any]],
    query: str,
) -> Optional[Dict[str, Any]]:
    """Pick the most query-aligned grounded numeric fact (no LLM)."""
    grounded = ground_facts(facts)
    if not grounded:
        return None
    q = str(query or "").lower()
    prefer_units = []
    if any(k in q for k in ("km", "kilometer", "perigee", "distance", "moon")):
        prefer_units.extend(["km", "kilometer"])
    if any(k in q for k in ("hour", "thousand")):
        prefer_units.append("hour")
    if any(k in q for k in ("article", "paper", "published", "how many")):
        prefer_units.extend(["article", "paper"])
    # Polarity for numeric tie-break (no entity hardcoding).
    prefer_min = bool(re.search(
        r"\b(minimum|minimal|smallest|closest|least|perigee)\b", q,
    ))
    prefer_max = bool(re.search(
        r"\b(maximum|maximal|largest|farthest|greatest|apogee)\b", q,
    )) and not prefer_min

    def score(f: Dict[str, Any]) -> Tuple[int, int]:
        unit = (f.get("unit") or "") or ""
        quote = str(f.get("quote") or "").lower()
        s = 0
        if unit and any(u in unit for u in prefer_units):
            s += 5
        for kw in ("perigee", "minimum", "article", "published", "marathon", "pace"):
            if kw in q and kw in quote:
                s += 2
        if f.get("confidence") == "high":
            s += 1
        val = f.get("value")
        size = 0
        if isinstance(val, (int, float)):
            size = int(val)
        # Tie-break by polarity: min→smaller, max→larger, else neutral (0).
        if prefer_min:
            size_key = -size
        elif prefer_max:
            size_key = size
        else:
            size_key = 0
        return (s, size_key)

    grounded.sort(key=score, reverse=True)
    return grounded[0]

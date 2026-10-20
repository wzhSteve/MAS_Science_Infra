"""Short prompts for the SFT copy, plus ``<search>`` / ``<python>`` → ``tool.execute``.

Official f-strings stay in the vendored package. These methods are bound onto
the episode's planner / executor / verifier only when ``AGENTFLOW_SFT_PROMPTS``
is on. Tool names are the ones ``extract_context_subgoal_and_tool`` already
matches.
"""

from __future__ import annotations

import json
import re
from types import MethodType
from pathlib import Path
from typing import Any, Dict, List, Optional

# Exact ``tool_name`` strings. Three of the five classes publish a longer name.
# Keep each line free of a second colon so ``_tool_names`` can split once.
TOOL_LINES: List[str] = [
    "Generalist_Solution_Generator_Tool: another LLM call to reason or draft from the question and information already obtained. It does not search and can invent facts.",
    "Python_Code_Generator_Tool: write and run one short program for arithmetic or a numeric task where code is more accurate than guessing. It does not look up facts, import libraries, or open pages.",
    "Bing_Search_Tool: search keywords. Returns titles, URLs, and short snippets only. It does not open pages. Snippets are not the page text.",
    "Wikipedia_RAG_Search_Tool: Wikipedia search. Returns titles, URLs, and short snippets only. It does not open pages.",
    "Web_Fetch_Tool: open exactly one URL and return page text or an error. It does not search. A valid URL may still lack the needed fact.",
]

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.I | re.S)
_PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.I | re.S)
# 1.7B sometimes emits the ARPO search wrapper instead of <search>.
_ARPO_SEARCH_RE = re.compile(
    r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", re.I | re.S
)
_URL_RE = re.compile(r"https?://[^\s\"'<>\]\)]+", re.I)
_SEARCH_TOOL_MARKERS = ("bing_search", "wikipedia", "wiki")
_FETCH_TOOL_MARKERS = ("fetch",)
_EMPTY_RESULT_MARKERS = (
    "tool was not found",
    "no result was generated",
    "search_unavailable",
    "fetch_unavailable",
    "output=[]",
    "command: no command found",
)
_OUTLINE_STATUSES = ("pending", "done", "failed")
_PREDICTIONS = ("failure", "partial", "success")


def _clip(text: Any, limit: int = 400) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "..."


def _field(text: Any, limit: int) -> str:
    """Truncate one Memory field to ``limit`` characters, keeping a ``...`` marker."""
    raw = " ".join(str(text or "").split())
    if limit <= 0 or len(raw) <= limit:
        return raw
    if limit <= 3:
        return raw[:limit]
    return raw[: limit - 3] + "..."


def _extract_urls(text: Any, *, limit: int = 5) -> List[str]:
    seen: List[str] = []
    for match in _URL_RE.findall(str(text or "")):
        url = match.rstrip(".,);]")
        if url and url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


def _tool_name_of(action: Dict[str, Any]) -> str:
    return str(action.get("tool_name") or action.get("tool") or "").lower()


def _result_text(action: Dict[str, Any]) -> str:
    return str(action.get("result") or action.get("output") or "")


def recent_has_search_urls(memory: Any) -> bool:
    """True when the latest search step exposed candidate URLs for Web_Fetch."""
    actions = memory.get_actions() if memory is not None else {}
    items = list((actions or {}).items())[-3:]
    for _key, action in reversed(items):
        if not isinstance(action, dict):
            continue
        name = _tool_name_of(action)
        if any(m in name for m in _FETCH_TOOL_MARKERS):
            continue
        if any(m in name for m in _SEARCH_TOOL_MARKERS) and _extract_urls(_result_text(action)):
            return True
    return False


def recent_has_fetch_body(memory: Any) -> bool:
    actions = memory.get_actions() if memory is not None else {}
    for _key, action in reversed(list((actions or {}).items())[-4:]):
        if not isinstance(action, dict):
            continue
        if not any(m in _tool_name_of(action) for m in _FETCH_TOOL_MARKERS):
            continue
        body = _result_text(action).strip()
        if len(body) >= 80 and not any(m in body.lower() for m in _EMPTY_RESULT_MARKERS):
            return True
    return False


def recent_steps_look_empty(memory: Any) -> bool:
    """No usable tool evidence yet (empty / not-found / snippet-only without fetch)."""
    actions = memory.get_actions() if memory is not None else {}
    items = list((actions or {}).items())
    if not items:
        return True
    if recent_has_fetch_body(memory):
        return False
    for _key, action in items:
        if not isinstance(action, dict):
            continue
        body = _result_text(action).lower()
        if body and not any(m in body for m in _EMPTY_RESULT_MARKERS):
            # Snippets alone are weak; still not "empty" for budget-stop gating
            # when search returned something — Verifier should CONTINUE for fetch.
            if any(m in _tool_name_of(action) for m in _SEARCH_TOOL_MARKERS):
                continue
            if len(body) >= 40:
                return False
    return True


def memory_field_limit() -> int:
    """Same per-field cap ``Memory.add_action`` stored, so the prompt is not a second clip.

    ``memory_result_chars: 0`` means the official 100000-character truncation.
    """
    raw = 0
    try:
        from context import load_runtime

        raw = int(load_runtime().get("memory_result_chars") or 0)
    except Exception:
        raw = 0
    if raw <= 0:
        try:
            import yaml

            path = Path(__file__).resolve().parent.parent / "contracts" / "runtime.yaml"
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            raw = int(cfg.get("memory_result_chars") or 0)
        except Exception:
            raw = 0
    return raw if raw > 0 else 100000


def recent_steps(memory: Any, n: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Every stored action, including command. ``n`` keeps a tail only when set.

    Training caps the episode at 3 steps, so the full list is the whole history.
    Eval may run longer; those steps stay visible instead of dropping to the last 3.
    """
    actions = memory.get_actions() if memory is not None else {}
    items = list((actions or {}).items())
    if n is not None:
        items = items[-int(n):]
    if not items:
        return "(none)"
    cap = memory_field_limit() if limit is None else int(limit)
    lines: List[str] = []
    for key, action in items:
        if not isinstance(action, dict):
            lines.append(f"{key}: {_clip(action, cap)}")
            continue
        result = _result_text(action)
        urls = _extract_urls(result)
        url_line = f"\nurls: {' | '.join(urls)}" if urls else ""
        extra = []
        for label in ("prediction", "actual", "conclusion"):
            value = action.get(label)
            if value:
                extra.append(f"{label}: {_field(value, 200)}")
        extra_line = ("\n" + "\n".join(extra)) if extra else ""
        lines.append(
            f"{key}: tool={action.get('tool_name')} sub_goal={_field(action.get('sub_goal'), cap)}\n"
            f"command: {_field(action.get('command'), cap)}\n"
            f"result: {_field(result, cap)}{url_line}{extra_line}"
        )
    return "\n".join(lines)


def ensure_memory_fields(memory: Any) -> None:
    """Hang outline and obtained facts on the existing Memory instance."""
    if memory is None:
        return
    if not isinstance(getattr(memory, "outline", None), list):
        memory.outline = []
    if not isinstance(getattr(memory, "obtained_information", None), list):
        memory.obtained_information = []


def normalize_outline(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            subgoal, status = item.strip(), "pending"
        elif isinstance(item, dict):
            subgoal = str(item.get("subgoal") or item.get("sub_goal") or "").strip()
            status = str(item.get("status") or "pending").strip().lower()
        else:
            continue
        if not subgoal or _is_planner_placeholder(subgoal):
            continue
        if status not in _OUTLINE_STATUSES:
            status = "pending"
        out.append({"subgoal": subgoal, "status": status})
    return out


def apply_outline(memory: Any, raw: Any) -> None:
    """Replace the outline only. History actions stay."""
    parsed = normalize_outline(raw)
    if not parsed or memory is None:
        return
    ensure_memory_fields(memory)
    memory.outline = parsed


def render_outline(memory: Any) -> str:
    items = getattr(memory, "outline", None) or []
    if not items:
        return "(none)"
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lines.append(f"- [{item.get('status') or 'pending'}] {item.get('subgoal')}")
    return "\n".join(lines) or "(none)"


def render_obtained(memory: Any) -> str:
    items = getattr(memory, "obtained_information", None) or []
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def memory_view(memory: Any) -> str:
    return (
        f"Outline:\n{render_outline(memory)}\n\n"
        f"Obtained information:\n{render_obtained(memory)}\n\n"
        f"Recent steps:\n{recent_steps(memory)}"
    )


def _norm_fact(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def add_obtained_information(memory: Any, text: Any) -> bool:
    """Append one verifier fact. Skip NONE, a repeated sentence, or a repeated URL."""
    raw = " ".join(str(text or "").split())
    if _blank_obtained(raw) or memory is None:
        return False
    ensure_memory_fields(memory)
    key = _norm_fact(raw)
    held = [str(item) for item in memory.obtained_information]
    if any(_norm_fact(item) == key for item in held):
        return False
    new_urls = [url.lower() for url in _extract_urls(raw)]
    old_urls = {url.lower() for item in held for url in _extract_urls(item)}
    prose = _norm_fact(_URL_RE.sub(" ", raw))
    old_prose = {_norm_fact(_URL_RE.sub(" ", item)) for item in held}
    if new_urls and set(new_urls) <= old_urls and (not prose or prose in old_prose):
        return False
    memory.obtained_information.append(raw[:400])
    return True


def normalize_prediction(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in _PREDICTIONS else ""


def actual_label(result: Any) -> str:
    """Tool success is not subgoal success. Non-empty output stays partial."""
    text = str(result if result is not None else "").strip()
    low = text.lower()
    if (
        not text
        or low.startswith(("error", "execution timed out", "search_unavailable", "fetch_unavailable"))
        or any(marker in low for marker in _EMPTY_RESULT_MARKERS)
    ):
        return "failure"
    return "partial"


def unresolved_note(memory: Any) -> str:
    pending = [
        str(item.get("subgoal"))
        for item in (getattr(memory, "outline", None) or [])
        if isinstance(item, dict) and item.get("status") in ("pending", "failed") and item.get("subgoal")
    ]
    if not pending:
        return ""
    return "unresolved: " + "; ".join(pending)


def budget_unresolved_reason(reason: str, memory: Any) -> str:
    note = unresolved_note(memory)
    if not note:
        return str(reason or "")
    return (str(reason or "") + " | " + note).strip(" |")


def _last_action(memory: Any) -> Optional[Dict[str, Any]]:
    actions = memory.get_actions() if memory is not None else {}
    items = [action for _key, action in (actions or {}).items() if isinstance(action, dict)]
    return items[-1] if items else None


def _repair_block(memory: Any) -> str:
    action = _last_action(memory)
    if not action:
        return ""
    actual = str(action.get("actual") or "")
    if actual not in ("failure", "partial"):
        return ""
    tool = action.get("tool_name") or ""
    tool_input = action.get("tool_input") or action.get("command") or ""
    prediction = action.get("prediction") or "(none)"
    return (
        f"Last prediction: {prediction}\n"
        f"Actual: {actual}. A returned page or snippet does not finish the subgoal.\n"
        f"Last tool: {tool}\n"
        f"Last input: {_clip(tool_input, 180)}\n"
        "Do one of these and name it in prediction_reason: split this subgoal, "
        "switch to another listed tool, or change the query or URL.\n"
        "Do not repeat the same tool with the same input.\n"
    )


def tags_to_execute(tool_name: str, raw: str, extracted: str) -> str:
    """Turn the last ``<search>`` or ``<python>`` tag into ``execution = tool.execute(...)``.

    Search-like tools and the generalist take ``query=``. ``Web_Fetch_Tool`` takes
    ``url=``. The Python tool only reads ``<python>``. No tag leaves ``extracted``.
    """
    name = str(tool_name or "")
    text = str(raw or "")
    is_python = "python" in name.lower()
    is_fetch = "fetch" in name.lower()
    if is_python:
        bodies = [b.strip() for b in _PYTHON_RE.findall(text) if b.strip()]
        if not bodies:
            return extracted
        return f"execution = tool.execute(query={repr(bodies[-1])})"
    bodies = [b.strip() for b in _SEARCH_RE.findall(text) if b.strip()]
    if not bodies:
        bodies = [b.strip() for b in _ARPO_SEARCH_RE.findall(text) if b.strip()]
    if not bodies:
        return extracted
    key = "url" if is_fetch else "query"
    return f"execution = tool.execute({key}={repr(bodies[-1])})"


def _tools_block() -> str:
    return "\n".join(f"- {line}" for line in TOOL_LINES)


def _tool_names() -> List[str]:
    return [line.split(":", 1)[0] for line in TOOL_LINES]


_PLACEHOLDER_VALUES = {
    "task",
    "this step",
    "one short line",
    "query or one url",
}


def _plain_field(text: Any) -> str:
    raw = str(text or "").strip().strip("<>").strip().strip("*`\"'")
    return " ".join(raw.split())


def _is_planner_placeholder(text: Any) -> bool:
    """Prompt leftovers the model copies instead of a real plan."""
    raw = _plain_field(text).lower()
    if not raw:
        return True
    if raw in _PLACEHOLDER_VALUES:
        return True
    return raw.startswith("one of ")


def _registered_tools_in(text: str) -> List[str]:
    raw = str(text or "")
    return [name for name in _tool_names() if name in raw]


def _blank_obtained(text: Any) -> bool:
    """NONE, null, n/a, and sentences that only start by saying there is no fact."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return True
    low = raw.lower()
    if low in {"none", "null", "n/a"}:
        return True
    return re.match(r"none\b", low) is not None


def _default_reason(conclusion: str) -> str:
    if conclusion == "STOP":
        return "现有信息够回答"
    return "关键事实还没有"


def _bare_tool_token(candidate: str) -> str:
    """Drop ``</think>`` and angle brackets glued onto an otherwise exact tool name.

    A clean ``Bing_Search_Tool`` is unchanged. ``<Bing_Search_Tool></think>`` becomes
    ``Bing_Search_Tool``.
    """
    text = re.sub(r"</?think>", " ", str(candidate or ""), flags=re.I)
    text = text.replace("<", " ").replace(">", " ")
    text = text.strip().strip("*`\"'")
    if not text:
        return ""
    return text.split()[0].strip("*`\"'.,:;")


def _json_blobs(text: str) -> List[str]:
    raw = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.I | re.S)
    blobs: List[str] = []
    start = 0
    while True:
        i = raw.find("{", start)
        if i < 0:
            break
        depth = 0
        closed = False
        for j in range(i, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    blobs.append(raw[i:j + 1])
                    start = j + 1
                    closed = True
                    break
        if not closed:
            break
    return blobs


def parse_planner_payload(text: str) -> Optional[Dict[str, Any]]:
    """Read the planner JSON. ``None`` means the caller should use the three lines.

    A JSON is skipped when its tool field is a prompt leftover or does not
    contain exactly one registered tool name. Later JSON in the same reply
    can still be used. Placeholder outline rows are dropped; an empty outline
    does not replace the previous plan.
    """
    for blob in _json_blobs(text):
        try:
            payload = json.loads(blob)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        tool_field = str(payload.get("selected_tool") or payload.get("tool_name") or "")
        if _is_planner_placeholder(tool_field):
            continue
        found = _registered_tools_in(tool_field)
        if len(found) != 1:
            continue
        tool_input = payload.get("tool_input")
        if tool_input is None:
            tool_input = payload.get("context") or ""
        if not isinstance(tool_input, str):
            tool_input = json.dumps(tool_input, ensure_ascii=False)
        tool_input = tool_input.strip()
        if _is_planner_placeholder(tool_input):
            tool_input = ""
        subgoal = str(payload.get("subgoal") or payload.get("sub_goal") or "").strip()
        if _is_planner_placeholder(subgoal):
            subgoal = tool_input
        reason = str(payload.get("prediction_reason") or "").strip()
        if _is_planner_placeholder(reason):
            reason = ""
        outline = normalize_outline(payload.get("outline"))
        return {
            "outline": outline,
            "subgoal": subgoal,
            "tool_name": found[0],
            "tool_input": tool_input,
            "prediction": normalize_prediction(payload.get("prediction")),
            "prediction_reason": reason,
        }
    return None


def _planner_for_parser(text: str) -> str:
    """Turn planner JSON into the three lines the official extractor already reads.

    A broken JSON falls back to the first Context / Sub-Goal / Tool Name triple.
    The official extractor swallows everything up to a blank line into Tool Name.
    A trailing ``</think>`` or extra paragraph makes a real tool name miss.
    """
    parsed = parse_planner_payload(text)
    if parsed and parsed.get("subgoal") and parsed.get("tool_name"):
        context = parsed.get("tool_input") or parsed["subgoal"]
        return (
            f"Context: {context}\n"
            f"Sub-Goal: {parsed['subgoal']}\n"
            f"Tool Name: {parsed['tool_name']}"
        )
    lines = str(text or "").replace("**", "").splitlines()
    context = ""
    subgoal = ""
    ctx_i: Optional[int] = None
    sub_i: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip().strip("`")
        if re.match(r"Context:\s*\S", stripped):
            value = stripped.split(":", 1)[1].strip()
            if _is_planner_placeholder(value):
                context = ""
                subgoal = ""
                ctx_i = None
                sub_i = None
                continue
            context = value
            subgoal = ""
            ctx_i = i
            sub_i = None
            continue
        if re.match(r"Sub-Goal:\s*\S", stripped):
            value = stripped.split(":", 1)[1].strip()
            if _is_planner_placeholder(value):
                value = context
            if value and ctx_i is not None:
                subgoal = value
                sub_i = i
            continue
        match = re.match(r"Tool Name:\s*(.+)$", stripped, re.I)
        if not match or ctx_i is None or sub_i is None or not context or not subgoal:
            continue
        raw_tool = match.group(1)
        if _is_planner_placeholder(raw_tool):
            continue
        found = _registered_tools_in(raw_tool)
        if len(found) != 1 or not (ctx_i < sub_i < i):
            continue
        return (
            f"Context: {context}\n"
            f"Sub-Goal: {subgoal}\n"
            f"Tool Name: {found[0]}"
        )
    return str(text or "")


_CONCLUSION_LINE = re.compile(r"Conclusion:\s*(STOP|CONTINUE)\b", re.I)


def parse_conclusion(text: str) -> str | None:
    """First ``Conclusion: STOP|CONTINUE`` only.

    1.7B writes that line, then keeps talking. A later ``conclusion followed``
    or a bare ``stop`` inside the ramble must not replace it. No Conclusion
    line means there is no verdict; callers default to CONTINUE instead of
    searching the substring ``stop``.
    """
    match = _CONCLUSION_LINE.search(str(text or ""))
    if not match:
        return None
    return match.group(1).upper()


_REASON_LINE = re.compile(r"^Reason:\s*(.*)$", re.I)
_OBTAINED_LINE = re.compile(r"^Obtained:\s*(.*)$", re.I)


def parse_verifier_text(text: str) -> Dict[str, str]:
    """Conclusion line, or a JSON decision, mapped to the existing STOP/CONTINUE token."""
    raw = str(text or "")
    for blob in _json_blobs(raw):
        try:
            payload = json.loads(blob)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        decision = str(payload.get("decision") or payload.get("conclusion") or "").strip().lower()
        if decision not in ("stop", "continue"):
            continue
        obtained = payload.get("obtained")
        if obtained is None:
            obtained = payload.get("obtained_information") or "NONE"
        if not isinstance(obtained, str):
            obtained = json.dumps(obtained, ensure_ascii=False)
        obtained_text = obtained.strip()
        if _blank_obtained(obtained_text):
            obtained_text = "NONE"
        conclusion = decision.upper()
        reason = str(payload.get("reason") or "").strip()
        return {
            "conclusion": conclusion,
            "reason": reason or _default_reason(conclusion),
            "obtained": obtained_text,
        }
    conclusion = parse_conclusion(raw) or "CONTINUE"
    reason = ""
    obtained = "NONE"
    for line in raw.splitlines():
        reason_match = _REASON_LINE.match(line.strip())
        if reason_match and not reason:
            reason = reason_match.group(1).strip()
        obtained_match = _OBTAINED_LINE.match(line.strip())
        if obtained_match:
            obtained = obtained_match.group(1).strip() or "NONE"
    if _blank_obtained(obtained):
        obtained = "NONE"
    return {
        "conclusion": conclusion,
        "reason": reason or _default_reason(conclusion),
        "obtained": obtained,
    }


def format_verdict(verdict: Dict[str, str]) -> str:
    conclusion = verdict.get("conclusion") if verdict.get("conclusion") in ("STOP", "CONTINUE") else "CONTINUE"
    reason = str(verdict.get("reason") or "").strip() or _default_reason(conclusion)
    obtained = verdict.get("obtained") or "NONE"
    if _blank_obtained(obtained):
        obtained = "NONE"
    return f"Conclusion: {conclusion}\nReason: {reason}\nObtained: {obtained}"


def _verifier_for_parser(text: str) -> str:
    """Normalize to Conclusion / Reason / Obtained so extract_conclusion still matches."""
    return format_verdict(parse_verifier_text(text))


def extract_conclusion(self, response: Any) -> tuple:
    text = response if isinstance(response, str) else str(response)
    token = parse_conclusion(text)
    if token:
        return text, token
    return text, "CONTINUE"


def analyze_query(self, question: str, image: str) -> str:
    del image
    prompt = f"""Question: {question}

Tools:
{_tools_block()}

Reply with exactly three short lines:
- Goal: the comparison or fact the question asks for
- Facts needed: the missing facts
- Likely tools: one or two tool names from the list
"""
    self.query_analysis = str(self.llm_engine_fixed(prompt) or "").strip()
    return self.query_analysis


def generate_next_step(self, question, image, query_analysis, memory, step_count, max_step_count, json_data=None):
    del image
    fetch_rule = ""
    if recent_has_search_urls(memory) and not recent_has_fetch_body(memory):
        fetch_rule = (
            "Search already listed URLs and no page text has been fetched. "
            "Choose Web_Fetch_Tool and one of those URLs. Do not search again yet.\n"
        )
    prompt = f"""Question: {question}

Tools (copy one name exactly):
{_tools_block()}

{memory_view(memory)}

Notes: {_clip(query_analysis, 300)}
Step {step_count} of {max_step_count}.
{fetch_rule}{_repair_block(memory)}
Rules:
- If you do not know where the fact is, use Bing_Search_Tool or Wikipedia_RAG_Search_Tool.
- If a search URL is relevant and the snippet is not enough, use Web_Fetch_Tool on that URL.
- If a fetch is not enough, do not open that URL again. Change the URL or the keywords.
- If Obtained information already has the fact, do not search or fetch it again.
- Prefer the unfinished subgoal the question needs most.
- A tool that returns text is not done until you set that subgoal to done.
- prediction is about the subgoal, not whether the tool process starts. Search snippets are partial. A page that should contain the fact is success. No usable path is failure.

Reply with one JSON object and nothing else.
Fields: outline, subgoal, selected_tool, tool_input, prediction, prediction_reason
status is pending, done, or failed. prediction is failure, partial, or success.
selected_tool is exactly one name from the tool list.
If you cannot write JSON, use exactly these three lines instead:
Context: <query or one URL>
Sub-Goal: <query or one URL>
Tool Name: <query or one URL>
"""
    raw = str(self.llm_engine(prompt) or "")
    self._last_plan = parse_planner_payload(raw) or {}
    next_step = _planner_for_parser(raw)
    if json_data is not None:
        json_data[f"action_predictor_{step_count}_prompt"] = prompt
        json_data[f"action_predictor_{step_count}_response"] = raw
    return next_step


def generate_tool_command(self, question, image, context, sub_goal, tool_name, tool_metadata, step_count=0, json_data=None):
    del question, image, tool_metadata
    name = str(tool_name or "")
    if "python" in name.lower():
        spec = "Output exactly one line and nothing else:\n<python>expression or short problem</python>"
    elif "fetch" in name.lower():
        spec = (
            "Pick one concrete https URL from Context / Sub-Goal / recent urls. "
            "Output exactly one line and nothing else:\n"
            "<search>https://one-url-from-previous-search</search>"
        )
    else:
        spec = "Output exactly one line and nothing else:\n<search>short query</search>"
    prompt = f"""Tool Name: {name}
Sub-Goal: {_clip(sub_goal, 400)}
Context: {_clip(context, 400)}

{spec}
Do not explain. Do not write tool.execute. Do not write <think>. Do not write <|begin_search_query|>.
"""
    tool_command = self.llm_generate_tool_command(prompt)
    if json_data is not None:
        json_data[f"tool_commander_{step_count}_prompt"] = prompt
        json_data[f"tool_commander_{step_count}_response"] = str(tool_command)
    return tool_command


def verificate_context(self, question, image, query_analysis, memory, step_count=0, json_data=None):
    del image, query_analysis
    steps = recent_steps(memory)
    if steps == "(none)":
        stop_verification = format_verdict({
            "conclusion": "CONTINUE",
            "reason": "no tool has run",
            "obtained": "NONE",
        })
        if json_data is not None:
            json_data[f"verifier_{step_count}_response"] = stop_verification
        return stop_verification
    force_continue = recent_has_search_urls(memory) and not recent_has_fetch_body(memory)
    guard = (
        "Search only returned titles, URLs, or snippets. Page text is not fetched. Conclusion must be CONTINUE.\n"
        if force_continue
        else
        "STOP only when the information already answers the question.\n"
        "A successful tool call is not enough. Partial information means CONTINUE.\n"
        "If a method failed, the next step should change the keywords, URL, or tool.\n"
        "Do not treat a missing fact as true.\n"
    )
    prompt = f"""Question: {question}

{memory_view(memory)}

{guard}Reply with exactly three lines and nothing else:
Conclusion: STOP or CONTINUE
Reason: one short line
Obtained: one new fact that is not already listed, or NONE
"""
    raw = str(self.llm_engine_fixed([prompt]) or "")
    verdict = parse_verifier_text(raw)
    if force_continue:
        verdict["conclusion"] = "CONTINUE"
    stop_verification = format_verdict(verdict)
    if json_data is not None:
        json_data[f"verifier_{step_count}_prompt"] = prompt
        json_data[f"verifier_{step_count}_response"] = raw
    return stop_verification


def generate_direct_output(self, question, image, memory) -> str:
    del image
    prompt = f"""Question: {question}

{memory_view(memory)}

Use only facts listed above. If a fact is missing, do not invent it.
Output exactly this and nothing else:
<answer>
\\boxed{{short answer}}
</answer>
"""
    return str(self.llm_engine_fixed(prompt) or "")


def bind_episode(planner, executor, verifier) -> None:
    planner.analyze_query = MethodType(analyze_query, planner)
    planner.generate_next_step = MethodType(generate_next_step, planner)
    planner.generate_direct_output = MethodType(generate_direct_output, planner)
    executor.generate_tool_command = MethodType(generate_tool_command, executor)
    if not getattr(verifier, "_sft_bound", False):
        verifier.verificate_context = MethodType(verificate_context, verifier)
        verifier.extract_conclusion = MethodType(extract_conclusion, verifier)
        verifier._sft_bound = True

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
    "Python_Code_Generator_Tool: write and run one short ASCII program for a numeric task. Import before use. Common modules are math, itertools, statistics, fractions, sympy, and numpy. It must print the result. It does not call input(), look up facts, or open pages.",
    "Bing_Search_Tool: search keywords. Returns titles, URLs, and short snippets only. It does not open pages. Snippets are not the page text.",
    "Web_Fetch_Tool: open one to three URLs copied from the latest Bing result, one per line. It does not search and does not invent URLs. A valid URL may still lack the needed fact.",
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
_PREDICTIONS = ("满足", "不满足")
_OUTLINE_HEAD = re.compile(r"(?im)^outline\s*:")
_STEP_PREFIX = re.compile(r"^(?:\d+[\.\)]\s*|[-*]\s+)")


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


def latest_bing_urls(memory: Any, *, limit: int = 10) -> List[str]:
    """URLs from the newest Bing result, if that step returned any."""
    actions = memory.get_actions() if memory is not None else {}
    for _key, action in reversed(list((actions or {}).items())):
        if not isinstance(action, dict):
            continue
        if "bing" not in _tool_name_of(action):
            continue
        return _extract_urls(_result_text(action), limit=limit)
    return []


def listed_fetch_command(tool_name: str, tool_input: Any, memory: Any) -> str:
    """Execute copied Bing URLs directly. An invented URL returns an empty string."""
    if "fetch" not in str(tool_name or "").lower():
        return ""
    wanted = _extract_urls(tool_input, limit=3)
    if not wanted:
        return ""
    listed = {url.rstrip("/") for url in latest_bing_urls(memory)}
    if not listed:
        return ""
    if any(url.rstrip("/") not in listed for url in wanted):
        return ""
    return f"execution = tool.execute(url={repr(chr(10).join(wanted))})"


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


_PROMPT_LAST_CHARS = 800
_PROMPT_EARLIER_CHARS = 240
_QUANTITY_WORD = re.compile(
    r"\b(how many|how much|total|left|remain(?:ing|der)?|altogether|each|per|times|cost)\b",
    re.I,
)
_INTEGER = re.compile(r"\b\d+\b")


def question_needs_python(question: Any) -> bool:
    """Arithmetic already stated in the question, not a fact looked up by name.

    One digit is enough when a quantity word is present. Some problems spell
    the other number ("three-quarters") and would otherwise miss the rule.
    """
    text = str(question or "")
    if not _QUANTITY_WORD.search(text):
        return False
    return len(_INTEGER.findall(text)) >= 1


def _program_key(command: str) -> str:
    from tools_local import python_source_from_command

    return " ".join(python_source_from_command(command).split())


def _python_actions(memory: Any) -> List[Dict[str, Any]]:
    actions = memory.get_actions() if memory is not None and hasattr(memory, "get_actions") else {}
    found = []
    for _key, action in (actions or {}).items():
        if isinstance(action, dict) and "python" in str(action.get("tool_name") or "").lower():
            found.append(action)
    return found


def repeated_python_program(command: str, memory: Any) -> bool:
    """True when this program is the same as the previous Python step."""
    key = _program_key(command)
    if not key:
        return False
    previous = _python_actions(memory)
    if not previous:
        return False
    return _program_key(str(previous[-1].get("command") or "")) == key


def _unwrap_result_text(result: Any) -> str:
    text = result
    if isinstance(result, list) and len(result) == 1:
        text = result[0]
    text = str(text or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list) and len(parsed) == 1:
            text = str(parsed[0]).strip()
    return text


def last_python_error(memory: Any) -> str:
    previous = _python_actions(memory)
    if not previous:
        return ""
    line = _unwrap_result_text(previous[-1].get("result")).splitlines()
    head = line[0] if line else ""
    low = head.lower()
    if low.startswith("error") or "syntaxerror" in low:
        return head[:240]
    return ""


def last_python_print(memory: Any) -> str:
    """One short successful Python print, newest first."""
    for action in reversed(_python_actions(memory)):
        text = _unwrap_result_text(action.get("result"))
        low = text.lower()
        if not text or "\n" in text or len(text) > 80:
            continue
        if low.startswith("error") or "syntaxerror" in low:
            continue
        return text
    return ""


def recent_steps(memory: Any, n: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Every stored action, including command. ``n`` keeps a tail only when set.

    The prompt view clips results: 800 characters on the latest step and 240
    on earlier steps. Stored Memory is unchanged. An explicit ``limit`` applies
    that cap to every step.
    """
    actions = memory.get_actions() if memory is not None else {}
    items = list((actions or {}).items())
    if n is not None:
        items = items[-int(n):]
    if not items:
        return "(none)"
    lines: List[str] = []
    last = len(items) - 1
    for index, (key, action) in enumerate(items):
        if limit is not None:
            cap = int(limit)
        else:
            cap = _PROMPT_LAST_CHARS if index == last else _PROMPT_EARLIER_CHARS
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


def normalize_outline(raw: Any) -> List[str]:
    """Ordered remaining steps. Status fields are ignored."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            raw = parsed
        elif isinstance(parsed, str) and parsed.strip():
            raw = [parsed]
        else:
            raw = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or _OUTLINE_HEAD.match(stripped):
                    continue
                raw.append(_STEP_PREFIX.sub("", stripped).strip())
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            step = item.strip()
        elif isinstance(item, dict):
            step = str(item.get("step") or item.get("subgoal") or item.get("sub_goal") or "").strip()
        else:
            continue
        step = _STEP_PREFIX.sub("", step).strip()
        if not step or _is_planner_placeholder(step):
            continue
        out.append(step)
    return out


def apply_outline(memory: Any, raw: Any) -> None:
    """Replace the outline only. History actions stay."""
    parsed = normalize_outline(raw)
    if not parsed or memory is None:
        return
    ensure_memory_fields(memory)
    memory.outline = parsed


def render_outline(memory: Any) -> str:
    items = normalize_outline(getattr(memory, "outline", None) or [])
    if not items:
        return "(none)"
    return "\n".join(f"{index}. {step}" for index, step in enumerate(items, 1))


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
    """Only the two planner labels. ``partial`` and English words are dropped."""
    token = str(value or "").strip().strip("\"'").strip()
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
    steps = normalize_outline(getattr(memory, "outline", None) or [])
    if not steps:
        return ""
    return "unresolved: " + "; ".join(steps)


def bind_current_subgoal(subgoal: Any, memory: Any) -> str:
    """The step to run is the first remaining outline item."""
    steps = normalize_outline(getattr(memory, "outline", None) or [])
    if steps:
        return steps[0]
    return str(subgoal or "")


def split_query_analysis(raw: Any) -> tuple[str, List[str]]:
    """Notes stay text. Numbered lines under Outline: become the first plan."""
    text = str(raw or "").strip()
    match = _OUTLINE_HEAD.search(text)
    if not match:
        return text, []
    notes = text[:match.start()].strip()
    return notes, normalize_outline(text[match.end():])


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
            bare = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
            bare = re.sub(r"</?python>", " ", bare, flags=re.I).strip()
            if re.search(r"\b(print|import)\b", bare):
                bodies = [bare]
        if not bodies:
            return extracted
        return f"execution = tool.execute(query={repr(bodies[-1])})"
    bodies = [b.strip() for b in _SEARCH_RE.findall(text) if b.strip()]
    if not bodies:
        bodies = [b.strip() for b in _ARPO_SEARCH_RE.findall(text) if b.strip()]
    if not bodies:
        return extracted
    if is_fetch:
        urls: List[str] = []
        for body in bodies:
            for url in _extract_urls(body, limit=3):
                if url not in urls:
                    urls.append(url)
                if len(urls) >= 3:
                    break
            if len(urls) >= 3:
                break
        if urls:
            return f"execution = tool.execute(url={repr(chr(10).join(urls))})"
        return f"execution = tool.execute(url={repr(bodies[-1])})"
    return f"execution = tool.execute(query={repr(bodies[-1])})"


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


def _compact_name(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def resolve_tool_name(text: Any) -> Optional[str]:
    """One registered tool, or one name that matches after dropping underscores.

    Several registered names, or a prompt leftover such as ``one of ...``, is
    not a choice.
    """
    raw = str(text or "")
    if _is_planner_placeholder(raw):
        return None
    if "wikipedia" in raw.lower() or "wiki" in _compact_name(raw):
        if "bing" not in raw.lower() and "fetch" not in raw.lower() and "python" not in raw.lower():
            return "Bing_Search_Tool"
    found = _registered_tools_in(raw)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return None
    compact = _compact_name(raw)
    if not compact:
        return None
    hits = [name for name in _tool_names() if _compact_name(name) in compact]
    if len(hits) == 1:
        return hits[0]
    return None


FALLBACK_TOOL = "Generalist_Solution_Generator_Tool"
PARSE_FAIL_NOTE = (
    "工具名没有解析出来，所以改用 Base_Generator_Tool。"
    "下一步必须从工具列表里原样复制一个工具名。"
)


PYTHON_FALLBACK = "Python_Code_Generator_Tool"
PYTHON_PARSE_FAIL_NOTE = (
    "工具名没有解析出来，所以改用 Python_Code_Generator_Tool。"
    "下一步必须从工具列表里原样复制一个工具名。"
)
PYTHON_OUTLINE_STEP = "用程序算出题目要的数"


def fallback_unparsed_tool(
    tool_name: Any, context: Any, subgoal: Any, available: Any,
    preferred: Any = None,
) -> tuple[str, str, str, str]:
    """Use a fallback when the parsed name is not a registered tool.

    ``preferred`` is the arithmetic fallback. Otherwise the base generator is
    used. The returned note is empty when the original tool stands.
    """
    name = str(tool_name or "")
    ctx = str(context or "")
    goal = str(subgoal or "").strip()
    names = {str(item) for item in (available or [])}
    if name in names:
        return name, ctx, goal, ""
    fallback = str(preferred or FALLBACK_TOOL)
    if fallback not in names:
        fallback = FALLBACK_TOOL if FALLBACK_TOOL in names else ""
    if not fallback:
        return name, ctx, goal, ""
    note = PYTHON_PARSE_FAIL_NOTE if "python" in fallback.lower() else PARSE_FAIL_NOTE
    merged = f"{note}\n{ctx}".strip() if ctx else note
    return fallback, merged, goal or "用已有信息继续回答", note


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


_LOOSE_STRING = re.compile(
    r'"(selected_tool|tool_name|subgoal|sub_goal|tool_input|context|prediction|prediction_reason)"'
    r'\s*:\s*"((?:\\.|[^"\\])*)"',
    re.I,
)


def _json_string(body: str) -> str:
    try:
        return str(json.loads('"' + body + '"'))
    except Exception:
        return body


def _loose_planner_payload(text: str) -> Optional[Dict[str, Any]]:
    """Read selected_tool from a JSON object that never closes.

    The model often writes a real tool name and then runs out of tokens before
    the final brace. A complete-object parser drops that reply and the step
    falls back to the base generator.
    """
    raw = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.I | re.S)
    fields: Dict[str, str] = {}
    tools: List[str] = []
    for match in _LOOSE_STRING.finditer(raw):
        key = match.group(1).lower()
        value = _json_string(match.group(2)).strip()
        if key in ("selected_tool", "tool_name"):
            resolved = resolve_tool_name(value)
            if resolved:
                tools.append(resolved)
            continue
        fields[key] = value
    unique = list(dict.fromkeys(tools))
    if len(unique) != 1:
        return None
    tool_input = fields.get("tool_input") or fields.get("context") or ""
    if _is_planner_placeholder(tool_input):
        tool_input = ""
    subgoal = fields.get("subgoal") or fields.get("sub_goal") or ""
    if _is_planner_placeholder(subgoal):
        subgoal = tool_input
    reason = fields.get("prediction_reason") or ""
    if _is_planner_placeholder(reason):
        reason = ""
    return {
        "outline": [],
        "subgoal": subgoal,
        "tool_name": unique[0],
        "tool_input": tool_input,
        "prediction": normalize_prediction(fields.get("prediction")),
        "prediction_reason": reason,
    }


def parse_planner_payload(text: str) -> Optional[Dict[str, Any]]:
    """Read the planner JSON. ``None`` means the caller should use the three lines.

    A JSON is skipped when its top-level tool field is a prompt leftover or
    does not contain exactly one registered tool name. A tool name written
    only inside an outline step does not count. Later JSON in the same reply
    can still be used. Placeholder outline rows are dropped; an empty outline
    does not replace the previous plan. When the outline is non-empty, the
    current subgoal is its first step. A reply that names one tool and then
    breaks the JSON still keeps that tool. A closed object is not scanned
    again for a tool name nested inside its outline.
    """
    saw_object = False
    for blob in _json_blobs(text):
        try:
            payload = json.loads(blob)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        saw_object = True
        tool_field = str(payload.get("selected_tool") or payload.get("tool_name") or "")
        tool = resolve_tool_name(tool_field)
        if tool is None:
            continue
        tool_input = payload.get("tool_input")
        if tool_input is None:
            tool_input = payload.get("context") or ""
        if not isinstance(tool_input, str):
            tool_input = json.dumps(tool_input, ensure_ascii=False)
        tool_input = tool_input.strip()
        if _is_planner_placeholder(tool_input):
            tool_input = ""
        outline = normalize_outline(payload.get("outline"))
        subgoal = str(payload.get("subgoal") or payload.get("sub_goal") or "").strip()
        if outline:
            subgoal = outline[0]
        elif _is_planner_placeholder(subgoal):
            subgoal = tool_input
        reason = str(payload.get("prediction_reason") or "").strip()
        if _is_planner_placeholder(reason):
            reason = ""
        return {
            "outline": outline,
            "subgoal": subgoal,
            "tool_name": tool,
            "tool_input": tool_input,
            "prediction": normalize_prediction(payload.get("prediction")),
            "prediction_reason": reason,
        }
    if saw_object:
        return None
    return _loose_planner_payload(text)


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
        found = resolve_tool_name(raw_tool)
        if not found or not (ctx_i < sub_i < i):
            continue
        return (
            f"Context: {context}\n"
            f"Sub-Goal: {subgoal}\n"
            f"Tool Name: {found}"
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
_SATISFIED_LINE = re.compile(r"^Satisfied:\s*(YES|NO)\s*$", re.I)


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
        satisfied_raw = payload.get("satisfied", payload.get("actual_satisfied"))
        if isinstance(satisfied_raw, bool):
            satisfied = "YES" if satisfied_raw else "NO"
        else:
            normalized = str(satisfied_raw or "").strip().upper()
            satisfied = normalized if normalized in ("YES", "NO") else "UNKNOWN"
        return {
            "conclusion": conclusion,
            "reason": reason or _default_reason(conclusion),
            "obtained": obtained_text,
            "satisfied": satisfied,
        }
    conclusion = parse_conclusion(raw) or "CONTINUE"
    reason = ""
    obtained = "NONE"
    satisfied = "UNKNOWN"
    for line in raw.splitlines():
        reason_match = _REASON_LINE.match(line.strip())
        if reason_match and not reason:
            reason = reason_match.group(1).strip()
        obtained_match = _OBTAINED_LINE.match(line.strip())
        if obtained_match:
            obtained = obtained_match.group(1).strip() or "NONE"
        satisfied_match = _SATISFIED_LINE.match(line.strip())
        if satisfied_match:
            satisfied = satisfied_match.group(1).upper()
    if _blank_obtained(obtained):
        obtained = "NONE"
    return {
        "conclusion": conclusion,
        "reason": reason or _default_reason(conclusion),
        "obtained": obtained,
        "satisfied": satisfied,
    }


def format_verdict(verdict: Dict[str, str]) -> str:
    conclusion = verdict.get("conclusion") if verdict.get("conclusion") in ("STOP", "CONTINUE") else "CONTINUE"
    reason = str(verdict.get("reason") or "").strip() or _default_reason(conclusion)
    obtained = verdict.get("obtained") or "NONE"
    satisfied = str(verdict.get("satisfied") or "UNKNOWN").upper()
    if satisfied not in ("YES", "NO"):
        satisfied = "UNKNOWN"
    if _blank_obtained(obtained):
        obtained = "NONE"
    return f"Conclusion: {conclusion}\nSatisfied: {satisfied}\nReason: {reason}\nObtained: {obtained}"


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

Reply with exactly these parts and nothing else:
- Goal: the comparison or fact the question asks for
- Facts needed: the missing facts
- Likely tools: one or two tool names from the list
Outline:
1. first step, one short sentence
2. next step
Use 2 to 4 numbered steps. Each step is work still required to answer the question.
"""
    raw = str(self.llm_engine_fixed(prompt) or "").strip()
    notes, steps = split_query_analysis(raw)
    self.query_analysis = notes
    self._initial_outline = steps
    return self.query_analysis


def generate_next_step(self, question, image, query_analysis, memory, step_count, max_step_count, json_data=None):
    del image
    fetch_rule = ""
    if recent_has_search_urls(memory) and not recent_has_fetch_body(memory):
        fetch_rule = (
            "Search already listed URLs and no page text has been fetched. "
            "Choose Web_Fetch_Tool. tool_input is 1 to 3 relevant URLs copied from that list, one per line. "
            "Do not invent a URL and do not search again yet.\n"
        )
    prompt = f"""Question: {question}

Tools (copy one name exactly):
{_tools_block()}

{memory_view(memory)}

Notes: {_clip(query_analysis, 300)}
Step {step_count} of {max_step_count}.
{fetch_rule}{_repair_block(memory)}
Using the question, the notes, Obtained information, and Recent steps, first rewrite the outline.
The outline is the ordered work still required. Drop a step that is already done or no longer needed.
Then fill subgoal, selected_tool, tool_input, prediction, and prediction_reason for the first outline item only.
Rules:
- If you do not know where the fact is, use Bing_Search_Tool.
- If a search URL is relevant and the snippet is not enough, use Web_Fetch_Tool. tool_input copies 1 to 3 URLs from that Bing result, one per line. Do not invent a URL.
- If a fetch is not enough, do not open those URLs again. Change the URL or the keywords.
- If Obtained information already has the fact, do not search or fetch it again.
- subgoal is the first outline sentence, copied exactly.
- prediction is whether this first item will satisfy the subgoal. Write 满足 or 不满足 and nothing else. Do not write partial.

Reply with one JSON object and nothing else.
Fields, in this order: outline, subgoal, selected_tool, tool_input, prediction, prediction_reason
outline is a list of short step sentences and has no status.
prediction is 满足 or 不满足, with no space after the quote.
selected_tool is exactly one name from the tool list.
If you cannot write JSON, use exactly these three lines instead:
Context: <query or one URL>
Sub-Goal: <query or one URL>
Tool Name: <query or one URL>
"""
    raw = str(self.llm_engine(prompt) or "")
    self._last_plan = parse_planner_payload(raw) or {}
    scored = getattr(self.llm_engine, "last_label_probs", None) or {}
    if isinstance(self._last_plan, dict) and isinstance(scored, dict):
        for key in (
            "p_satisfy", "p_unsatisfy", "s_satisfy", "s_unsatisfy",
            "prediction_mismatch", "satisfy_token_ids", "unsatisfy_token_ids",
            "prediction_token_start",
        ):
            if key in scored:
                self._last_plan[key] = scored[key]
    next_step = _planner_for_parser(raw)
    if json_data is not None:
        json_data[f"action_predictor_{step_count}_prompt"] = prompt
        json_data[f"action_predictor_{step_count}_response"] = raw
    return next_step


def generate_tool_command(self, question, image, context, sub_goal, tool_name, tool_metadata, step_count=0, json_data=None):
    del question, image, tool_metadata
    name = str(tool_name or "")
    if "python" in name.lower():
        from tools_local import PYTHON_CODE_SPEC

        spec = PYTHON_CODE_SPEC
    elif "fetch" in name.lower():
        spec = (
            "Copy 1 to 3 https URLs from the suggestion or the latest Bing list. "
            "Do not invent a URL. Put every URL inside one search tag, one URL per line:\n"
            "<search>\nhttps://first-url-from-previous-search\nhttps://second-url-from-previous-search\n</search>"
        )
    else:
        spec = "Output exactly one line and nothing else:\n<search>short query</search>"
    repair = str(getattr(self, "_python_repair", "") or "").strip()
    repair_line = ""
    if repair and "python" in name.lower():
        repair_line = (
            f"The previous program failed: {_clip(repair, 240)}\n"
            "Write a different program. Do not repeat that program.\n"
        )
    prompt = f"""Tool Name: {name}
Sub-Goal: {_clip(sub_goal, 400)}
Suggestion from the planner (not the final parameter): {_clip(context, 400)}

You choose the final query, URL, or code.
{repair_line}{spec}
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
            "satisfied": "NO",
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

{guard}Judge Satisfied only against the current sub-goal, independently of whether the whole task should stop.
Reply with exactly four lines and nothing else:
Conclusion: STOP or CONTINUE
Satisfied: YES or NO
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


def format_planning_records(records: Any) -> str:
    """Text for a candidate-universe prompt.

    Verified successes and counterexamples stay in their own sections.
    Diagnostic hypotheses are labeled unverified and are not evidence.
    """

    def section(items: Any, title: str, *, verified: bool) -> str:
        rows: List[str] = []
        for record in list(items or ())[:8]:
            tool = _clip(getattr(record, "tool", ""), 80)
            action = _clip(getattr(record, "action", ""), 120)
            note = _clip(getattr(record, "note", ""), 160)
            status = "verified" if verified else "UNVERIFIED hypothesis, not evidence"
            extra = ""
            constraint = getattr(record, "constraint", None)
            condition = getattr(constraint, "condition", None)
            predicate = getattr(condition, "predicate", "") if condition is not None else ""
            if predicate:
                extra = f" predicate={_clip(predicate, 40)}"
            rows.append(f"- {status}: tool={tool} action={action}{extra} note={note}")
        if not rows:
            rows.append("(none)")
        return title + "\n" + "\n".join(rows)

    if records is None:
        successes: Any = ()
        counterexamples: Any = ()
        hypotheses: Any = ()
    else:
        successes = getattr(records, "verified_successes", ())
        counterexamples = getattr(records, "verified_counterexamples", ())
        hypotheses = getattr(records, "diagnostic_hypotheses", ())
    return "\n".join((
        section(successes, "Verified successes:", verified=True),
        section(counterexamples, "Verified counterexamples:", verified=True),
        section(
            hypotheses,
            "Unverified hypotheses (not evidence; do not treat these as facts):",
            verified=False,
        ),
    ))


def _candidate_plans_prompt(question: str, query_analysis: str, memory: Any, records: Any, n: int) -> str:
    count = max(1, int(n))
    return f"""Question: {question}

Tools (copy one name exactly):
{_tools_block()}

{memory_view(memory)}

Notes: {_clip(query_analysis, 300)}

{format_planning_records(records)}

Write {count} different plans that still address the question.
Do not repeat a verified counterexample.
Unverified hypotheses above are not evidence.

Reply with one JSON object and nothing else:
{{"plans":[{{"plan_id":"p1","steps":[{{"subgoal":"one step","action":"what to run","tool":"Bing_Search_Tool","preconditions":[],"required_state":[],"estimated_cost":1}}]}}]}}
plans has {count} objects. Each step.tool is exactly one name from the tool list.
preconditions lists only a premise that step actually depends on.
If you cannot write that JSON, reply with exactly these three lines instead:
Context: query or one URL
Sub-Goal: query or one URL
Tool Name: one tool name
"""


def parse_candidate_plans(text: str, *, limit: int = 4) -> List[Any]:
    """Parse already-structured candidate plans. Invalid output is an empty list."""
    from cghe_types import candidate_plan_from_dict

    cap = max(1, int(limit))
    raw_plans: Optional[List[Any]] = None
    for blob in _json_blobs(text):
        try:
            parsed = json.loads(blob)
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("plans"), list):
            raw_plans = list(parsed["plans"])
            break
        if isinstance(parsed, list) and parsed and all(isinstance(item, dict) for item in parsed):
            raw_plans = list(parsed)
            break
    if not raw_plans:
        return []
    plans: List[Any] = []
    seen = set()
    for index, item in enumerate(raw_plans, 1):
        if len(plans) >= cap or not isinstance(item, dict):
            continue
        steps_in = item.get("steps")
        if not isinstance(steps_in, list) or not steps_in:
            if item.get("subgoal") or item.get("tool") or item.get("selected_tool"):
                steps_in = [item]
            else:
                continue
        step_dicts: List[Dict[str, Any]] = []
        for step in steps_in:
            if not isinstance(step, dict):
                continue
            tool = resolve_tool_name(step.get("tool") or step.get("selected_tool") or step.get("tool_name") or "")
            action = _plain_field(step.get("action") or step.get("tool_input") or "")
            subgoal = _plain_field(step.get("subgoal") or step.get("sub_goal") or "")
            if _is_planner_placeholder(subgoal):
                subgoal = ""
            if not subgoal:
                subgoal = action if action and not _is_planner_placeholder(action) else ""
            if _is_planner_placeholder(action):
                action = subgoal
            if not tool or not subgoal:
                continue
            try:
                cost = float(step.get("estimated_cost", 1.0))
            except (TypeError, ValueError):
                cost = 1.0
            required = step.get("required_state") or []
            if isinstance(required, str):
                required = [required] if required.strip() else []
            elif not isinstance(required, list):
                required = []
            premises: List[Dict[str, str]] = []
            for prem in step.get("preconditions") or []:
                if isinstance(prem, str) and prem.strip():
                    premises.append({"predicate": prem.strip(), "tool": tool})
                    continue
                if not isinstance(prem, dict):
                    continue
                predicate = str(prem.get("predicate") or prem.get("name") or "").strip()
                if not predicate or _is_planner_placeholder(predicate):
                    continue
                premises.append({
                    "predicate": predicate,
                    "tool": str(prem.get("tool") or tool),
                    "resource": str(prem.get("resource") or ""),
                    "identity": str(prem.get("identity") or ""),
                    "parameter": str(prem.get("parameter") or ""),
                    "argument": str(prem.get("argument") or ""),
                    "schema_signature": str(prem.get("schema_signature") or ""),
                })
            step_dicts.append({
                "subgoal": subgoal,
                "action": action or subgoal,
                "tool": tool,
                "preconditions": premises,
                "required_state": required,
                "estimated_cost": cost,
                "step_id": str(step.get("step_id") or ""),
                "predecessor_ids": list(step.get("predecessor_ids") or ()),
                "links_previous": bool(step.get("links_previous", True)),
            })
        if not step_dicts:
            continue
        plan_id = _plain_field(item.get("plan_id") or "") or f"p{index}"
        if plan_id in seen:
            plan_id = f"{plan_id}-{index}"
        seen.add(plan_id)
        context = item.get("context") if isinstance(item.get("context"), dict) else {}
        try:
            plans.append(candidate_plan_from_dict({
                "plan_id": plan_id,
                "steps": step_dicts,
                "context": context,
            }))
        except (TypeError, ValueError, KeyError):
            continue
    return plans


def solicit_candidate_plans(planner: Any, question: str, query_analysis: str, memory: Any, records: Any, n: int) -> List[Any]:
    """One planner call for N structured plans. Any failure returns an empty list."""
    prompt = _candidate_plans_prompt(question, query_analysis, memory, records, n)
    engine = getattr(planner, "llm_engine", None)
    if not callable(engine):
        return []
    try:
        raw = str(engine(prompt) or "")
    except Exception:
        return []
    try:
        return parse_candidate_plans(raw, limit=n)
    except Exception:
        return []


def bind_episode(planner, executor, verifier) -> None:
    planner.analyze_query = MethodType(analyze_query, planner)
    planner.generate_next_step = MethodType(generate_next_step, planner)
    planner.generate_direct_output = MethodType(generate_direct_output, planner)
    executor.generate_tool_command = MethodType(generate_tool_command, executor)
    if not getattr(verifier, "_sft_bound", False):
        verifier.verificate_context = MethodType(verificate_context, verifier)
        verifier.extract_conclusion = MethodType(extract_conclusion, verifier)
        verifier._sft_bound = True

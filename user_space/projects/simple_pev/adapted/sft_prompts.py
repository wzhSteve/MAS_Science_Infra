"""Short prompts for the SFT copy, plus ``<search>`` / ``<python>`` → ``tool.execute``.

Official f-strings stay in the vendored package. These methods are bound onto
the episode's planner / executor / verifier only when ``AGENTFLOW_SFT_PROMPTS``
is on. Tool names are the ones ``extract_context_subgoal_and_tool`` already
matches.
"""

from __future__ import annotations

import re
from types import MethodType
from typing import Any, List

# Exact ``tool_name`` strings. Three of the five classes publish a longer name.
TOOL_LINES: List[str] = [
    "Generalist_Solution_Generator_Tool: answer from the question text.",
    "Python_Code_Generator_Tool: compute one short arithmetic expression.",
    "Bing_Search_Tool: web search; titles, URLs, and short descriptions.",
    "Wikipedia_RAG_Search_Tool: Wikipedia search; titles, URLs, and short descriptions.",
    "Web_Fetch_Tool: open one URL and return its text.",
]

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.I | re.S)
_PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.I | re.S)
# 1.7B sometimes emits the ARPO search wrapper instead of <search>.
_ARPO_SEARCH_RE = re.compile(
    r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", re.I | re.S
)


def _clip(text: Any, limit: int = 400) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "..."


def recent_steps(memory: Any, n: int = 2, limit: int = 400) -> str:
    actions = memory.get_actions() if memory is not None else {}
    items = list((actions or {}).items())[-n:]
    if not items:
        return "(none)"
    lines: List[str] = []
    for key, action in items:
        if not isinstance(action, dict):
            lines.append(f"{key}: {_clip(action, limit)}")
            continue
        lines.append(
            f"{key}: tool={action.get('tool_name')} sub_goal={_clip(action.get('sub_goal'), 160)}\n"
            f"result: {_clip(action.get('result'), limit)}"
        )
    return "\n".join(lines)


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
    return text.split()[0].strip("*`\"'")


def _planner_for_parser(text: str) -> str:
    """Keep the first Context / Sub-Goal / Tool Name triple and drop the ramble after it.

    The official extractor swallows everything up to a blank line into Tool Name.
    A trailing ``</think>`` or extra paragraph makes a real tool name miss.
    """
    names = set(_tool_names())
    lines = str(text or "").replace("**", "").splitlines()
    ctx_i = sub_i = tool_i = None
    tool_clean = ""
    for i, line in enumerate(lines):
        stripped = line.strip().strip("`")
        if ctx_i is None and re.match(r"Context:\s*\S", stripped):
            ctx_i = i
        if sub_i is None and re.match(r"Sub-Goal:\s*\S", stripped):
            sub_i = i
        match = re.match(r"Tool Name:\s*(.+)$", stripped, re.I)
        if match and tool_i is None:
            candidate = _bare_tool_token(match.group(1))
            if candidate in names:
                tool_i = i
                tool_clean = candidate
    if None not in (ctx_i, sub_i, tool_i) and ctx_i < sub_i < tool_i:  # type: ignore[operator]
        kept = lines[ctx_i:tool_i] + [f"Tool Name: {tool_clean}"]
        return "\n".join(kept).strip()
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


def _verifier_for_parser(text: str) -> str:
    """Cut the reply at the first Conclusion line so the official extractor agrees."""
    match = _CONCLUSION_LINE.search(str(text or ""))
    if not match:
        return str(text or "")
    return str(text or "")[: match.end()]


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
    names = ", ".join(line.split(":", 1)[0] for line in TOOL_LINES)
    prompt = f"""Question: {question}

Tools (copy one name exactly):
{_tools_block()}

Recent steps:
{recent_steps(memory)}

Notes: {_clip(query_analysis, 400)}
Step {step_count} of {max_step_count}.

Write one short <think>...</think> block, then stop.
After </think>, the last three lines must be exactly these, with no markdown, no asterisks, and nothing after them:
Context: <facts the tool needs, one line>
Sub-Goal: <one concrete action, one line>
Tool Name: <one of {names}>
Copy one bare tool name. Do not leave angle brackets or </think> on the Tool Name line.
"""
    raw = str(self.llm_engine(prompt) or "")
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
        spec = "Output exactly one line and nothing else:\n<search>https://one-url</search>"
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
        prompt = f"""Question: {question}

Recent steps:
(none)

No tool has run. The entire reply is this one line and nothing else:
Conclusion: CONTINUE
"""
    else:
        prompt = f"""Question: {question}

Recent steps:
{steps}

Reply with one line only. Do not write <think>. Do not explain. Do not use the word conclusion except on that line.
If the steps already contain the facts the question asks for, the line is:
Conclusion: STOP
If they do not, the line is:
Conclusion: CONTINUE
That line is the entire reply. Stop after it.
"""
    raw = str(self.llm_engine_fixed([prompt]) or "")
    stop_verification = _verifier_for_parser(raw)
    if json_data is not None:
        json_data[f"verifier_{step_count}_prompt"] = prompt
        json_data[f"verifier_{step_count}_response"] = raw
    return stop_verification


def generate_direct_output(self, question, image, memory) -> str:
    del image
    prompt = f"""Question: {question}

Recent steps:
{recent_steps(memory)}

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

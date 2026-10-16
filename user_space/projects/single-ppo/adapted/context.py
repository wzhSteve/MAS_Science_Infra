"""Single-PPO episode context: one LLM ReAct tool loop (APPO/ARPO paradigm).

The whole agent is ONE user window. Inside it a chat loop runs the ARPO tag
protocol from ref_Rep/APPO-main (train_10k.parquet system prompt):

    assistant chunk -> open <search>q</search> -> mas.tools kernel
                    -> inject <result>...</result> (800-char cap) as the next
                       user turn -> repeat -> model closes with ONE
                       <answer>...</answer> (nothing may follow it).

Windows are kind-addressed (plan_step / tool_invoke / verify) like agentflow,
but only plan_window carries the real loop; exec/verify mirrors are defensive.
The window NEVER raises: every failure path lands in trace.error with an
empty-but-valid plan_step payload, so the centralized episode still closes.

Import jail (sandbox.assert_user_module_imports): no workflow.* / science_infra
/ rl / agentlightning imports here. mas.* kernels and the plain openai client
are allowed (agentflow precedent).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT = Path(__file__).resolve().parents[1]
ENABLED_TOOLS = [
    "Wikipedia_Search_Tool",
    "Bing_Search_Tool",
    "Web_Fetch_Tool",
    "Python_Coder_Tool",
]

# --- system prompt ------------------------------------------------------------
#
# ARPO-style single template, distilled from ref_Rep/APPO-main
# rl_datasets/train_10k.parquet row 0 (1146 chars). Keeping one template under
# 1200 chars instead of AgentFlow's 6399-char planner prompt is the point of
# the single-agent paradigm.
SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of a wikipedia search tool and a python interpreter tool. "
    "Given a question, you need to first think about the reasoning process, then "
    "provide the answer. During thinking, you can invoke the wikipedia search "
    "tool for facts and the python interpreter tool for calculations. The search "
    "query and result are enclosed within <search> </search> and <result> "
    "</result> tags respectively, and the python code and its output are "
    "enclosed within <python> </python> and <result> </result> tags "
    "respectively. For example: <search> query here </search> <result> search "
    "result here </result> <python> print(1 + 1) </python> <result> python "
    "interpreter result here </result>. Prefix a search query with bing: to use "
    "Bing web search instead of wikipedia. A query that is a URL opens that "
    "page. When you are ready, output the final answer enclosed within exactly "
    "one <answer> </answer> tag pair. Never generate any content after the "
    "</answer> tag."
)

# --- tag grammar --------------------------------------------------------------

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)
_PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.S)
_RESULT_OPEN_RE = re.compile(r"<result>")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.I)
# Instruction tails that datasets append to every question (the centralized
# runtime also appends its own "Decide the next step..." nudge in payload.input;
# those must never leak into a search query).
_INSTRUCTION_TAIL_RE = re.compile(
    r"(?i)\??\s*when\s+ready\s*,?\s*output\s+the\s+final\s+answer.*$"
    r"|\?\s*when\s+ready\b.*$"
    r"|do\s+not\s+generate\s+any\s+content\b.*$"
    r"|decide\s+the\s+next\s+step.*$",
    re.S,
)
_MAX_QUERY_CHARS = 160
_REPLY_BUDGET_DEFAULT = 800

_CTX: Optional["EpisodeContext"] = None


# --- runtime.yaml -------------------------------------------------------------


def _load_runtime() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "n": 1,
        "max_steps": 6,        # internal ReAct loop LLM-call budget
        "max_time": 240,      # seconds; below the gateway's 300s window timeout
        "max_tokens": 1024,   # per-chunk completion budget (RL budget: 1024)
        "temperature": 0.7,   # agentflow/fast alignment: rollout sampling temp
        "enabled_tools": list(ENABLED_TOOLS),
        "reply_budget": _REPLY_BUDGET_DEFAULT,  # <result> truncation, chars
    }
    path = _PROJECT / "contracts" / "runtime.yaml"
    if not path.is_file():
        return cfg
    try:
        import yaml
    except ImportError:
        return cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict):
        cfg.update(raw)
    tools = cfg.get("enabled_tools")
    if not isinstance(tools, list) or not tools:
        cfg["enabled_tools"] = list(ENABLED_TOOLS)
    else:
        cfg["enabled_tools"] = [str(item) for item in tools]
    try:
        int(cfg.get("reply_budget") or 0)
    except (TypeError, ValueError):
        cfg["reply_budget"] = _REPLY_BUDGET_DEFAULT
    return cfg


def enabled_tool_names() -> List[str]:
    names = [str(item) for item in (_load_runtime().get("enabled_tools") or [])]
    return names or list(ENABLED_TOOLS)


# --- LLM env + client ---------------------------------------------------------


def _map_llm_env() -> str:
    base = (
        os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if base:
        os.environ["OPENAI_API_BASE_URL"] = base
        os.environ["OPENAI_API_BASE"] = base
        os.environ["OPENAI_BASE_URL"] = base
    model = (
        os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL_Name")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or "gpt-4o"
    ).strip()
    os.environ["MODEL_NAME"] = model
    os.environ["MODEL_Name"] = model
    os.environ["OPENAI_MODEL"] = model
    os.environ["MODEL"] = model
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    return model


def _clamp_max_tokens(requested: Any, prompt: Any) -> int:
    try:
        want = int(requested)
    except (TypeError, ValueError):
        want = 512
    runtime = _load_runtime()
    cap = int(runtime.get("max_tokens") or 1024)
    model_len = int(os.environ.get("TIR_MAX_MODEL_LEN") or 0) or 8192
    text = "" if prompt is None else (prompt if isinstance(prompt, str) else str(prompt))
    # Rough tokens estimate (like agentflow's clamping); assume ~3 chars/token.
    prompt_tokens = max(1, (len(text) + 3) // 3)
    remaining = max(64, model_len - prompt_tokens - 64)
    return max(64, min(want, cap, remaining))


_NO_THINK_SUFFIX = "\n/no_think"


def _merge_no_think(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    extra = dict(kwargs.get("extra_body") or {})
    extra["enable_thinking"] = False
    ck = dict(extra.get("chat_template_kwargs") or {})
    ck["enable_thinking"] = False
    extra["chat_template_kwargs"] = ck
    reason = extra.get("reasoning")
    extra["reasoning"] = dict(reason) if isinstance(reason, dict) else {}
    extra["reasoning"]["enabled"] = False
    out = dict(kwargs)
    out["extra_body"] = extra
    # Qwen3 honours the soft /no_think switch inside the message content.
    messages = out.get("messages")
    if isinstance(messages, list) and messages:
        for item in reversed(messages):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                if not content.rstrip().endswith("/no_think"):
                    item["content"] = content.rstrip() + _NO_THINK_SUFFIX
            break
    return out


_LLM_CALL_LOG: List[Dict[str, Any]] = []
_LLM_CALL_LOG_CAP = 32


def _llm_chat(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout_s: float = 60.0,
) -> str:
    """One chat completion chunk. Patched in tests; raises on transport failure."""
    _map_llm_env()
    base = (
        os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()
    if not base:
        raise RuntimeError("no OPENAI_API_BASE_URL bound for the single-ppo window")
    payload_text = " ".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))
    out = ""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=base,
            timeout=timeout_s,
        )
        kwargs = _merge_no_think(
            {
                "model": os.environ.get("MODEL_NAME") or "gpt-4o",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": _clamp_max_tokens(max_tokens, payload_text),
            }
        )
        resp = client.chat.completions.create(**kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        out = ""
        try:
            out = str(choices[0].message.content or "")
        except Exception:  # noqa: BLE001
            out = str(choices[0].text or "")
        return out
    finally:
        # Ring-buffered call log: capped, digest-shape, mirror of agentflow
        # trace.llm_calls so the eval/RL UI can show 模型 N 次调用 lines.
        _LLM_CALL_LOG.append(
            {
                "model": str(os.environ.get("MODEL_NAME") or ""),
                "prompt_head": " ".join(payload_text.split())[:300],
                "response_head": " ".join((out if isinstance(out, str) else "").split())[:200],
            }
        )
        if len(_LLM_CALL_LOG) > _LLM_CALL_LOG_CAP:
            del _LLM_CALL_LOG[: len(_LLM_CALL_LOG) - _LLM_CALL_LOG_CAP]


def drain_llm_calls() -> List[Dict[str, Any]]:
    out = list(_LLM_CALL_LOG)
    _LLM_CALL_LOG.clear()
    return out


def reset_llm_call_log() -> None:
    _LLM_CALL_LOG.clear()


# --- parsing 三板斧 (strip fence -> first balanced JSON -> tail <answer>) -------


def strip_code_fence(text: str) -> str:
    """Strip one outer ``` / ```json / ```python fence, keep the inner body."""
    t = str(text or "").strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", t, re.S)
    if m:
        return m.group(1).strip()
    return t


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First balanced {...} object in the text (handles nested braces/strings).

    Returns the parsed dict, or None when no balanced object exists.
    """
    import json

    blob = strip_code_fence(str(text or ""))
    start = blob.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(blob)):
            ch = blob[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(blob[start : i + 1])
                    except Exception:  # noqa: BLE001
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = blob.find("{", start + 1)
    return None


def extract_tail_answer(text: str) -> str:
    """Tolerant tail <answer> extraction: last paired body, else unclosed tail.

    Preference order (mirrors "尾部 <answer> 正则"):
      1. the LAST complete <answer>...</answer> pair (model's newest intent);
      2. a trailing unclosed <answer>... (truncated stream) -> rest-of-text body;
      3. "" when no answer tag content exists.
    """
    blob = strip_code_fence(str(text or ""))
    pairs = _ANSWER_RE.findall(blob)
    if pairs:
        body = pairs[-1]
        return body.strip()
    opens = list(_ANSWER_OPEN_RE.finditer(blob))
    if opens:
        start = opens[-1].end()
        tail = blob[start:]
        cut = tail.find("</answer>")
        if cut != -1:
            tail = tail[:cut]
        tail = tail.strip()
        if tail:
            return tail
    return ""


def _unwrap_boxed(body: str) -> str:
    """Remove a single wrapping \\boxed{...} (ARPO remove_boxed, tolerant)."""
    text = str(body or "").strip()
    if not text:
        return text
    idx = text.rfind("\\boxed")
    if "\\fbox" in text and idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return text
    probe = text[idx:]
    m = re.match(r"\\boxed\s*(.*)$", probe, re.S)
    if not m:
        return text
    rest = m.group(1)
    if rest.startswith("{"):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return rest[1:i].strip() or text
        return rest[1:].strip() or text
    return rest.strip() or text


# --- tag scan + strict validation (ARPO validate_format, minus boxed mandate) ---


def _open_tag_pair(text: str, pair_re: re.Pattern) -> List[Tuple[int, int, "re.Match"]]:
    return [(m.start(), m.end(), m) for m in pair_re.finditer(text)]


def _find_open_tool(text: str) -> Optional[Tuple[str, str]]:
    """The LAST unclosed <search>q</search> / <python>code</python> pair.

    A pair is unclosed when no <result> block sits between its close and the
    next pair's open (or end of text). Returns (kind, content) where kind is
    "search" or "python", or None.
    """
    searches = _open_tag_pair(text, _SEARCH_RE)
    pythons = _open_tag_pair(text, _PYTHON_RE)
    if not searches and not pythons:
        return None

    def _result_patch(start: int, limit: int) -> bool:
        for m in _RESULT_OPEN_RE.finditer(text):
            if start < m.start() < limit:
                return True
        return False

    candidates: List[Tuple[int, str, str]] = []
    for kind, pairs in (("search", searches), ("python", pythons)):
        for i, (s, e, m) in enumerate(pairs):
            nxt = pairs[i + 1][0] if i + 1 < len(pairs) else len(text)
            # A result between this pair's close and the next open of the SAME
            # kind closes it. Cross-kind results also close (we inject one
            # result per step, appended at text end before the next chunk).
            limit = nxt
            for other_s, other_e, _ in (searches if kind == "python" else pythons):
                if other_s > e:
                    limit = min(limit, other_s)
            if not _result_patch(e, limit):
                candidates.append((s, kind, m.group(1)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    s, kind, content = candidates[-1]
    return kind, content


def validate_format(text: str) -> Tuple[bool, str]:
    """Strict tag validation, ported from ARPO deep_research.validate_format.

    Checks (per the task contract; \\boxed{} deliberately NOT required so QA
    f1 scoring stays compatible):
      - both <answer> and </answer> present, exactly once;
      - every <search>...</search> followed by <result>...</result> in order;
      - every <python>...</python> followed by <result>...</result> in order.
    """
    t = str(text or "")
    if t.count("<answer>") == 0 or t.count("</answer>") == 0:
        return False, "<answer> or </answer> not found"
    if t.count("<answer>") != 1 or t.count("</answer>") != 1:
        return False, "<answer> must appear exactly once"
    answer_start = t.find("<answer>")
    answer_end = t.find("</answer>")
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"

    for open_tag, close_tag in (("<search>", "</search>"), ("<python>", "</python>")):
        current_pos = 0
        while True:
            open_pos = t.find(open_tag, current_pos)
            if open_pos == -1:
                break
            result_pos = t.find("<result>", open_pos)
            close_pos = t.find(close_tag, open_pos)
            result_end_pos = t.find("</result>", result_pos) if result_pos != -1 else -1
            if -1 in (result_pos, close_pos, result_end_pos):
                return False, f"{open_tag[1:-1]}/result tags are incomplete"
            if not (open_pos < close_pos < result_pos < result_end_pos):
                return False, f"{open_tag[1:-1]}/result tags are nested in the wrong order"
            if result_pos < answer_start < result_end_pos:
                return False, f"<answer> nested inside a {open_tag[1:-1]} result block"
            current_pos = result_end_pos
    return True, "format is correct"


# --- tool kernels (forward to mas.tools; cache branch is transparent) ---------


def _condense_query(query: str) -> str:
    """Clean a model-issued <search> query before it hits a kernel.

    Strips dataset instruction tails ("When ready, output..." / the runtime's
    "Decide the next step..." nudge), trims to entity-window size, and cuts at
    the first "?" so residue never leaks into the SERP.
    """
    q = str(query or "").strip()
    text = _INSTRUCTION_TAIL_RE.sub("", q).strip()
    if "?" not in text:
        return text[:_MAX_QUERY_CHARS].strip()
    text = text.split("?", 1)[0]
    return text[:_MAX_QUERY_CHARS].strip()


def _looks_like_url(content: str) -> bool:
    return bool(re.match(r"(?i)^\s*https?://\S+$", str(content or "").strip()))


def _wiki(query: str) -> str:
    q = _condense_query(query)
    if not q:
        return "wikipedia search requires a query"
    try:
        from mas.tools.wikipedia import wikipedia_search as _ws

        return str(_ws(q))
    except Exception as exc:  # noqa: BLE001
        return f"wikipedia unreachable: {exc}"


def _bing(query: str) -> str:
    q = _condense_query(query)
    if not q:
        return "bing search requires a query"
    try:
        from mas.tools.search import bing_search

        out = str(bing_search(q, 10) or "")
        return out
    except Exception as exc:  # noqa: BLE001
        return f"Bing unreachable: {exc}"


def _fetch(url: str) -> str:
    target = str(url or "").strip()
    if not _looks_like_url(target):
        return "web fetch requires a url"
    try:
        from mas.tools.search import fetch_page

        return str(fetch_page(target))
    except Exception as exc:  # noqa: BLE001
        return f"web fetch failed: {exc}"


def _py_execute(code: str) -> str:
    body = str(code or "").strip()
    if not body:
        return "python tool requires code"
    try:
        try:
            from mas.workflow.user_gateway.tools import get_user_tool_agent

            ta = get_user_tool_agent("python_coder")
            if ta is None:
                from mas.tools.tool_agents import get_tool_agent as _gta

                ta = _gta("python_coder")
            if ta is not None:
                return str(ta.invoke({"query": body}))
        except Exception:  # noqa: BLE001
            pass
        from python_tool import execute_python as _ep

        return str(_ep(body))
    except Exception as exc:  # noqa: BLE001
        return f"python tool failed: {exc}"


_TOOL_UNSAFE_TAGS_RE = re.compile(r"</?(?:result|answer|search|python|think)>", re.I)


def run_tool(kind: str, content: str) -> Tuple[str, str]:
    """Dispatch one tool call. Returns (tool_name, output_text)."""
    if kind == "python":
        return "Python_Coder_Tool", _py_execute(content)
    raw = str(content or "").strip()
    lowered = raw.lower()
    if _looks_like_url(raw):
        return "Web_Fetch_Tool", _fetch(raw)
    if lowered.startswith("bing:") or lowered.startswith("bing "):
        tail = raw.split(":", 1)[1].strip() if ":" in raw else raw[4:].strip()
        return "Bing_Search_Tool", _bing(tail)
    if lowered.startswith("google:") or lowered.startswith("web:"):
        # bing kernel is the SERP backend for both user-facing prefixes.
        tail = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return "Bing_Search_Tool", _bing(tail)
    return "Wikipedia_Search_Tool", _wiki(_condense_query(raw))


def _sanitize_for_injection(text: str) -> str:
    t = str(text or "")
    if t.startswith("tier=") and "\n" in t:
        t = t.split("\n", 1)[1]
    t = _TOOL_UNSAFE_TAGS_RE.sub("", t)
    return t.strip()


# --- episode context ----------------------------------------------------------


def _payload(message: Any) -> Dict[str, Any]:
    if hasattr(message, "payload") and isinstance(getattr(message, "payload"), dict):
        return dict(message.payload)
    if isinstance(message, dict):
        inner = message.get("payload")
        if isinstance(inner, dict):
            return dict(inner)
        return dict(message)
    return {}


def _text(*vals: Any) -> str:
    for val in vals:
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


class EpisodeContext:
    """One OpenAI-style ReAct loop per question. Never raises."""

    def __init__(self) -> None:
        self.question = ""
        self.answer_cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self.last_llm_error = ""

    # -- helpers ---------------------------------------------------------------

    def _stop_payload(self, question: str, answer: str, trace: Dict[str, Any]) -> Dict[str, Any]:
        wrapped = f"<answer>{answer}</answer>" if answer else ""
        return {
            "kind": "plan_step",
            "payload": {
                "next": "",
                "args": {"question": question},
                "sub_goal": "",
                "done": True,
                "answer": wrapped[:2000],
                "ready_to_stop": True,
                "trace": trace,
            },
        }

    def _drain_llm_calls(self) -> Dict[str, Any]:
        calls = drain_llm_calls()
        return {
            "count": len(calls),
            "calls": [
                {
                    "model": str(c.get("model") or ""),
                    "prompt_head": str(c.get("prompt_head") or "")[:300],
                    "response_head": str(c.get("response_head") or "")[:200],
                }
                for c in calls
            ],
        }

    # -- the loop --------------------------------------------------------------

    def _run_loop(self, question: str, runtime: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        reset_llm_call_log()
        max_steps = max(1, int(runtime.get("max_steps") or 6))
        max_time = float(runtime.get("max_time") or 240)
        reply_budget = max(64, int(runtime.get("reply_budget") or 800))
        temperature = float(runtime.get("temperature") or 0.7)
        max_tokens = int(runtime.get("max_tokens") or 1024)
        deadline = time.time() + max_time

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        full_text = ""          # composed trajectory (chunks + injected results)
        steps: List[Dict[str, Any]] = []
        n_search = 0
        n_python = 0
        stop_reason = ""
        found_answer = ""
        corrected = False
        self.last_llm_error = ""

        for _step in range(max_steps):
            if time.time() >= deadline:
                stop_reason = "timeout"
                break
            try:
                chunk = _llm_chat(
                    list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=max(10.0, min(90.0, max_time / max(1, max_steps))),
                )
            except Exception as exc:  # noqa: BLE001
                self.last_llm_error = str(exc)
                stop_reason = "error"
                break
            chunk = str(chunk or "")
            messages.append({"role": "assistant", "content": chunk})
            full_text += chunk

            answer_body = extract_tail_answer(chunk)
            if answer_body and chunk.count("<answer>") >= 1:  # a closing answer chunk
                found_answer = answer_body
                stop_reason = "answer"
                break

            open_tool = _find_open_tool(full_text)
            if open_tool is not None:
                kind, content = open_tool
                tool_name, output = run_tool(kind, content)
                if kind == "python":
                    n_python += 1
                else:
                    n_search += 1
                injected = _sanitize_for_injection(output)[:reply_budget]
                steps.append(
                    {
                        "tool": tool_name,
                        "query": str(content)[:200],
                        "ok": not _tool_failed(injected),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<result>\n{injected}\n</result>\n"
                            "Continue reasoning. When ready, output the final answer "
                            "enclosed within exactly one <answer> </answer> tag pair, "
                            "with no content after it."
                        ),
                    }
                )
                full_text += f"<result>\n{injected}\n</result>\n"
                continue

            # Neither answer nor tool call: one nudge, keep the loop on rails.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue solving the question. Use <search> query </search> "
                        "or <python> code </python> when you need facts or computation, "
                        "or output the final answer in <answer> </answer>."
                    ),
                }
            )

        if not stop_reason:
            stop_reason = "max_steps"

        # Post-loop: strict validation of the composed trajectory's answer.
        if stop_reason == "answer":
            ok, reason = validate_format(_cut_after_answer(full_text))
        else:
            ok, reason = (False, stop_reason)

        answer = ""
        if stop_reason == "error":
            trace_error = f"llm transport error: {self.last_llm_error}"
        elif ok:
            answer = extract_tail_answer(full_text)
            trace_error = ""
        else:
            # One self-correction round (also the forced-answer path on budget
            # exhaustion): inject the violation / budget note, ask for a clean
            # single <answer>.
            correction_prompt = ""
            if stop_reason == "answer":
                correction_prompt = (
                    f"Your output violates the required format: {reason}. "
                    "Reply with ONLY the final answer enclosed in exactly one "
                    "<answer> </answer> tag pair. No other tags, no text after it."
                )
            else:
                correction_prompt = (
                    "The step budget is exhausted. Reply with ONLY the final answer "
                    "enclosed in exactly one <answer> </answer> tag pair, based on "
                    "the information gathered so far. No other tags, no text after it."
                )
            answer, corrected, trace_error = self._correction_round(
                messages, full_text, correction_prompt, temperature, max_tokens, deadline
            )

        answer = _unwrap_boxed(answer)
        trace: Dict[str, Any] = {
            "stop": stop_reason,
            "steps": steps,
            "tool": steps[-1]["tool"] if steps else "",
            "n_search": n_search,
            "n_python": n_python,
            "self_corrected": corrected,
            "enabled_tools": enabled_tool_names(),
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "llm_calls": self._drain_llm_calls(),
        }
        if trace_error:
            trace["error"] = trace_error[:500]
        return answer, trace

    def _correction_round(
        self,
        messages: List[Dict[str, Any]],
        full_text: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        deadline: float,
    ) -> Tuple[str, bool, str]:
        """Inject the violation/budget note once; take clean or tolerant answer."""
        try:
            if time.time() >= deadline:
                return extract_tail_answer(full_text), True, "correction skipped: deadline"
            messages.append({"role": "user", "content": prompt})
            try:
                chunk = _llm_chat(
                    list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=90.0,
                )
            except Exception as exc:  # noqa: BLE001
                return extract_tail_answer(full_text), True, f"correction llm error: {exc}"
            chunk = str(chunk or "")
            messages.append({"role": "assistant", "content": chunk})
            combined = full_text + "\n" + chunk
            fixed = extract_tail_answer(combined)
            if fixed:
                chunk_pairs = chunk.count("<answer>") == 1 and chunk.count("</answer>") == 1
                if chunk_pairs:
                    return fixed, True, ""
                return fixed, True, "tolerant extraction: answer tag count != 1"
            return "", True, "correction produced no <answer>"
        except Exception as exc:  # noqa: BLE001
            return extract_tail_answer(full_text), True, f"correction error: {exc}"

    # -- plan window ----------------------------------------------------------

    def plan_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        question = _text(payload.get("question"), payload.get("input"), payload.get("query"))
        if not question and hasattr(message, "dst"):
            question = _text(self.question)
        runtime = _load_runtime()
        try:
            if not question:
                return self._stop_payload(
                    "",
                    "",
                    {"stop": "error", "error": "plan window received no question",
                     "llm_calls": self._drain_llm_calls()},
                )
            if question in self.answer_cache:
                answer, trace = self.answer_cache[question]
                return self._stop_payload(question, answer, trace)
            answer, trace = self._run_loop(question, runtime)
            self.answer_cache[question] = (answer, dict(trace))
            self.question = question
            return self._stop_payload(question, answer, trace)
        except Exception as exc:  # noqa: BLE001 — never break the episode
            return self._stop_payload(
                question,
                "",
                {"stop": "error", "error": f"window internal error: {exc}",
                 "llm_calls": self._drain_llm_calls()},
            )

    # -- defensive mirrors -----------------------------------------------------

    def exec_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        question = _text(payload.get("question"), payload.get("query"), payload.get("input"))
        answer, trace = self.answer_cache.get(question, ("", {}))
        if not answer and question:
            answer, trace = self._run_loop(question, _load_runtime())
            self.answer_cache[question] = (answer, dict(trace))
        wrapped = f"<answer>{answer}</answer>" if answer else ""
        return {
            "kind": "tool_result",
            "payload": {
                "output": wrapped[:2000] or "no answer produced",
                "ok": bool(answer),
                "evidence_type": "DIRECT" if answer else "EMPTY",
                "tool": "SingleAgent",
                "trace": trace,
            },
        }

    def verify_window(self, message: Any) -> Dict[str, Any]:
        answer = _text(getattr(self, "_last_answer", ""))
        ok = bool(answer)
        return {
            "kind": "verify",
            "payload": {
                "ok": ok,
                "reason": "" if ok else "single window produced no answer",
                "step_conclusion": "COMPLETE" if ok else "INCOMPLETE",
                "slot_updates": [],
                "answer": answer,
                "ready_to_stop": True,
                "trace": {},
            },
        }


def _cut_after_answer(text: str) -> str:
    """Cut the composed trajectory at </answer>: nothing after may matter."""
    end = text.rfind("</answer>")
    if end == -1:
        start = text.rfind("<answer>")
        return text[: (start + len("<answer>"))] if start != -1 else text
    return text[: end + len("</answer>")]


def _tool_failed(output: str) -> bool:
    head = str(output or "").lstrip().lower()
    return head.startswith(
        (
            "search_unavailable",
            "wikipedia unreachable",
            "bing unreachable",
            "web fetch failed",
            "web fetch requires",
            "fetch_unavailable",
            "error:",
            "wikipedia search requires",
            "bing search requires",
            "python tool failed",
            "python tool requires",
        )
    )


def get_episode_context() -> EpisodeContext:
    global _CTX
    fingerprint = "|".join(
        [
            os.environ.get("OPENAI_API_BASE_URL") or "",
            os.environ.get("MODEL_Name") or os.environ.get("MODEL_NAME") or "",
        ]
    )
    if _CTX is None or getattr(_CTX, "_endpoint_fp", None) != fingerprint:
        _CTX = EpisodeContext()
        _CTX._endpoint_fp = fingerprint  # type: ignore[attr-defined]
    return _CTX

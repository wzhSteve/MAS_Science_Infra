"""OpenAI-compatible engine that stands in for AgentFlow's ``create_llm_engine``.

Official AgentFlow trains the planner through a vLLM endpoint whose engine
ignores ``response_format`` and returns plain text; every ``extract_*`` method
in ``agentflow.models`` has a string branch for that. This engine does the
same against whatever endpoint the MAS runtime bound (``OPENAI_API_BASE_URL`` /
``MODEL_NAME``), read at call time so one solver serves every rollout.

Roles mirror ``construct_solver(model_engine=[trainable, fixed, fixed, fixed])``:
only the planner's ``llm_engine`` (``generate_next_step``) is trainable. Every
other call runs inside ``frozen_llm_scope`` so the trainer's ``agent_match``
drops it from the actor batch.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.I)
# ChatVLLM._generate_text default (agentflow/engine/vllm.py).
_FREQUENCY_PENALTY = 1.2
_CALL_LOG_CAP = 64

try:
    from workflow.user_gateway.public import frozen_llm_scope
except Exception:  # noqa: BLE001
    def frozen_llm_scope(role: str) -> Any:  # type: ignore[misc]
        return nullcontext()


def _note_active_http(client: Any) -> None:
    """Let a window timeout close this socket. Missing the helper is not fatal."""
    try:
        from workflow.user_gateway.sandbox import note_active_http
    except Exception:  # noqa: BLE001
        return
    note_active_http(client)


def build_chat_client(http_client: Any, *, base_url: str, api_key: str, timeout: float) -> Any:
    """One attempt. The SDK default retries a 90s read until the window is full."""
    from openai import OpenAI

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
        timeout=timeout,
        max_retries=0,
    )


_LOG_LOCK = threading.Lock()
_CALL_LOG: Dict[str, List[Dict[str, Any]]] = {}


def _choice_entropy(choice: Any) -> Tuple[Optional[float], List[float]]:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None, []
    tokens: List[Dict[str, Any]] = []
    for item in content:
        if hasattr(item, "model_dump"):
            tokens.append(item.model_dump())
        elif isinstance(item, dict):
            tokens.append(item)
    try:
        from workflow.token_entropy import entropy_from_choice_logprobs, token_entropies_from_choice
    except Exception:
        return None, []
    per_token = token_entropies_from_choice(tokens)
    return entropy_from_choice_logprobs(tokens), per_token


def record_call(key: str, entry: Dict[str, Any]) -> None:
    with _LOG_LOCK:
        rows = _CALL_LOG.setdefault(key, [])
        rows.append(entry)
        del rows[: max(0, len(rows) - _CALL_LOG_CAP)]


def drain_calls(key: str) -> List[Dict[str, Any]]:
    with _LOG_LOCK:
        return _CALL_LOG.pop(key, [])


def _rollout_binding() -> Dict[str, str]:
    try:
        from workflow.llm_scope import current_rollout_llm
    except Exception:
        return {}
    return current_rollout_llm()


def _endpoint(frozen: bool, pinned: Optional[Dict[str, str]] = None) -> str:
    pin = (pinned or {}).get("endpoint") or ""
    if pin:
        return str(pin).rstrip("/")
    bound = _rollout_binding().get("endpoint") or ""
    if bound:
        return bound.rstrip("/")
    if frozen and os.environ.get("AGENTFLOW_FIXED_BASE_URL"):
        return os.environ["AGENTFLOW_FIXED_BASE_URL"].rstrip("/")
    return (
        os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    ).rstrip("/")


def _model(frozen: bool, pinned: Optional[Dict[str, str]] = None) -> str:
    pin = (pinned or {}).get("model") or ""
    if pin:
        return str(pin)
    bound = _rollout_binding().get("model") or ""
    if bound:
        return bound
    if frozen and os.environ.get("AGENTFLOW_FIXED_MODEL"):
        return os.environ["AGENTFLOW_FIXED_MODEL"]
    return (
        os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL_Name")
        or os.environ.get("OPENAI_MODEL")
        or "Qwen3-1.7B"
    )


def _api_key(pinned: Optional[Dict[str, str]] = None) -> str:
    pin = (pinned or {}).get("api_key") or ""
    if pin:
        return str(pin)
    bound = _rollout_binding().get("api_key") or ""
    if bound:
        return bound
    return os.environ.get("OPENAI_API_KEY") or "EMPTY"


def _no_think() -> bool:
    """Local Qwen switch. Official prompts are unchanged unless this is on."""
    return os.environ.get("AGENTFLOW_NO_THINK", "").strip().lower() in ("1", "true", "yes", "on")


_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")


def thinking_enabled() -> bool:
    """``.env`` key ``enable_thinking`` controls every role.

    Unset means off. ``AGENTFLOW_NO_THINK`` still forces off when the key is absent.
    """
    raw = os.environ.get("enable_thinking")
    if raw is None or not str(raw).strip():
        raw = os.environ.get("ENABLE_THINKING")
    text = str(raw or "").strip().lower()
    if text in _ON:
        return True
    if text in _OFF:
        return False
    if _no_think():
        return False
    return False


_TOKENIZER: Any = None


def _planner_tokenizer() -> Any:
    """Qwen tokenizer used only to count the prediction-label tokens."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    raw = os.environ.get("VLLM_MODEL_PATH", "").strip()
    path = Path(raw) if raw else Path("/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B")
    if not path.is_dir():
        return None
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    return _TOKENIZER


def skip_thinking(trainable: bool) -> bool:
    """True when this call must send ``enable_thinking=False``.

    The role does not matter. The ``.env`` switch covers the planner and the frozen tools.
    """
    del trainable
    return not thinking_enabled()


_CONTEXT_MARGIN = 128
_PROMPT_SLACK = 192
_TRUNC_MARK = "...[truncated]..."


def _serving_model_len(override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    raw = int(os.environ.get("TIR_MAX_MODEL_LEN") or 0)
    return raw or 8192


def _conservative_tokens(text: str) -> int:
    """Non-ASCII is one token each. ASCII is one token per two characters."""
    ascii_chars = 0
    other = 0
    for ch in text or "":
        if ord(ch) < 128:
            ascii_chars += 1
        else:
            other += 1
    return other + (ascii_chars + 1) // 2


def count_chat_tokens(system_prompt: str, user: str) -> int:
    """Token count for the chat the server will see, plus no char/3 estimate."""
    try:
        tok = _load_tokenizer()
        ids = tok.apply_chat_template(
            [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user or ""},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
        return max(1, len(list(ids)))
    except Exception:
        return max(1, _conservative_tokens(system_prompt) + _conservative_tokens(user) + 32)


def truncate_user_middle(user: str, head: int = 1200, tail: int = 1200) -> str:
    """Keep the question at the start and the rules at the end."""
    text = str(user or "")
    head = max(1, int(head))
    tail = max(1, int(tail))
    if len(text) <= head + tail:
        return text
    return text[:head] + _TRUNC_MARK + text[-tail:]


def fit_user_to_context(
    user: str,
    *,
    system_prompt: str = "",
    model_len: Optional[int] = None,
    counter: Optional[Callable[[str], int]] = None,
) -> str:
    """Drop the middle of the user prompt when it exceeds the context slack."""
    limit = _serving_model_len(model_len)
    budget = limit - _PROMPT_SLACK
    current = str(user or "")
    head, tail = 1200, 1200

    def _tokens(text: str) -> int:
        if counter is not None:
            return int(counter(system_prompt + text))
        return count_chat_tokens(system_prompt, text)

    for _ in range(8):
        if _tokens(current) <= budget:
            return current
        nxt = truncate_user_middle(current, head, tail)
        if nxt == current:
            head = max(200, head // 2)
            tail = max(200, tail // 2)
            nxt = truncate_user_middle(current, head, tail)
            if nxt == current:
                return current
        current = nxt
        head = max(200, (head * 2) // 3)
        tail = max(200, (tail * 2) // 3)
    return current


def clamp_max_tokens(
    requested: int,
    prompt: str,
    cap: int,
    *,
    counter: Optional[Callable[[str], int]] = None,
    model_len: Optional[int] = None,
) -> int:
    """Fit the completion inside the serving context (``TIR_MAX_MODEL_LEN``)."""
    limit = _serving_model_len(model_len)
    if counter is not None:
        prompt_tokens = max(1, int(counter(prompt)))
    else:
        prompt_tokens = max(1, _conservative_tokens(prompt) + 32)
    remaining = limit - prompt_tokens - _CONTEXT_MARGIN
    return max(64, min(int(requested), int(cap), remaining))


def _is_context_error(exc: BaseException) -> bool:
    parts = [str(exc).lower()]
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(str(getattr(response, "text", "") or "").lower())
        except Exception:
            pass
    blob = " ".join(parts)
    return any(
        s in blob
        for s in (
            "context length",
            "maximum context",
            "context window",
            "contextwindow",
            "please reduce the length",
        )
    )


_TOKENIZER: Any = None
_TOKENIZER_LOCK = threading.Lock()
_REPO = Path(__file__).resolve().parents[4]


def tokenizer_path() -> str:
    """HF tokenizer matching the served Qwen weights. Override with AGENTFLOW_TOKENIZER."""
    env = os.environ.get("AGENTFLOW_TOKENIZER", "").strip()
    if env:
        return env
    name = (
        os.environ.get("AGENTFLOW_FIXED_MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or "Qwen3-4B"
    ).strip()
    named = _REPO / "LLM" / name
    if named.is_dir():
        return str(named)
    fallback = _REPO / "LLM" / "Qwen3-4B"
    return str(fallback if fallback.is_dir() else named)


def _load_tokenizer() -> Any:
    global _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path(), trust_remote_code=True)
        return _TOKENIZER


def _as_id_list(value: Any) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(x) for x in value]


def encode_chat_ids(messages: List[Dict[str, Any]], completion: str) -> Tuple[List[int], List[int]]:
    """Token ids for a chat completion when the server does not return them.

    vLLM 0.9.2 ignores ``return_token_ids``. agentops only keeps ids that are
    already on the parsed response, so the HTTP body is filled before parse.
    """
    tok = _load_tokenizer()
    prompt_ids = _as_id_list(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    text = completion or ""
    resp_ids = _as_id_list(tok.encode(text, add_special_tokens=False)) if text else []
    if not resp_ids and getattr(tok, "eos_token_id", None) is not None:
        resp_ids = [int(tok.eos_token_id)]
    return prompt_ids, resp_ids


def attach_token_ids(request_body: bytes, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy ids onto the chat JSON when the backend omitted them."""
    if not isinstance(payload, dict) or payload.get("object") != "chat.completion":
        return payload
    existing = payload.get("prompt_token_ids")
    if isinstance(existing, list) and existing:
        return payload
    try:
        req = json.loads(request_body.decode("utf-8"))
    except Exception:
        return payload
    messages = req.get("messages") if isinstance(req, dict) else None
    if not isinstance(messages, list) or not messages:
        return payload
    choices = payload.get("choices")
    content = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")
    try:
        prompt_ids, resp_ids = encode_chat_ids(messages, content)
    except Exception:
        return payload
    if not prompt_ids or not resp_ids:
        return payload
    out = dict(payload)
    out["prompt_token_ids"] = prompt_ids
    out["response_token_ids"] = [resp_ids]
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = dict(choices[0])
        choice["token_ids"] = resp_ids
        out["choices"] = [choice, *choices[1:]]
    return out


def _install_token_hook(client: Any) -> None:
    """Mutate chat JSON before the OpenAI client parses it."""

    def _hook(response: Any) -> None:
        if response.status_code >= 400:
            return
        try:
            response.read()
            raw = response.content
            payload = json.loads(raw)
        except Exception:
            return
        updated = attach_token_ids(response.request.content or b"", payload)
        if updated is payload:
            return
        response._content = json.dumps(updated).encode("utf-8")
        response.headers["content-length"] = str(len(response._content))

    hooks = client.event_hooks.setdefault("response", [])
    hooks.append(_hook)


class LocalChatEngine:
    """Callable with the ``EngineLM`` surface AgentFlow models use."""

    def __init__(
        self,
        role: str,
        *,
        trainable: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.99,
        max_tokens_cap: int = 4096,
        log_key: Any = None,
    ) -> None:
        self.role = role
        self.trainable = trainable
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens_cap = int(max_tokens_cap)
        self._pinned: Dict[str, str] = {}
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.is_multimodal = False
        self._log_key = log_key or (lambda: "")
        self.last_label_probs: Dict[str, Any] = {}

    def pin_endpoint(self, bound: Dict[str, str]) -> None:
        """Remember the rollout endpoint on this instance.

        Tool execution runs in a fresh thread, and that thread does not see
        the request's context variable. The pin is set on the calling thread
        before ``execute_tool_command`` starts.
        """
        self._pinned = {
            "endpoint": str(bound.get("endpoint") or "").rstrip("/"),
            "model": str(bound.get("model") or ""),
            "api_key": str(bound.get("api_key") or "EMPTY"),
        }

    def _pinned_value(self, key: str) -> str:
        return str((self._pinned or {}).get(key) or "")

    @property
    def model_string(self) -> str:
        return _model(not self.trainable, self._pinned)

    def __call__(self, content: Any, **kwargs: Any) -> str:
        return self.generate(content, **kwargs)

    def generate(
        self,
        content: Any,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        response_format: Any = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(content, (list, tuple)):
            prompt = "\n".join(str(x) for x in content if isinstance(x, str))
        else:
            prompt = str(content)
        scope = nullcontext() if self.trainable else frozen_llm_scope(self.role)
        with scope:
            return self._complete(prompt, system_prompt or self.system_prompt, max_tokens)

    def _echo_token_logprobs(self, model: str, frozen: bool, prompt: str) -> List[Any]:
        """Prompt-token logprobs for ``prompt``. ``max_tokens=0`` adds no new tokens."""
        import httpx

        llm_timeout = float(os.environ.get("AGENTFLOW_LLM_TIMEOUT") or 90)
        with httpx.Client(trust_env=False, timeout=llm_timeout) as http_client:
            _note_active_http(http_client)
            client = build_chat_client(
                http_client,
                base_url=_endpoint(frozen, self._pinned),
                api_key=_api_key(self._pinned),
                timeout=llm_timeout,
            )
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0,
            )
        choice = resp.choices[0]
        payload = getattr(choice, "logprobs", None)
        values = getattr(payload, "token_logprobs", None)
        if values is None and isinstance(payload, dict):
            values = payload.get("token_logprobs")
        return list(values or [])

    def _score_from_choice_logprobs(self, choice: Any, completion: str) -> Dict[str, Any]:
        """Score 满足/不满足 from the generation's top-logprobs, without a second prefill."""
        from prediction_probs import SATISFY, UNSATISFY, prediction_record, written_prediction

        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        if not content:
            return {}
        pieces: List[str] = []
        items: List[Any] = []
        for item in content:
            token = getattr(item, "token", None)
            if token is None and isinstance(item, dict):
                token = item.get("token")
            pieces.append(str(token or "").replace("Ġ", " ").replace("▁", " "))
            items.append(item)
        acc = "".join(pieces)
        match = re.search(r'"prediction"\s*:\s*"', acc)
        if not match:
            match = re.search(r'"prediction"\s*:\s*"', str(completion or ""))
            if not match:
                return {}
            # Generation logprobs still include <think> tokens; locate the quote in pieces.
            match = re.search(r'"prediction"\s*:\s*"', acc)
            if not match:
                return {}
        value_start = match.end()
        pos = 0
        start = None
        for i, piece in enumerate(pieces):
            nxt = pos + len(piece)
            if nxt > value_start:
                start = i if pos >= value_start else i
                break
            pos = nxt
        if start is None or start >= len(items):
            return {}

        def _top_map(item: Any) -> Dict[str, float]:
            out: Dict[str, float] = {}
            lp = getattr(item, "logprob", None)
            token = getattr(item, "token", None)
            if isinstance(item, dict):
                lp = item.get("logprob", lp)
                token = item.get("token", token)
            if token is not None and lp is not None:
                out[str(token)] = float(lp)
            tops = getattr(item, "top_logprobs", None)
            if tops is None and isinstance(item, dict):
                tops = item.get("top_logprobs")
            for top in tops or []:
                top_token = getattr(top, "token", None)
                top_lp = getattr(top, "logprob", None)
                if isinstance(top, dict):
                    top_token = top.get("token", top_token)
                    top_lp = top.get("logprob", top_lp)
                if top_token is not None and top_lp is not None:
                    out[str(top_token)] = float(top_lp)
            return out

        def _pick(dist: Dict[str, float], *targets: str) -> Optional[float]:
            for target in targets:
                if target in dist:
                    return dist[target]
                for key, value in dist.items():
                    if target and target in key:
                        return value
            return None

        first = _top_map(items[start])
        s_satisfy = _pick(first, SATISFY)
        un_first = _pick(first, "不", UNSATISFY)
        s_unsatisfy = None
        if un_first is not None and start + 1 < len(items):
            second = _top_map(items[start + 1])
            rest = _pick(second, "满足")
            if rest is not None:
                s_unsatisfy = float(un_first) + float(rest)
            elif _pick(first, UNSATISFY) is not None:
                s_unsatisfy = _pick(first, UNSATISFY)
        if s_unsatisfy is None:
            s_unsatisfy = _pick(first, UNSATISFY)
        if s_satisfy is None or s_unsatisfy is None:
            return {}
        record = prediction_record(written_prediction(completion), float(s_satisfy), float(s_unsatisfy))
        tok = _planner_tokenizer()
        if tok is not None:
            record["satisfy_token_ids"] = list(tok.encode(SATISFY, add_special_tokens=False))
            record["unsatisfy_token_ids"] = list(tok.encode(UNSATISFY, add_special_tokens=False))
            record["prediction_token_start"] = 0
        return record

    def _score_prediction_labels(
        self,
        system_prompt: str,
        user_text: str,
        completion: str,
        model: str,
        frozen: bool,
        choice: Any = None,
    ) -> None:
        """Score 满足 and 不满足 at the prediction quote. Failures leave probs empty."""
        self.last_label_probs = {}
        if self.role != "planner":
            return
        if choice is not None:
            try:
                scored = self._score_from_choice_logprobs(choice, completion)
            except Exception:
                scored = {}
            if scored:
                self.last_label_probs = scored
                return
        try:
            from prediction_probs import (
                SATISFY,
                UNSATISFY,
                prediction_prefix,
                prediction_record,
                sum_suffix_logprobs,
                written_prediction,
            )
        except Exception:
            return
        prefix = prediction_prefix(completion)
        if not prefix:
            return
        try:
            tok = _planner_tokenizer()
            if tok is None:
                return
            base = tok.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not skip_thinking(self.trainable),
            )
            stem = str(base) + prefix
            stem_ids = tok.encode(stem, add_special_tokens=False)
            scores: Dict[str, float] = {}
            candidate_token_ids: Dict[str, List[int]] = {}
            for label in (SATISFY, UNSATISFY):
                full = stem + label
                full_ids = tok.encode(full, add_special_tokens=False)
                if full_ids[: len(stem_ids)] == stem_ids:
                    n_prefix = len(stem_ids)
                    n_label = len(full_ids) - n_prefix
                else:
                    label_ids = tok.encode(label, add_special_tokens=False)
                    n_label = len(label_ids)
                    n_prefix = max(0, len(full_ids) - n_label)
                candidate_token_ids[label] = list(full_ids[n_prefix:])
                echoed = self._echo_token_logprobs(model, frozen, full)
                scores[label] = sum_suffix_logprobs(echoed, n_prefix, n_label)
            self.last_label_probs = prediction_record(
                written_prediction(completion),
                scores[SATISFY],
                scores[UNSATISFY],
            )
            self.last_label_probs["satisfy_token_ids"] = candidate_token_ids[SATISFY]
            self.last_label_probs["unsatisfy_token_ids"] = candidate_token_ids[UNSATISFY]
            self.last_label_probs["prediction_token_start"] = len(
                tok.encode(prefix, add_special_tokens=False)
            )
        except Exception:
            self.last_label_probs = {}

    def _complete(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        import httpx

        frozen = not self.trainable
        model = _model(frozen, self._pinned)
        # enable_thinking in .env covers every role, including the planner.
        quiet = skip_thinking(self.trainable)
        user = prompt.rstrip() + "\n/no_think" if quiet else prompt
        model_len = _serving_model_len()
        user = fit_user_to_context(user, system_prompt=system_prompt, model_len=model_len)
        # trust_env=False: an ambient HTTP proxy 502s the trainable rollout URL
        # (172.17/10.x) and the planner window never reaches exec.
        self.last_label_probs = {}
        entry: Dict[str, Any] = {"role": self.role, "model": model, "prompt_head": " ".join(prompt.split())[:300]}

        def _send(user_text: str) -> str:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": _FREQUENCY_PENALTY,
                "presence_penalty": 0,
                "max_tokens": clamp_max_tokens(
                    max_tokens,
                    user_text,
                    self.max_tokens_cap,
                    model_len=model_len,
                    counter=lambda _prompt: count_chat_tokens(system_prompt, user_text),
                ),
            }
            if self.trainable or self.role == "planner":
                # ARPO reads entropy; FCR reads 满足/不满足 from the same top-10.
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 10
            enabled = not quiet
            kwargs["extra_body"] = {
                "enable_thinking": enabled,
                "chat_template_kwargs": {"enable_thinking": enabled},
            }
            # trust_env=False: the ambient HTTP proxy 502s the rollout URL.
            # The response hook fills prompt/response token ids. vLLM 0.9.2
            # does not honor return_token_ids; agentops reads them off the
            # parsed body of with_raw_response.create().
            # max_retries=0: the SDK default of 2 retries stacked three 90s
            # waits into the 300s window and left the last request on the GPU.
            llm_timeout = float(os.environ.get("AGENTFLOW_LLM_TIMEOUT") or 90)
            with httpx.Client(trust_env=False, timeout=llm_timeout) as http_client:
                _note_active_http(http_client)
                client = build_chat_client(
                    http_client,
                    base_url=_endpoint(frozen, self._pinned),
                    api_key=_api_key(self._pinned),
                    timeout=llm_timeout,
                )
                resp = client.chat.completions.with_raw_response.create(**kwargs).parse()
            text = _THINK_RE.sub("", str(resp.choices[0].message.content or "")).strip()
            entry["response_head"] = " ".join(text.split())[:300]
            self._score_prediction_labels(
                system_prompt, user_text, text, model, frozen, choice=resp.choices[0]
            )
            if self.last_label_probs:
                entry["prediction_probs"] = dict(self.last_label_probs)
            if self.trainable:
                entropy, per_token = _choice_entropy(resp.choices[0])
                if entropy is not None:
                    entry["entropy"] = entropy
                if per_token:
                    entry["token_entropies"] = per_token
            return text

        try:
            try:
                return _send(user)
            except Exception as exc:
                if not _is_context_error(exc):
                    raise
                shorter = truncate_user_middle(user, 600, 600)
                if shorter == user:
                    shorter = truncate_user_middle(user, 200, 200)
                return _send(shorter)
        except Exception as exc:
            detail = ""
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = " " + (response.text or "")[:180]
                except Exception:
                    detail = ""
            entry["response_head"] = f"{type(exc).__name__}: {exc}{detail}"[:300]
            raise
        finally:
            record_call(str(self._log_key()), entry)

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
from typing import Any, Dict, List, Optional, Tuple

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


_LOG_LOCK = threading.Lock()
_CALL_LOG: Dict[str, List[Dict[str, Any]]] = {}


def record_call(key: str, entry: Dict[str, Any]) -> None:
    with _LOG_LOCK:
        rows = _CALL_LOG.setdefault(key, [])
        rows.append(entry)
        del rows[: max(0, len(rows) - _CALL_LOG_CAP)]


def drain_calls(key: str) -> List[Dict[str, Any]]:
    with _LOG_LOCK:
        return _CALL_LOG.pop(key, [])


def _endpoint(frozen: bool) -> str:
    if frozen and os.environ.get("AGENTFLOW_FIXED_BASE_URL"):
        return os.environ["AGENTFLOW_FIXED_BASE_URL"].rstrip("/")
    return (
        os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    ).rstrip("/")


def _model(frozen: bool) -> str:
    if frozen and os.environ.get("AGENTFLOW_FIXED_MODEL"):
        return os.environ["AGENTFLOW_FIXED_MODEL"]
    return (
        os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL_Name")
        or os.environ.get("OPENAI_MODEL")
        or "Qwen3-1.7B"
    )


def _no_think() -> bool:
    """Local Qwen switch. Official prompts are unchanged unless this is on."""
    return os.environ.get("AGENTFLOW_NO_THINK", "").strip().lower() in ("1", "true", "yes", "on")


def clamp_max_tokens(requested: int, prompt: str, cap: int) -> int:
    """Fit the completion inside the serving context (``TIR_MAX_MODEL_LEN``)."""
    model_len = int(os.environ.get("TIR_MAX_MODEL_LEN") or 0) or 8192
    prompt_tokens = max(1, (len(prompt) + 3) // 3)
    remaining = max(64, model_len - prompt_tokens - 64)
    return max(64, min(int(requested), int(cap), remaining))


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
        max_tokens_cap: int = 2048,
        log_key: Any = None,
    ) -> None:
        self.role = role
        self.trainable = trainable
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens_cap = int(max_tokens_cap)
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.is_multimodal = False
        self._log_key = log_key or (lambda: "")

    @property
    def model_string(self) -> str:
        return _model(not self.trainable)

    def __call__(self, content: Any, **kwargs: Any) -> str:
        return self.generate(content, **kwargs)

    def generate(
        self,
        content: Any,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
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

    def _complete(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        import httpx
        from openai import OpenAI

        frozen = not self.trainable
        model = _model(frozen)
        # /no_think is a local switch. Default off so official f-strings stay intact.
        user = prompt.rstrip() + "\n/no_think" if _no_think() else prompt
        # trust_env=False: an ambient HTTP proxy 502s the trainable rollout URL
        # (172.17/10.x) and the planner window never reaches exec.
        entry: Dict[str, Any] = {"role": self.role, "model": model, "prompt_head": " ".join(prompt.split())[:300]}
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": _FREQUENCY_PENALTY,
            "presence_penalty": 0,
            "max_tokens": clamp_max_tokens(max_tokens, system_prompt + user, self.max_tokens_cap),
        }
        if _no_think():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            # trust_env=False: the ambient HTTP proxy 502s the rollout URL.
            # The response hook fills prompt/response token ids. vLLM 0.9.2
            # does not honor return_token_ids; agentops reads them off the
            # parsed body of with_raw_response.create().
            with httpx.Client(trust_env=False) as http_client:
                _install_token_hook(http_client)
                client = OpenAI(
                    base_url=_endpoint(frozen),
                    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    http_client=http_client,
                )
                resp = client.chat.completions.with_raw_response.create(**kwargs).parse()
            text = _THINK_RE.sub("", str(resp.choices[0].message.content or "")).strip()
            entry["response_head"] = " ".join(text.split())[:300]
            return text
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

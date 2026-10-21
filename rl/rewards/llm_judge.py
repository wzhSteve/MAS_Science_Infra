"""Flow-GRPO-style LLM judge (0/1) via EVALUATION_MODEL_* OpenAI-compatible API.

Prompt and verdict semantics mirror AgentFlow ``train/utils.py`` / ``train/rollout.py``.
Failures (missing creds, timeout, parse error) log and return 0.0 so training continues.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_LOG = logging.getLogger(__name__)

_TRUE_FALSE_RE = re.compile(
    r"<true_false>\s*(true|false)\s*</true_false>",
    re.IGNORECASE,
)
_TRUE_FALSE_LOOSE_RE = re.compile(
    r"true_false\s*[:\-=]?\s*(true|false)",
    re.IGNORECASE,
)

_CACHE_LOCK = threading.Lock()
_MEMORY_CACHE: Dict[str, float] = {}
_CLIENT = None
_CLIENT_KEY: Optional[Tuple[str, str, str]] = None


def _load_dotenv() -> None:
    """Best-effort .env load without importing mas (rl must stay free of workflow)."""
    try:
        from science_infra.env import load_science_env

        load_science_env()
        return
    except Exception:
        pass
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if not k or k in os.environ:
                continue
            v = v.strip().strip('"').strip("'")
            os.environ[k] = v
    except OSError:
        pass


def resolve_evaluation_endpoint() -> Tuple[str, str, str]:
    """Return (api_key, base_url, model). Never log the key."""
    _load_dotenv()
    key = (
        os.environ.get("EVALUATION_MODEL_API_KEY")
        or os.environ.get("EVALUATION_API_KEY")
        or ""
    ).strip()
    base = (
        os.environ.get("EVALUATION_MODEL_API_BASE")
        or os.environ.get("EVALUATION_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE_URL")
        or ""
    ).strip().rstrip("/")
    model = (
        os.environ.get("EVALUATION_MODEL")
        or os.environ.get("EVALUATION_MODEL_NAME")
        or "gpt-4o"
    ).strip()
    return key, base, model


def build_judge_prompt(question: str, groundtruth: str, answer_extracted: str) -> str:
    """Judge prompt. The format example is VERDICT so a copied template is not a yes."""
    return f"""
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1. The Model Response is already the extracted answer. Do not treat reasoning as the answer.
2. Equivalent after normalization means the same value, ignoring case, spacing, and commas.
3. A formula that was not evaluated, a placeholder, or a partial name is not a match.
4. A single option letter matches only when that option's text is the Ground Truth.

**Inputs:**
Question: {question}
Model Response: {answer_extracted}
Ground Truth: {groundtruth}

**Format:** output exactly one verdict tag, then one short analysis. Do not write both an affirmative and a negative verdict.
<true_false>VERDICT</true_false>
<analysis>One short sentence.</analysis>
Replace VERDICT with true when they match. Otherwise replace VERDICT with false. Output only one verdict tag.
"""


def parse_true_false(text: str) -> Optional[bool]:
    if not text:
        return None
    # A copied template that lists both verdicts is not a decision.
    words = {item.lower() for item in re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)}
    if words == {"true", "false"}:
        return None
    m = _TRUE_FALSE_RE.search(text)
    if m:
        return m.group(1).lower() == "true"
    m = _TRUE_FALSE_LOOSE_RE.search(text)
    if m:
        return m.group(1).lower() == "true"
    # bare tag without closing (truncated): <true_false>True
    m = re.search(r"<true_false>\s*(true|false)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    # last-resort: trailing True/False token
    m = re.search(r"\b(True|False)\b\s*$", text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def _cache_dir() -> Optional[Path]:
    raw = os.environ.get("EVALUATION_JUDGE_CACHE_DIR", "").strip()
    if raw.lower() in ("0", "false", "off", "none"):
        return None
    if not raw:
        raw = "/tmp/mas_eval_judge_cache"
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
    return h


def _cache_get(key: str) -> Optional[float]:
    with _CACHE_LOCK:
        if key in _MEMORY_CACHE:
            return _MEMORY_CACHE[key]
    cdir = _cache_dir()
    if cdir is None:
        return None
    fp = cdir / f"{key}.json"
    if not fp.is_file():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        score = float(data["score"])
        with _CACHE_LOCK:
            _MEMORY_CACHE[key] = score
        return score
    except Exception:
        return None


def _cache_put(key: str, score: float) -> None:
    with _CACHE_LOCK:
        _MEMORY_CACHE[key] = score
    cdir = _cache_dir()
    if cdir is None:
        return
    fp = cdir / f"{key}.json"
    try:
        fp.write_text(json.dumps({"score": score}), encoding="utf-8")
    except OSError:
        pass


def _get_client(api_key: str, base_url: str):
    global _CLIENT, _CLIENT_KEY
    ck = (api_key, base_url)
    if _CLIENT is not None and _CLIENT_KEY == ck:
        return _CLIENT
    from openai import OpenAI

    timeout = float(os.environ.get("EVALUATION_JUDGE_TIMEOUT", "60") or 60)
    _CLIENT = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None, timeout=timeout)
    _CLIENT_KEY = ck
    return _CLIENT


def llm_judge_correct(
    question: str,
    groundtruth: str,
    answer_extracted: str,
    *,
    model: Optional[str] = None,
) -> float:
    """Return 1.0 if the judge says True, else 0.0 (including on errors)."""
    q = str(question or "")
    g = str(groundtruth or "")
    a = str(answer_extracted or "")
    if not a.strip():
        return 0.0

    api_key, base_url, env_model = resolve_evaluation_endpoint()
    use_model = (model or env_model or "gpt-4o").strip()
    if not base_url:
        _LOG.warning("llm_judge: EVALUATION_MODEL_API_BASE unset; score=0")
        return 0.0

    prompt = build_judge_prompt(q, g, a)
    ck = _cache_key(use_model, prompt)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        client = _get_client(api_key, base_url)
        resp = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        content = ""
        if resp.choices:
            content = str(resp.choices[0].message.content or "")
        verdict = parse_true_false(content)
        if verdict is None:
            _LOG.warning("llm_judge: unparseable verdict from %s; score=0", use_model)
            score = 0.0
        else:
            score = 1.0 if verdict else 0.0
        _cache_put(ck, score)
        return score
    except Exception as exc:
        _LOG.warning("llm_judge: call failed (%s); score=0", type(exc).__name__)
        return 0.0


def probe_evaluation_model(candidate: str) -> Dict[str, Any]:
    """One Flow-style smoke call; returns {ok, model, score, error}."""
    api_key, base_url, _ = resolve_evaluation_endpoint()
    out: Dict[str, Any] = {"ok": False, "model": candidate, "score": None, "error": None}
    if not base_url:
        out["error"] = "EVALUATION_MODEL_API_BASE unset"
        return out
    question = "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    groundtruth = "C"
    answer = "The correct answer is C."
    try:
        # bypass disk cache for probe uniqueness
        prompt = build_judge_prompt(question, groundtruth, answer)
        client = _get_client(api_key, base_url)
        resp = client.chat.completions.create(
            model=candidate,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        content = str(resp.choices[0].message.content or "") if resp.choices else ""
        verdict = parse_true_false(content)
        if verdict is None:
            out["error"] = f"unparseable: {content[:200]!r}"
            return out
        out["ok"] = True
        out["score"] = 1.0 if verdict else 0.0
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

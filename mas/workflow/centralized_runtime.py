"""Centralized planner-led MAS runtime (phase 5).

Walks the compiled graph with explicit ``AgentMessage`` envelopes:
planner -> router -> tool-agent(s) -> verifier, one message per edge. Each
window close appends a MAS-layer blackboard log entry. Multi-router fan-out
runs the selected tool-agents in parallel. Termination prefers
``ready_to_stop`` / ``payload.answer``; built-in graphs still stop on
``plan_step.done=true`` AND ``verify.ok=true``.

See docs/MAS_AGENT_LANDING.md §5.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Protocol

from .contracts import EventKind, ExecutionEvent
from .protocol import (
    DEFAULT_INPUT_SCHEMA,
    DEFAULT_OUTPUT_SCHEMA,
    AgentMessage,
    commit_to_fact,
    make_message,
    validate_json_schema,
    validate_payload,
)
from .router import select as router_select


class WindowLLM(Protocol):
    """One LLM call per agent window. Returns raw text; the caller parses JSON."""

    def invoke(self, system_prompt: str, user_text: str, *, agent_id: str, kind: str) -> str: ...


class LocalWindowLLM:
    """WindowLLM backed by an OpenAI-compatible endpoint (local vLLM by default)."""

    def __init__(self, llm: Optional[Any]) -> None:
        self.llm = llm
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return self._client
        if not self.llm or not getattr(self.llm, "endpoint", ""):
            raise RuntimeError("LocalWindowLLM has no endpoint (start scripts/serve_local_llm.sh)")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"openai package unavailable: {e}")
        key = (getattr(self.llm, "api_key", None) or "EMPTY") or "EMPTY"
        self._client = OpenAI(base_url=self.llm.endpoint, api_key=key)
        return self._client

    def invoke(self, system_prompt: str, user_text: str, *, agent_id: str, kind: str) -> str:
        client = self._ensure()
        sys = system_prompt or ""
        if "JSON" not in sys and "json" not in sys:
            sys = (sys + "\n" if sys else "") + "Reply with a single JSON object only. No markdown, no thinking."
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": user_text})
        resp = client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            temperature=getattr(self.llm, "temperature", 0.3) or 0.3,
            max_tokens=getattr(self.llm, "max_tokens", 1024) or 1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return str(resp.choices[0].message.content or "")


class MockWindowLLM:
    """Scripted WindowLLM for tests. ``responses`` maps agent_id -> list of
    JSON strings consumed in order (planner/verifier re-used across turns)."""

    def __init__(self, responses: Mapping[str, List[str]], **_kwargs):
        self._responses = {k: list(v) for k, v in responses.items()}
        self._cursors: Dict[str, int] = {}

    def invoke(self, system_prompt: str, user_text: str, *, agent_id: str, kind: str) -> str:
        seq = self._responses.get(agent_id) or self._responses.get(kind) or []
        i = self._cursors.get(agent_id, 0)
        if i >= len(seq):
            # default: planner keeps done=True, verifier keeps ok=True
            if agent_id == "planner" or kind == "planner":
                return json.dumps({"next": "", "args": {}, "sub_goal": "done", "done": True, "answer": "mock"})
            if agent_id == "verifier" or kind == "verifier":
                return json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})
            return json.dumps({"output": "mock", "ok": True, "evidence_type": "DIRECT"})
        self._cursors[agent_id] = i + 1
        return seq[i]


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Qwen3 thinking traces; keep only the post-think payload.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else text
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        blob = blob[start : end + 1]
    try:
        v = json.loads(blob)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _emit_window(
    window_events: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    *,
    agent_id: str,
    kind: str,
    turn: int,
    metrics: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> None:
    event: Dict[str, Any] = {
        "agent_id": agent_id,
        "kind": kind,
        "turn": turn,
        "metrics": dict(metrics or {}),
        "snapshot_ref": f"{agent_id}:{turn}:{kind}",
        "event_id": f"{agent_id}:{turn}:{kind}",
        "messages": list(messages),
    }
    event.update(extra)
    window_events.append(event)


def _log_window(archive, agent_id: str, turn: int, in_msg: AgentMessage, out_msg: AgentMessage, ok: bool) -> None:
    archive.append(
        ExecutionEvent(
            kind=EventKind.AGENT_MESSAGE,
            agent_id=agent_id,
            payload={
                "phase": "window_close",
                "agent_id": agent_id,
                "turn": turn,
                "input_msg_id": in_msg.msg_id,
                "output_msg_id": out_msg.msg_id,
                "ok": ok,
                "payload_digest": out_msg.payload_digest(),
            },
        )
    )


def run_centralized_episode(
    task: Dict[str, Any],
    llm: Optional[Any],
    archive,
    spec,
    memory,
    compiled,
    *,
    agent_llms: Optional[Mapping[str, Any]] = None,
    window_llm: Optional[WindowLLM] = None,
):
    """Centralized planner-led walk with explicit AgentMessage envelopes."""
    from .runtime import EpisodeRaw, LLMConfig  # local to avoid cycle
    from tools.tool_agents import effective_tier, get_tool_agent, validate_tool_args

    task_id = str(task.get("id") or task.get("_rollout_id") or "task")
    question = str(task.get("question") or "")
    win = window_llm or LocalWindowLLM(llm)

    planner_node = compiled.agents.get("planner")
    if planner_node is None:
        return EpisodeRaw(messages=[], error="centralized topology requires a planner node")
    planner_prompt = (
        planner_node.system_prompt or spec.hub.system_prompt
        or "Decompose the task into the next step and pick one tool. Output JSON only."
    )

    verifier_id = None
    for aid, node in compiled.agents.items():
        if getattr(node, "kind", None) == "verifier":
            verifier_id = aid
            break
    verifier_node = compiled.agents.get(verifier_id) if verifier_id else None
    verifier_prompt = (verifier_node.system_prompt if verifier_node else "") or ""

    router_id = compiled.route_out.get("planner")
    router_ids_run = list(compiled.fan_out.get("planner") or [])
    if not router_ids_run and router_id:
        router_ids_run = [router_id]
    direct_executor = ""
    if not router_ids_run:
        candidate = str(compiled.message_out.get("planner") or "")
        node = compiled.agents.get(candidate) if candidate else None
        profile = dict(getattr(node, "profile", None) or {}) if node else {}
        kind = getattr(node, "kind", None) if node else None
        if node is not None and (kind in ("blank", "tool") or str(profile.get("backend") or "") == "user_space"):
            direct_executor = candidate
        else:
            return EpisodeRaw(messages=[], error="centralized topology requires a router")
    else:
        for rid in router_ids_run:
            if rid not in compiled.routers:
                return EpisodeRaw(messages=[], error=f"router {rid!r} missing from compiled graph")
        if router_id not in compiled.routers:
            router_id = router_ids_run[0]

    blank_ids = {aid for aid, node in compiled.agents.items() if getattr(node, "kind", None) == "blank"}

    max_turns = max(4, int(spec.hub.max_feedback_hops or 1) * 3 + 4)
    turn = 0
    last_feedback: Optional[str] = None
    all_messages: List[Dict[str, Any]] = []
    window_events: List[Dict[str, Any]] = []
    n_search = 0
    n_python = 0
    final_answer: Optional[str] = None
    error: Optional[str] = None
    done = False

    while turn < max_turns and not done:
        from .user_gateway.sandbox import eval_cancelled

        if eval_cancelled():
            error = "用户停止"
            break
        turn += 1
        # 1. planner window -> plan_step
        if last_feedback:
            user = (
                f"{question}\n\n[Verifier feedback from previous turn]\n{last_feedback}\n"
                "Decide the next step. Output JSON with keys: next, args, sub_goal, done."
            )
        else:
            if router_ids_run:
                all_cands = []
                for rid in router_ids_run:
                    all_cands.extend(list(compiled.routers[rid].candidates or []))
                user = (
                    f"{question}\n\nDecide the next step and pick one tool from "
                    f"{all_cands}. Output JSON with keys: next, args, sub_goal, done."
                )
            else:
                user = (
                    f"{question}\n\nDecide the next step. next must be {direct_executor}. "
                    "Output JSON with keys: next, args, sub_goal, done."
                )
        try:
            planner_profile = dict(getattr(planner_node, "profile", None) or {})
            if str(planner_profile.get("backend") or "") == "user_space":
                inbound = make_message(
                    task_id=task_id, turn=turn, src="orchestrator", dst="planner",
                    kind="plan_step",
                    payload={"next": "", "args": {}, "sub_goal": "", "done": False, "question": question, "input": user},
                )
                from .user_gateway.loader import maybe_invoke_user_node

                umsg = maybe_invoke_user_node(planner_node, inbound, expected_kind="plan_step")
                if umsg is None:
                    p_text = win.invoke(planner_prompt, user, agent_id="planner", kind="planner")
                    p_json = _parse_json(p_text) or {}
                elif umsg.kind == "error":
                    error = f"planner window failed: {umsg.payload.get('error')}"
                    break
                else:
                    p_json = dict(umsg.payload or {})
            else:
                p_text = win.invoke(planner_prompt, user, agent_id="planner", kind="planner")
                p_json = _parse_json(p_text) or {}
        except Exception as e:  # noqa: BLE001
            error = f"planner window failed: {e}"
            break
        nxt = p_json.get("next")
        if nxt is None:
            nxt = ""
        args = p_json.get("args") if "args" in p_json else {}
        if not isinstance(args, (dict, list)):
            args = {"query": str(args)}
        if not nxt:
            only: List[str] = []
            for rid in router_ids_run:
                only.extend(list(compiled.routers[rid].candidates or []))
            if len(only) == 1:
                nxt = only[0]
            elif direct_executor:
                nxt = direct_executor
        p_payload = {
            "next": nxt,
            "args": args,
            "sub_goal": str(p_json.get("sub_goal") or ""),
            "done": bool(p_json.get("done")),
        }
        for extra in ("trace", "answer", "ready_to_stop"):
            if extra in p_json:
                p_payload[extra] = p_json[extra]
        try:
            plan_msg = make_message(
                task_id=task_id, turn=turn, src="planner", dst=router_id or direct_executor or "orchestrator",
                kind="plan_step", payload=p_payload,
            )
        except ValueError as ve:
            last_feedback = f"planner produced invalid plan_step: {ve}"
            err_msg = make_message(
                task_id=task_id, turn=turn, src="planner", dst="orchestrator",
                kind="error", payload={"agent_id": "planner", "error": str(ve)},
            )
            _log_window(archive, "planner", turn, err_msg, err_msg, False)
            all_messages.append(err_msg.model_dump())
            _emit_window(window_events, all_messages, agent_id="planner", kind="after_agent_turn", turn=turn, metrics={"ok": False})
            continue

        if p_payload["done"] and not p_payload.get("next"):
            final_answer = str(p_json.get("answer") or p_payload.get("sub_goal") or "")
            fa_msg = make_message(
                task_id=task_id, turn=turn, src="planner", dst="orchestrator",
                kind="final_answer", payload={"answer": final_answer},
                trace_ref=plan_msg.msg_id,
            )
            _log_window(archive, "planner", turn, plan_msg, fa_msg, True)
            all_messages.extend([plan_msg.model_dump(), fa_msg.model_dump()])
            _emit_window(window_events, all_messages, agent_id="planner", kind="after_agent_turn", turn=turn, metrics={})
            done = True
            break

        ok, reason = validate_payload("plan_step", plan_msg.payload)
        _log_window(archive, "planner", turn, plan_msg, plan_msg, ok)
        all_messages.append(plan_msg.model_dump())
        _emit_window(window_events, all_messages, agent_id="planner", kind="after_agent_turn", turn=turn, metrics={"ok": ok})
        if not ok:
            last_feedback = f"planner produced invalid plan_step: {reason}"
            continue

        # 2. Dispatch the executor window. A declared router still selects
        # candidates. A user wrap with planner → executor has no router: the
        # planner's next (or the message edge) is invoked directly.
        hops: List[tuple] = []  # (invoke_msg, agent_id, args)
        strict = len(router_ids_run) == 1
        fail_reason = ""
        if not router_ids_run:
            target = str(p_payload.get("next") or "") or direct_executor
            if target not in compiled.agents:
                target = direct_executor
            hop_args = args if isinstance(args, dict) else {}
            if target and target in compiled.agents:
                invoke_msg = make_message(
                    task_id=task_id, turn=turn, src="planner", dst=target,
                    kind="tool_invoke", payload=hop_args,
                    trace_ref=plan_msg.msg_id,
                )
                hops.append((invoke_msg, target, hop_args))
            else:
                fail_reason = "planner next is not an executor window"
        for rid in router_ids_run:
            rs = compiled.routers[rid]
            cand_runner = None
            score_runner = None
            if str(rs.strategy or "") == "score":
                def cand_runner(cid: str, ctx: Dict[str, Any]) -> str:
                    node = compiled.agents.get(cid)
                    prompt = (getattr(node, "system_prompt", None) if node else "") or f"You are {cid}."
                    payload = ctx if isinstance(ctx, dict) else {}
                    try:
                        return win.invoke(
                            prompt,
                            json.dumps(payload, ensure_ascii=False),
                            agent_id=cid,
                            kind="blank",
                        )
                    except Exception as e:  # noqa: BLE001
                        return str(e)

                def score_runner(cid: str, ctx: Dict[str, Any], _rs=rs) -> tuple:
                    scorer_id = _rs.scorer or cid
                    node = compiled.agents.get(scorer_id)
                    prompt = (
                        (getattr(node, "system_prompt", None) if node else "")
                        or "Score this candidate from 0 to 1. Output JSON {score: number}."
                    )
                    text = win.invoke(
                        prompt,
                        json.dumps({"candidate": cid, **(ctx or {})}, ensure_ascii=False),
                        agent_id=scorer_id,
                        kind="blank",
                    )
                    parsed = _parse_json(text) or {}
                    val = parsed.get("score")
                    if val is None:
                        val = parsed.get("value") or 0
                    return cid, float(val)

            route_res = router_select(
                rs, plan_msg, rs.candidates, strict=strict,
                scorer_runner=score_runner, candidate_runner=cand_runner,
            )
            _emit_window(
                window_events, all_messages,
                agent_id=rid, kind="after_agent_turn", turn=turn,
                metrics={**route_res.metrics, "selected": list(route_res.selected), "strategy": route_res.strategy},
            )
            if not route_res.ok:
                fail_reason = route_res.reason
                if strict:
                    hops = []
                    break
                continue
            for sel, args in zip(route_res.selected, route_res.args_list):
                invoke_msg = make_message(
                    task_id=task_id, turn=turn, src=rid, dst=sel,
                    kind="tool_invoke", payload=args if isinstance(args, dict) else {},
                    trace_ref=plan_msg.msg_id,
                )
                hops.append((invoke_msg, sel, args if isinstance(args, dict) else {}))
        if not hops:
            last_feedback = fail_reason or "no executor window selected"
            src = router_ids_run[0] if router_ids_run else "planner"
            fb = make_message(
                task_id=task_id, turn=turn, src=src, dst="planner",
                kind="feedback",
                payload={"reason": last_feedback, "attributed_to": src},
                trace_ref=plan_msg.msg_id,
            )
            all_messages.append(fb.model_dump())
            continue

        # 3. tool-agent / blank windows -> tool_result (parallel hops, then post-tool blank pipeline)
        def _invoke_blank(aid: str, targs: Dict[str, Any], prev_msg: AgentMessage) -> AgentMessage:
            node = compiled.agents.get(aid)
            profile = dict(getattr(node, "profile", None) or {}) if node else {}
            in_schema = profile.get("input_schema") or DEFAULT_INPUT_SCHEMA
            out_schema = profile.get("output_schema") or DEFAULT_OUTPUT_SCHEMA
            wrapped = targs if isinstance(targs, dict) else {"input": str(targs)}
            user_space = str(profile.get("backend") or "") == "user_space"
            if "input" not in wrapped and not user_space:
                wrapped = {"input": str(wrapped.get("output") or wrapped.get("query") or wrapped)}
            ok_in, why = (True, "") if user_space else validate_json_schema(wrapped, in_schema)
            dst = verifier_id or "verifier"
            if not ok_in:
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"invalid input: {why}", "ok": False, "evidence_type": "EMPTY"},
                    trace_ref=prev_msg.msg_id,
                )
            if str(profile.get("backend") or "") == "user_space":
                inbound = make_message(
                    task_id=task_id, turn=turn, src=prev_msg.src, dst=aid,
                    kind="tool_invoke", payload=wrapped, trace_ref=prev_msg.msg_id,
                )
                try:
                    from .user_gateway.loader import maybe_invoke_user_node

                    out = maybe_invoke_user_node(node, inbound, expected_kind="tool_result")
                    if out is not None:
                        return out
                except Exception as e:  # noqa: BLE001
                    return make_message(
                        task_id=task_id, turn=turn, src=aid, dst=dst,
                        kind="tool_result",
                        payload={"output": f"user agent error: {e}", "ok": False, "evidence_type": "ERROR"},
                        trace_ref=prev_msg.msg_id,
                    )
            prompt = (getattr(node, "system_prompt", None) or f"You are {aid}.") if node else f"You are {aid}."
            try:
                text = win.invoke(prompt, json.dumps(wrapped, ensure_ascii=False), agent_id=aid, kind="blank")
            except Exception as e:  # noqa: BLE001
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"blank agent error: {e}", "ok": False, "evidence_type": "ERROR"},
                    trace_ref=prev_msg.msg_id,
                )
            parsed = _parse_json(text)
            if parsed is None:
                parsed = {"output": text}
            ok_out, why = validate_json_schema(parsed, out_schema)
            if not ok_out:
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"invalid output: {why}", "ok": False, "evidence_type": "EMPTY"},
                    trace_ref=prev_msg.msg_id,
                )
            return make_message(
                task_id=task_id, turn=turn, src=aid, dst=dst,
                kind="tool_result",
                payload={"output": str(parsed.get("output") or json.dumps(parsed)), "ok": True, "evidence_type": "DIRECT"},
                trace_ref=prev_msg.msg_id,
            )

        def _invoke_tool(aid: str, targs: Dict[str, Any], prev_msg: AgentMessage) -> AgentMessage:
            dst = verifier_id or "verifier"
            node = compiled.agents.get(aid)
            profile = getattr(node, "profile", None) if node is not None else {}
            if not isinstance(profile, dict):
                profile = {}
            tier = effective_tier(profile.get("tier"))
            if str(profile.get("backend") or "") == "user_space" and node is not None:
                inbound = make_message(
                    task_id=task_id, turn=turn, src=prev_msg.src, dst=aid,
                    kind="tool_invoke", payload=targs if isinstance(targs, dict) else {},
                    trace_ref=prev_msg.msg_id,
                )
                try:
                    from .user_gateway.loader import maybe_invoke_user_node

                    umsg = maybe_invoke_user_node(node, inbound, expected_kind="tool_result")
                    if umsg is not None:
                        payload = dict(umsg.payload or {})
                        payload.setdefault("output", "")
                        payload.setdefault("ok", True)
                        payload.setdefault(
                            "evidence_type",
                            "DIRECT" if str(payload.get("output") or "").strip() else "EMPTY",
                        )
                        payload["tier"] = tier
                        return make_message(
                            task_id=task_id, turn=turn, src=aid, dst=dst,
                            kind="tool_result", payload=payload, trace_ref=prev_msg.msg_id,
                        )
                except Exception as e:  # noqa: BLE001
                    from .user_gateway.sandbox import EvalCancelled

                    if isinstance(e, EvalCancelled):
                        raise
                    return make_message(
                        task_id=task_id, turn=turn, src=aid, dst=dst,
                        kind="tool_result",
                        payload={"output": f"user tool error: {e}", "ok": False, "evidence_type": "ERROR", "tier": tier},
                        trace_ref=prev_msg.msg_id,
                    )
            ok_args, why = validate_tool_args(aid, targs if isinstance(targs, dict) else {})
            if not ok_args:
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"invalid args: {why}", "ok": False, "evidence_type": "ERROR", "tier": tier},
                    trace_ref=prev_msg.msg_id,
                )
            ta = get_tool_agent(aid)
            if ta is None:
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"unknown tool-agent {aid}", "ok": False, "evidence_type": "ERROR", "tier": tier},
                    trace_ref=prev_msg.msg_id,
                )
            try:
                out = ta.invoke(targs, tier=tier)
            except Exception as e:  # noqa: BLE001
                return make_message(
                    task_id=task_id, turn=turn, src=aid, dst=dst,
                    kind="tool_result",
                    payload={"output": f"tool error: {e}", "ok": False, "evidence_type": "ERROR", "tier": tier},
                    trace_ref=prev_msg.msg_id,
                )
            evidence = "EMPTY" if not str(out).strip() else "DIRECT"
            payload = {"output": str(out), "ok": True, "evidence_type": evidence, "tier": tier}
            last = getattr(ta, "last_payload", None)
            if isinstance(last, dict):
                for key in ("trace", "command", "analysis", "explanation", "ok", "evidence_type", "output"):
                    if key in last:
                        payload[key] = last[key]
                if last.get("output") is not None:
                    payload["output"] = str(last.get("output"))
                payload["ok"] = bool(last.get("ok", payload["ok"]))
                if not str(payload.get("output") or "").strip():
                    payload["evidence_type"] = last.get("evidence_type") or "EMPTY"
            return make_message(
                task_id=task_id, turn=turn, src=aid, dst=dst,
                kind="tool_result", payload=payload,
                trace_ref=prev_msg.msg_id,
            )

        def _run_hop(idx: int) -> List[AgentMessage]:
            invoke_msg, aid, targs = hops[idx]
            node = compiled.agents.get(aid)
            kind = getattr(node, "kind", None) if node else None
            if kind == "blank" or aid in blank_ids:
                first = _invoke_blank(aid, targs, invoke_msg)
            else:
                first = _invoke_tool(aid, targs, invoke_msg)
            chain = [invoke_msg, first]
            nxt = compiled.message_out.get(aid)
            prev = first
            seen = {aid}
            while nxt and nxt in blank_ids and nxt not in seen:
                seen.add(nxt)
                bmsg = _invoke_blank(nxt, {"input": str(prev.payload.get("output") or "")}, prev)
                chain.append(bmsg)
                nxt = compiled.message_out.get(nxt)
                prev = bmsg
            return chain

        chains: List[List[AgentMessage]] = []
        if len(hops) == 1:
            chains.append(_run_hop(0))
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(hops))) as ex:
                chains = list(ex.map(_run_hop, range(len(hops))))

        tool_msgs: List[AgentMessage] = []
        for chain in chains:
            for tmsg in chain:
                all_messages.append(tmsg.model_dump())
                if tmsg.kind != "tool_result":
                    continue
                tool_msgs.append(tmsg)
                in_msg = next((m for m in chain if m.kind == "tool_invoke" and m.dst == tmsg.src), tmsg)
                _log_window(archive, tmsg.src, turn, in_msg, tmsg, bool(tmsg.payload.get("ok")))
                _emit_window(
                    window_events, all_messages,
                    agent_id=tmsg.src, kind="after_agent_turn", turn=turn,
                    metrics={"ok": bool(tmsg.payload.get("ok"))},
                    tool_id=tmsg.src,
                )
                if tmsg.src in ("wikipedia_search", "bing_search", "web_fetch", "google_search", "web_search"):
                    n_search += 1
                if tmsg.src == "python_coder":
                    n_python += 1

        last_tool = tool_msgs[-1] if tool_msgs else None

        # 4. verifier window -> verify
        if last_tool is None:
            last_feedback = "no tool_result produced"
            continue
        if verifier_id is None:
            v_ok = any(m.payload.get("ok") for m in tool_msgs)
            v_payload = {
                "ok": v_ok,
                "reason": "" if v_ok else "all tool results failed",
                "step_conclusion": "COMPLETE" if v_ok else "INCOMPLETE",
                "slot_updates": [],
            }
            v_msg = make_message(
                task_id=task_id, turn=turn, src="verifier", dst="planner",
                kind="verify", payload=v_payload, trace_ref=last_tool.msg_id,
            )
            _log_window(archive, "verifier", turn, last_tool, v_msg, v_ok)
            all_messages.append(v_msg.model_dump())
            _emit_window(window_events, all_messages, agent_id="verifier", kind="after_verifier", turn=turn, metrics={"ok": v_ok})
        else:
            tool_summary = "\n".join(f"- {m.src}: {m.payload.get('output')}" for m in tool_msgs)
            v_user = (
                f"Sub-goal: {p_payload.get('sub_goal')}\n\nTool results:\n{tool_summary}\n\n"
                "Judge if the sub-goal is complete. Output JSON with keys: ok, reason, "
                "step_conclusion (COMPLETE|INCOMPLETE), slot_updates."
            )
            try:
                v_profile = dict(getattr(verifier_node, "profile", None) or {}) if verifier_node else {}
                if str(v_profile.get("backend") or "") == "user_space" and verifier_node is not None:
                    inbound = make_message(
                        task_id=task_id, turn=turn, src=last_tool.src, dst=verifier_id,
                        kind="verify",
                        payload={"ok": False, "reason": v_user, "step_conclusion": "INCOMPLETE", "slot_updates": []},
                        trace_ref=last_tool.msg_id,
                    )
                    from .user_gateway.loader import maybe_invoke_user_node

                    umsg = maybe_invoke_user_node(verifier_node, inbound, expected_kind="verify")
                    if umsg is None:
                        v_text = win.invoke(verifier_prompt, v_user, agent_id=verifier_id, kind="verifier")
                        v_json = _parse_json(v_text) or {}
                    elif umsg.kind == "error":
                        error = f"verifier window failed: {umsg.payload.get('error')}"
                        break
                    else:
                        v_json = dict(umsg.payload or {})
                else:
                    v_text = win.invoke(verifier_prompt, v_user, agent_id=verifier_id, kind="verifier")
                    v_json = _parse_json(v_text) or {}
            except Exception as e:  # noqa: BLE001
                error = f"verifier window failed: {e}"
                break
            v_payload = {
                "ok": bool(v_json.get("ok")),
                "reason": str(v_json.get("reason") or ""),
                "step_conclusion": str(v_json.get("step_conclusion") or "INCOMPLETE"),
                "slot_updates": list(v_json.get("slot_updates") or []),
            }
            for extra in ("trace", "answer", "ready_to_stop", "direct_output"):
                if extra in v_json:
                    v_payload[extra] = v_json[extra]
            if "evidence_type" not in v_payload:
                v_payload["evidence_type"] = last_tool.payload.get("evidence_type") or "DIRECT"
            try:
                v_msg = make_message(
                    task_id=task_id, turn=turn, src=verifier_id, dst="planner",
                    kind="verify", payload=v_payload, trace_ref=last_tool.msg_id,
                )
                v_ok = v_payload["ok"]
            except ValueError as ve:
                v_ok = False
                v_msg = make_message(
                    task_id=task_id, turn=turn, src=verifier_id, dst="planner",
                    kind="error", payload={"agent_id": verifier_id, "error": str(ve)},
                    trace_ref=last_tool.msg_id,
                )
                last_feedback = f"verifier produced invalid verify: {ve}"
            _log_window(archive, verifier_id, turn, last_tool, v_msg, v_ok)
            all_messages.append(v_msg.model_dump())
            _emit_window(window_events, all_messages, agent_id=verifier_id, kind="after_verifier", turn=turn, metrics={"ok": v_ok})

        # 5. termination / feedback / fact commit
        if v_msg.kind == "verify" and commit_to_fact(v_msg.payload, last_tool.payload if last_tool else {}):
            archive.append(
                ExecutionEvent(
                    kind=EventKind.MEMORY_WRITE,
                    agent_id=verifier_id or "verifier",
                    payload={"phase": "fact_commit", "turn": turn, "ok": True},
                )
            )
        answer = str(
            v_payload.get("answer")
            or v_payload.get("direct_output")
            or p_json.get("answer")
            or p_payload.get("answer")
            or ""
        ).strip()
        ready = bool(v_payload.get("ready_to_stop") or p_payload.get("ready_to_stop"))
        if ready:
            final_answer = answer
            done = True
        elif answer and (v_payload.get("ok") or ready):
            final_answer = answer
            done = True
        elif v_msg.payload.get("ok") and p_payload.get("done"):
            final_answer = answer or str(p_payload.get("sub_goal") or "")
            done = True
        else:
            last_feedback = str(v_msg.payload.get("reason") or last_feedback or "verify_failed")
            fb = make_message(
                task_id=task_id, turn=turn,
                src=verifier_id or "verifier", dst="planner",
                kind="feedback",
                payload={
                    "reason": last_feedback,
                    "attributed_to": last_tool.src if last_tool else (verifier_id or "verifier"),
                },
                trace_ref=v_msg.msg_id,
            )
            all_messages.append(fb.model_dump())
            _log_window(archive, verifier_id or "verifier", turn, v_msg, fb, False)

    raw = EpisodeRaw(
        messages=all_messages,
        final_answer=final_answer,
        format_ok=bool(final_answer),
        n_search=n_search,
        n_python=n_python,
        window_events=window_events,
        error=error,
    )
    return raw

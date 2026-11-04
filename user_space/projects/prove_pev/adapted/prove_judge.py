"""LLM-as-a-Judge protocol ported from MAS-ProVe ``server_judge.py``.

Keeps the same system prompt, candidate framing, and ``<ranking>`` parse rules.
Default backend is inline (LocalChatEngine / frozen verifier stack). Optional TCP
backend talks to a running ``mas_proceval`` Judge server.
"""

from __future__ import annotations

import json
import socket
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence


def format_trajectory_for_eval(trajectory: Dict[str, Any]) -> str:
    context = trajectory["context"] if isinstance(trajectory.get("context"), str) else str(trajectory.get("context") or "")
    current_step = (
        trajectory["current-step"]
        if isinstance(trajectory.get("current-step"), str)
        else str(trajectory.get("current-step") or "")
    )
    return "\n".join([context, current_step])


def feedback2rankings(feedback: str, num_candidates: int) -> List[int]:
    """Parse 1-indexed ``<ranking>2,1,3</ranking>`` into 0-indexed ranks."""
    text = str(feedback or "")
    if "<ranking>" in text and "</ranking>" in text:
        start = text.find("<ranking>") + len("<ranking>")
        end = text.find("</ranking>")
        ranking_text = text[start:end].strip()
        try:
            ranks = [int(x.strip()) - 1 for x in ranking_text.split(",")]
            if len(ranks) == num_candidates and set(ranks) == set(range(num_candidates)):
                return ranks
        except (ValueError, IndexError):
            pass
    return list(range(num_candidates))


def build_messages_for_judge(
    trajectories_for_eval: Sequence[str],
    task_type: Optional[str] = None,
    question: str = "",
) -> List[Dict[str, str]]:
    """Exact prompt construction from MAS-ProVe ``build_messages_for_judge``."""
    PROMPT_RANKING = """
    <|User Prompt|>
    {question}

    {candidates_text}
    """.strip()

    kind = str(task_type or "qa").lower()
    if "qa" in kind or "math" in kind:
        system_prompt = """Please act as an impartial judge and evaluate the quality of multiple responses provided by AI assistants to the user prompt displayed below. You will be given several candidate responses.

Your task is to rank these responses from best to worst based on their correctness, reasoning quality, and completeness. Consider:
- Correctness of the final answer
- Quality and clarity of reasoning
- Completeness of the solution
- Logical consistency

Be as objective as possible. Avoid any biases such as length or stylistic elements like formatting.

Before providing your ranking, think through the evaluation process and output your thoughts as an explanation.

After providing your explanation, you must output the ranking as a comma-separated list of candidate numbers (e.g., "2,1,3" means candidate 2 is best, candidate 1 is second, candidate 3 is worst). Please enclose your ranking in <ranking> and </ranking> tags.
""".strip()
        candidates_parts = []
        for i, trajectory in enumerate(trajectories_for_eval, 1):
            candidates_parts.append(f"<|Candidate {i} Response|>\n{trajectory}\n<|End of Candidate {i}|>")
        candidates_text = "\n\n".join(candidates_parts)
        prompt_formatted = PROMPT_RANKING.format(question=question, candidates_text=candidates_text)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_formatted},
        ]

    warnings.warn(f"Unknown task type: {task_type}, using the default prompt for judge.")
    return [
        {
            "role": "system",
            "content": (
                "You are an impartial judge who's good at reasoning and deciding the quality of "
                "the given complex systems' trajectories. Judge the quality of the given "
                "trajectories in a fair and objective manner and give out the rankings of them."
            ),
        },
        {"role": "user", "content": "\n\n".join(trajectories_for_eval)},
    ]


def judge_inline(
    partial_trajectories: Sequence[Dict[str, Any]],
    *,
    question: str,
    task_type: str,
    llm_call: Callable[[str, str], str],
) -> Dict[str, Any]:
    """Run Judge via ``llm_call(system, user) -> feedback``."""
    trajectories_for_eval = [format_trajectory_for_eval(dict(t)) for t in partial_trajectories]
    messages = build_messages_for_judge(trajectories_for_eval, task_type, question)
    system = messages[0]["content"] if messages else ""
    user = messages[1]["content"] if len(messages) > 1 else ""
    feedback = llm_call(system, user)
    rankings = feedback2rankings(feedback, len(trajectories_for_eval))
    return {
        "messages-for-proc-eval": messages,
        "judge-feedback": feedback,
        "rankings": rankings,
    }


def judge_tcp(
    partial_trajectories: Sequence[Dict[str, Any]],
    *,
    question: str,
    task_type: str,
    host: str = "127.0.0.1",
    port: int = 5555,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Minimal TCP client compatible with ``mas_proceval.clients.client_base.BaseClient``."""
    request = {
        "task-type": task_type,
        "judge-type": "judge",
        "partial_trajectories": list(partial_trajectories),
        "question": question,
    }
    payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall(payload)
        chunks: List[bytes] = []
        while True:
            block = sock.recv(65536)
            if not block:
                break
            chunks.append(block)
            if b"\n" in block:
                break
    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    try:
        result = json.loads(raw.split("\n", 1)[0])
    except json.JSONDecodeError:
        result = {}
    rankings = result.get("rankings")
    n = len(partial_trajectories)
    if not isinstance(rankings, list) or len(rankings) != n:
        rankings = list(range(n))
    return {
        "messages-for-proc-eval": result.get("messages-for-proc-eval") or [],
        "judge-feedback": str(result.get("judge-feedback") or ""),
        "rankings": [int(x) for x in rankings],
    }

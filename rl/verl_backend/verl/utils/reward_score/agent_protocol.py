"""Shared APPO/ARPO agent protocol: system prompt and Qwen3 chat-template helpers."""

from __future__ import annotations

from typing import Any, List, Optional

# Keep this under ~1.2k chars so it still fits data.max_prompt_length=2048 with the question.
SYSTEM_PROMPT = """You are a helpful assistant that solves questions with a Wikipedia search tool and a Python interpreter.

Follow this protocol exactly:
1. Reason only inside <think>...</think>. Close </think> BEFORE any tool call.
2. Search with <search>short specific query</search>. After <result> appears, open a new <think> to interpret it.
3. Calculate with <python>code</python> the same way. Do not invent tool results.
4. When ready, output exactly:
<answer>
\\boxed{SHORT_ANSWER}
</answer>
5. Never emit extra </think> after a result. Never emit <tool_call>. Never write placeholder queries like "search for the answer". One <answer> only. Keep the boxed answer short (entity, date, number, or short phrase).
6. Search at most 3 times. As soon as you can answer, output <answer>\\boxed{SHORT_ANSWER}</answer> immediately. Do not stop after </result> without an answer.

Example:
<think>
I need the nationality of the director of A Perfect Couple (1979).
</think>
<search>A Perfect Couple 1979 director</search>
<result>
...
</result>
<think>
The director is Robert Altman, who was American.
</think>
<answer>
\\boxed{American}
</answer>"""


def format_appo_chat_prefix(messages: List[dict], add_generation_prompt: bool = True) -> str:
    """Qwen chat prefix without native thinking injection.

    ``enable_thinking=False`` inserts an empty ``<think></think>`` before
    generation, which stops the model from emitting APPO ``<think>`` and makes
    it drop the turn after tool results. ``enable_thinking=True`` injects a
    second think channel. Manual prefix avoids both.
    """
    system = ""
    user = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            pass
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def apply_chat_template_no_native_think(tokenizer: Any, messages: List[dict], **kwargs) -> Any:
    tokenize = kwargs.pop("tokenize", False)
    add_generation_prompt = kwargs.pop("add_generation_prompt", True)
    text = format_appo_chat_prefix(messages, add_generation_prompt=add_generation_prompt)
    if tokenize:
        return tokenizer.encode(text, add_special_tokens=False)
    return text

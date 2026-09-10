# Copyright (c) Microsoft. All rights reserved.

"""可调用 Python 工具的 GSM8K 数学 Agent，并通过 Agent-lightning 做 RL 训练。

整体模式对齐 Spider SQL 示例（LangGraph 图 + LitAgent.rollout），
但任务换成数学题：工具是受限 ``execute_python``，奖励是答案匹配而非 SQL 执行。

模块分层:
1. 解析 / 奖励工具函数：从文本或消息列表抽出答案，并算 reward。
2. MathAgent：纯推理图（可独立 debug，不依赖 VERL）。
3. LitMathAgent：训练封装，对接 resources['main_llm'] 与 emit_reward。
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from python_tool import execute_python

import agentlightning as agl

agl.setup_logging(apply_to=[__name__])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 设计
# - SYSTEM_PROMPT：教模型「可调工具 + 最终必须用 ### NUMBER ### 收口」。
# - FINALIZE_PROMPT：当轮数将尽仍无答案时，强制再生成一次短答案（关掉 tools）。
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful math assistant.
Solve the user's math problem step by step.
You may call the `execute_python_tool` tool to run short Python snippets for calculation.
Rules for code:
- No imports (math and operator are already available as `math` / `operator`).
- Prefer assigning the final numeric value to a variable named `result`.
- Keep code short and self-contained.

When you know the final answer, stop calling tools and reply with EXACTLY this format
(replace NUMBER with the real answer, not the word answer):
### NUMBER ###
""".strip()

FINALIZE_PROMPT = (
    "Stop using tools. Based on the conversation above, output ONLY the final numeric answer "
    "in this exact format (no other text):\n### NUMBER ###"
)


class MathProblem(TypedDict):
    """数据集单条样本的字段约定（与 prepare_data 写出的 parquet 对齐）。"""

    id: str
    question: str
    answer: str


class AgentState(TypedDict):
    """LangGraph 状态。

    messages: 完整对话（含 System/Human/AI/Tool），也是 RL 轨迹的载体。
    question: 原始题目，便于日志；推理主要看 messages。
    num_turns: 已调用模型的次数，用于限制多轮，防止无限 tool 循环。
    asked_finalize: 是否已经注入过 FINALIZE_PROMPT；只允许强制收口一次。
    """

    messages: List[AnyMessage]
    question: str
    num_turns: int
    asked_finalize: bool


@tool
def execute_python_tool(code: str) -> str:
    """供 LLM bind_tools 使用的 LangChain Tool 包装。

    参数:
        code: 模型生成的短 Python 片段。沙箱禁止 import；约定把最终值放进 ``result``。

    返回:
        工具 stdout / ``result=...`` 字符串，或错误信息（错误也回写轨迹，供模型改写）。
    """
    return execute_python(code)


# 当前只挂一个工具；TOOL_MAP 用 name 索引，便于 call_tools 分发。
TOOLS = [execute_python_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

# 模型有时会原样输出占位词（抄 system 里的 NUMBER/answer）；这些不算有效答案。
_PLACEHOLDER_ANSWERS = {
    "",
    "<answer>",
    "answer",
    "ans",
    "number",
    "<number>",
}


def _strip_thinking(text: str) -> str:
    """去掉 ``<think>...</think>`` 段，避免思考链里的中间数字干扰答案解析。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _clean_candidate(ans: str) -> Optional[str]:
    """清洗候选答案字符串；若是占位词则返回 None（视为「尚未给出真答案」）。"""
    ans = ans.strip()
    ans = re.sub(r"</?answer>", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip("`\"' ")
    if ans.lower() in _PLACEHOLDER_ANSWERS:
        return None
    return ans or None


def extract_final_answer(text: str) -> Optional[str]:
    """从单段模型文本中抽取最终答案。

    设计逻辑（按优先级，从前到后）:
    1. GSM8K 官方 ``#### answer``（必须先于 ``###``，否则会被 ``###`` 规则误伤）。
    2. 本项目主格式 ``### NUMBER ###``（取最后一次匹配，兼容多轮改口）。
    3. ``<answer>...</answer>``。
    4. 英文口头禅 “the answer is / final answer:”。
    5. 兜底：取非 thinking 文本中最后一个数字（噪声最大，但能救部分漏格式样本）。

    返回:
        清洗后的答案字符串；解析失败为 None（训练里通常对应 reward=0）。
    """
    if not text:
        return None

    cleaned = _strip_thinking(text) or text

    # 1) GSM8K #### answer（必须先于 ### 匹配）
    match = re.search(r"####\s*(.+)", cleaned)
    if match:
        ans = _clean_candidate(match.group(1).split("\n")[0])
        if ans is not None:
            return ans

    # 2) ### ... ###  （排除 ####）
    matches = list(re.finditer(r"(?<!#)###\s*(.+?)(\s*###|$)", cleaned, flags=re.DOTALL))
    if matches:
        ans = _clean_candidate(matches[-1].group(1))
        if ans is not None:
            return ans

    # 3) <answer>...</answer>
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        ans = _clean_candidate(match.group(1))
        if ans is not None:
            return ans

    # 4) "the answer is 42" / "final answer: 42"
    match = re.search(
        r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=])\s*(-?\d[\d,]*(?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

    # 5) 非 thinking 文本中最后一个独立数字
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_answer_from_messages(messages: List[AnyMessage]) -> Optional[str]:
    """从整段对话里抽答案：优先 assistant 文本，其次 tool 的 ``result=``。

    为什么从后往前扫:
        多轮改写时，最后一次给出的答案更可能是「最终决定」。
        若模型只算到 tool、忘记口头收口，仍可从 ``result = 42`` 拿到数
        （这也是一种潜在 reward 噪声/泄漏路径，面试可当 hacking 例子讲）。
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "") or ""
            ans = extract_final_answer(str(content))
            if ans is not None:
                return ans
        if isinstance(msg, ToolMessage):
            content = str(getattr(msg, "content", "") or "")
            match = re.search(r"result\s*=\s*(.+)", content)
            if match:
                ans = _clean_candidate(match.group(1))
                if ans is not None:
                    # 去掉 repr 风格引号，例如 '180' → 180
                    if (ans.startswith("'") and ans.endswith("'")) or (ans.startswith('"') and ans.endswith('"')):
                        ans = ans[1:-1]
                    return ans
            ans = extract_final_answer(content)
            if ans is not None:
                return ans
    return None


def normalize_answer(value: str) -> str:
    """规范化答案以便比较：去千分位逗号、货币/%、末尾句点，并抽出前导数值。

    例如 ``$1,234.00`` → ``1234.00``；``16 apples`` → ``16``。
    """
    value = value.strip()
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.replace("%", "")
    value = value.rstrip(".")
    # 若混有单位/文字，只取开头的数字 token。
    match = re.match(r"(-?\d+(?:\.\d+)?)", value)
    if match:
        return match.group(1)
    return value.strip()


# 奖励模式：
# - binary: 训练默认，只看对错（经典 RLVR）
# - shaped: 正确性 + 格式奖励 - 长度惩罚（研究 shaping / hacking）
# - format_only: 只看有没有 ### N ###（故意展示「刷格式不刷正确」）
RewardMode = Literal["binary", "shaped", "format_only"]


def _answers_match(prediction: Optional[str], ground_truth: str) -> bool:
    """判断预测与金标是否一致：优先数值近似，否则不区分大小写的字符串相等。"""
    if prediction is None:
        return False
    pred = normalize_answer(prediction)
    gt = normalize_answer(str(ground_truth))
    if not pred:
        return False
    try:
        if np.isclose(float(pred), float(gt), rtol=1e-5, atol=1e-8):
            return True
    except (TypeError, ValueError):
        pass
    return pred.lower() == gt.lower()


def _has_strict_format(text: Optional[str]) -> bool:
    """是否包含严格的 ``### 数字 ###``（用于 shaped / format_only 消融）。"""
    if not text:
        return False
    cleaned = _strip_thinking(text) or text
    return re.search(r"(?<!#)###\s*(-?\d[\d,]*(?:\.\d+)?)\s*###", cleaned) is not None


def compute_reward(
    prediction: Optional[str],
    ground_truth: str,
    *,
    mode: RewardMode = "binary",
    raw_assistant_text: Optional[str] = None,
    num_assistant_chars: int = 0,
    length_penalty_start: int = 1500,
    length_penalty_coef: float = 0.0002,
) -> float:
    """根据预测与金标计算标量奖励。

    参数:
        prediction: 从轨迹解析出的答案；None 表示解析失败。
        ground_truth: 数据集金标。
        mode: 奖励模式，见 ``RewardMode``。
        raw_assistant_text: 用于检测严格格式的原始 assistant 文本；
            若为 None，则退化为用 ``prediction`` 字符串检测（较弱）。
        num_assistant_chars: 全部 AIMessage 字符数，供长度惩罚使用。
        length_penalty_start: 超过该长度才开始罚分（避免误伤正常 CoT）。
        length_penalty_coef: 每超出 1 字符的惩罚系数。

    返回:
        标量 reward。binary/format_only 为 0/1；shaped 夹在 [0, 1.2]。

    设计意图:
        正式训准确率用 binary；shaped/format_only 用于面试消融，
        证明「优化目标 ≠ 评测口径」以及 format_only 的 hacking。
    """
    correct = _answers_match(prediction, ground_truth)
    formatted = _has_strict_format(raw_assistant_text if raw_assistant_text is not None else prediction)

    if mode == "binary":
        return 1.0 if correct else 0.0

    if mode == "format_only":
        return 1.0 if formatted else 0.0

    # shaped：主信号仍是正确性；格式小奖励鼓励可解析；过长扣分防废话刷分。
    reward = 0.0
    if correct:
        reward += 1.0
    if formatted:
        reward += 0.1
    if num_assistant_chars > length_penalty_start:
        reward -= length_penalty_coef * float(num_assistant_chars - length_penalty_start)
    return float(max(0.0, min(1.2, reward)))


def last_assistant_text(messages: List[AnyMessage]) -> Optional[str]:
    """取对话中最后一条 AIMessage 的文本（用于格式检测 / 日志）。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "") or ""
            return str(content)
    return None


def count_assistant_chars(messages: List[AnyMessage]) -> int:
    """统计所有 AIMessage 字符总数（shaped 长度惩罚的输入）。

    只计 assistant，不计 tool/user：惩罚的是「模型自己说太多」，
    而不是工具回传或题目本身。
    """
    total = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            total += len(str(getattr(msg, "content", "") or ""))
    return total


class MathAgent:
    """LangGraph 数学 Agent：可多轮调用 ``execute_python_tool``，并强制收口答案。

    与训练框架解耦：只依赖 OpenAI-compatible ``endpoint``。
    训练时 endpoint 指向 VERL/vLLM；debug 时指向任意兼容服务。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float = 0.7,
        max_turns: int = 8,
        max_tokens: int = 1024,
        debug: bool = False,
    ) -> None:
        """初始化两套 LLM 客户端。

        参数:
            endpoint: OpenAI API base（如 http://host:port/v1）。
            model_name: 模型名或本地路径（需与服务端加载的一致）。
            temperature: 主循环采样温度；train 通常偏高以利探索。
            max_turns: 最多调用模型多少次（含 finalize 那轮）。
            max_tokens: 单次生成上限。
            debug: 预留调试开关（当前逻辑未强依赖）。

        为何两套 llm:
            - ``self.llm``：bind_tools，负责推理与 tool-call。
            - ``self.llm_finalize``：不绑工具、更低温度、更短 max_tokens，
              专门在收口阶段输出 ``### N ###``，降低继续瞎调工具的概率。
        """
        self.max_turns = max_turns
        self.debug = debug
        self.model_name = model_name
        self.llm = init_chat_model(
            model_name,
            model_provider="openai",
            openai_api_base=endpoint,
            # 本地 vLLM 常不校验 key；给 dummy 即可满足客户端库要求。
            openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=temperature,
            max_retries=0,
            max_tokens=max_tokens,
        ).bind_tools(TOOLS)
        # Finalize 轮：无 tools，强制短答案格式。
        self.llm_finalize = init_chat_model(
            model_name,
            model_provider="openai",
            openai_api_base=endpoint,
            openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=min(temperature, 0.3),
            max_retries=0,
            max_tokens=min(max_tokens, 256),
        )

    def call_model(self, state: AgentState) -> AgentState:
        """图节点：调用 LLM，把 AIMessage 追加进 messages，并 ``num_turns += 1``。

        若 ``asked_finalize`` 已为 True，改走 ``llm_finalize``（无工具、短输出）。
        调用失败时写入占位 ``### None ###``，保证图能继续走到 reward=0，而不是整进程崩。
        """
        messages = list(state["messages"])
        asked_finalize = bool(state.get("asked_finalize", False))
        num_turns = state.get("num_turns", 0)

        use_finalize = asked_finalize
        llm = self.llm_finalize if use_finalize else self.llm

        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("LLM invoke failed: %s", e)
            response = AIMessage(content="### None ###")

        return {
            **state,
            "messages": messages + [response],
            "num_turns": num_turns + 1,
            "asked_finalize": asked_finalize,
        }

    def request_finalize(self, state: AgentState) -> AgentState:
        """图节点：追加一条 HumanMessage（FINALIZE_PROMPT），并标记已请求收口。

        下一跳会回到 ``agent`` 节点；因 ``asked_finalize=True``，将使用无工具 LLM。
        """
        messages = list(state["messages"]) + [HumanMessage(content=FINALIZE_PROMPT)]
        return {
            **state,
            "messages": messages,
            "asked_finalize": True,
        }

    def call_tools(self, state: AgentState) -> AgentState:
        """图节点：执行上一条 AIMessage 中的 tool_calls，追加 ToolMessage。

        设计细节:
        - 兼容 dict / 对象两种 tool_call 结构（不同 langchain 版本字段形态不同）。
        - 未知工具或执行异常都写成错误字符串回传，让模型有机会自我纠正。
        - 不在这里算 reward；reward 在整段 rollout 结束后统一计算。
        """
        last = state["messages"][-1]
        tool_messages: List[ToolMessage] = []
        tool_calls = getattr(last, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            tool_fn = TOOL_MAP.get(name or "")
            if tool_fn is None:
                content = f"Error: unknown tool {name}"
            else:
                try:
                    content = tool_fn.invoke(args)
                except Exception as e:
                    content = f"Error invoking tool: {e}"
            tool_messages.append(ToolMessage(content=str(content), tool_call_id=call_id or name or "tool"))
        return {**state, "messages": state["messages"] + tool_messages}

    def should_continue(self, state: AgentState) -> Literal["tools", "finalize", "end"]:
        """条件边：在 agent 节点之后决定下一步。

        优先级:
        1. 有 tool_calls 且未超轮次且未 finalize → ``tools``
        2. 轨迹里已能解析出答案 → ``end``
        3. 还没 finalize 且还有轮次 → ``finalize``（再给一次收口机会）
        4. 否则 ``end``（可能无答案，reward=0）
        """
        last = state["messages"][-1]
        num_turns = state.get("num_turns", 0)
        asked_finalize = bool(state.get("asked_finalize", False))

        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls and num_turns < self.max_turns and not asked_finalize:
            return "tools"

        # 从整段 transcript（assistant + tool）判断是否已有答案。
        if extract_answer_from_messages(state["messages"]) is not None:
            return "end"

        if not asked_finalize and num_turns < self.max_turns:
            return "finalize"

        return "end"

    def graph(self) -> CompiledStateGraph:
        """编译状态机:

        START → agent ⇄ tools
                 ↓
              finalize → agent → END

        ``recursion_limit`` 在 invoke 时另设，防止图递归超限。
        """
        builder = StateGraph(AgentState)
        builder.add_node("agent", self.call_model)
        builder.add_node("tools", self.call_tools)
        builder.add_node("finalize", self.request_finalize)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "finalize": "finalize", "end": END},
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", "agent")
        return builder.compile()

    def run(self, question: str) -> str:
        """便捷入口：跑完整张图，返回最后一条消息文本（本地手工试调用用）。"""
        initial: AgentState = {
            "question": question,
            "num_turns": 0,
            "asked_finalize": False,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }
        result = self.graph().invoke(initial, {"recursion_limit": 40})
        last = result["messages"][-1]
        content = getattr(last, "content", "") or ""
        return str(content)


def resolve_reward_mode(explicit: Optional[str] = None) -> RewardMode:
    """解析奖励模式：显式参数优先，否则读环境变量 ``MATH_REWARD_MODE``，默认 binary。

    参数:
        explicit: ``LitMathAgent(reward_mode=...)`` 传入的字符串；可为 None。

    抛出:
        ValueError: 模式名非法时，提示合法取值，避免静默落到错误目标。
    """
    raw = (explicit or os.environ.get("MATH_REWARD_MODE") or "binary").strip().lower()
    if raw not in ("binary", "shaped", "format_only"):
        raise ValueError(
            f"Unknown reward mode {raw!r}. Expected binary|shaped|format_only "
            "(set MATH_REWARD_MODE or LitMathAgent(reward_mode=...))."
        )
    return cast(RewardMode, raw)


class LitMathAgent(agl.LitAgent[Dict[str, Any]]):
    """Agent-lightning 训练包装器：一次 rollout = 一题 → 跑图 → emit_reward。

    与 ``MathAgent`` 的分工:
    - MathAgent：纯推理图。
    - LitMathAgent：从 Trainer 注入的 ``resources['main_llm']`` 取 endpoint，
      按 train/val 选温度，打日志，并通过 ``agl.emit_reward`` 上报奖励。
    """

    def __init__(
        self,
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 8,
        max_tokens: int = 1024,
        reward_mode: Optional[str] = None,
    ) -> None:
        """参数:
            trained_agents: 交给基类，用于 adapter 匹配「要训练的 agent span」；span就是单次agent的输入输出这一个过程
                默认 None 表示不过滤。
            val_temperature: 验证时采样温度；默认 0 降低随机性，便于比基线。
            max_turns / max_tokens: 透传给内部 MathAgent。
            reward_mode: 覆盖 ``MATH_REWARD_MODE``；正式训练保持 binary。
        """
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.reward_mode = resolve_reward_mode(reward_mode)

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """Runner 回调：执行单个 episode 并上报 reward。

        参数:
            task: 数据样本 dict，至少含 ``question`` / ``answer``。
            resources: 命名资源；训练时 ``main_llm`` 指向当前策略的 OpenAI endpoint。
            rollout: 框架提供的 rollout 元信息（id、mode=train|val、attempt 等）。

        返回:
            固定返回 ``None``，奖励只通过 ``agl.emit_reward`` 发送。
            （若同时 ``return float`` 会与 emit 重复计奖，故二选一。）

        关键步骤:
            1. 按 train/val 选择温度（train 探索，val 稳定）。
            2. ``llm.get_base_url(rollout_id, attempt_id)`` 拿到本次尝试专用 URL。
            3. 编译并 invoke LangGraph；挂上 langchain tracer 以便 adapter 收集轨迹。
            4. 解析答案 → ``compute_reward`` → ``emit_reward``。
        """
        question = task["question"]
        ground_truth = str(task["answer"])
        start = time.time()
        llm = cast(agl.LLM, resources["main_llm"])
        rollout_id = rollout.rollout_id

        # train：沿用采样参数里的温度（默认 0.7）以增加组内多样性（利好 GRPO）。
        # val：默认 0.0，减少随机，方便对比 checkpoint。
        if rollout.mode == "train":
            temperature = float(llm.sampling_parameters.get("temperature", 0.7))
        else:
            temperature = (
                self.val_temperature
                if self.val_temperature is not None
                else float(llm.sampling_parameters.get("temperature", 0.0))
            )

        endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id) 
        print(f"endpoint: {endpoint}")
        print(f"llm.model: {llm.model}")
        print(f"rollout.rollout_id: {rollout.rollout_id}")
        print(f"rollout.attempt.attempt_id: {rollout.attempt.attempt_id}")
        print(f"temperature: {temperature}")
        print(f"self.max_turns: {self.max_turns}")
        print(f"self.max_tokens: {self.max_tokens}")
        print(f"self.reward_mode: {self.reward_mode}")
        print(f"rollout.mode: {rollout.mode}")
         # type: ignore[arg-type]
        agent = MathAgent(
            endpoint=endpoint,
            model_name=llm.model,
            temperature=temperature,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
        )

        logger.info("[Rollout %s] Question: %s", rollout_id, question)
        prediction: Optional[str] = None
        try:
            # tracer → langchain callback：把 LLM/tool span 记下来供 Adapter 转 triplet。
            handler = self.tracer.get_langchain_handler()
            initial: AgentState = {
                "question": question,
                "num_turns": 0,
                "asked_finalize": False,
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ],
            }
            result_state = agent.graph().invoke(
                initial,
                {"callbacks": [handler] if handler else [], "recursion_limit": 40},
            )
            messages = result_state["messages"]
            prediction = extract_answer_from_messages(messages)
            raw_text = last_assistant_text(messages)
            n_chars = count_assistant_chars(messages)
        except Exception as e:
            # 单条失败不应杀死整个 runner：记 0 分并继续下一条。
            logger.exception("[Rollout %s] Agent failed: %s", rollout_id, e)
            prediction = None
            raw_text = None
            n_chars = 0

        reward = compute_reward(
            prediction,
            ground_truth,
            mode=self.reward_mode,
            raw_assistant_text=raw_text,
            num_assistant_chars=n_chars,
        )
        logger.info(
            "[Rollout %s] mode=%s pred=%r gt=%r reward=%s chars=%d time=%.2fs",
            rollout_id,
            self.reward_mode,
            prediction,
            ground_truth,
            reward,
            n_chars,
            time.time() - start,
        )
        # 只 emit、不 return float，避免双计奖励。
        agl.emit_reward(reward)
        return None


def debug_math_agent() -> None:
    """不启 VERL：用 ``Trainer.dev`` + 外部 OpenAI-compatible endpoint 冒烟 Agent 图。

    环境变量:
        OPENAI_API_BASE / OPENAI_BASE_URL: 必填，推理服务地址。
        OPENAI_API_KEY: 可选。
        OPENAI_MODEL / MODEL: 模型名，默认本地 Qwen3-4B 路径。

    数据: 若存在 ``data/val.parquet`` 则取前 4 条，否则用内置两道简单题。
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "val.parquet")
    if os.path.exists(data_path):
        records = cast(List[Dict[str, Any]], pd.read_parquet(data_path).head(4).to_dict(orient="records"))
    else:
        records = [
            {"id": "debug-1", "question": "What is 12 multiplied by 15?", "answer": "180"},
            {"id": "debug-2", "question": "What is the square root of 256?", "answer": "16"},
        ]

    endpoint = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if not endpoint:
        raise RuntimeError("Set OPENAI_API_BASE (or OPENAI_BASE_URL) to an OpenAI-compatible endpoint.")

    model = os.environ.get("OPENAI_MODEL", os.environ.get("MODEL", "/root/autodl-tmp/LLM/Qwen3-4B"))
    trainer = agl.Trainer(
        n_runners=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=endpoint,
                model=model,
                sampling_parameters={"temperature": 0.3},
            )
        },
    )
    trainer.dev(LitMathAgent(), records)


if __name__ == "__main__":
    debug_math_agent()

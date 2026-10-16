/** Same text as ``DEFAULT_PLANNER_PROMPT`` / ``DEFAULT_VERIFIER_PROMPT`` in mas/workflow/prompts.py. */

export const DEFAULT_PLANNER_PROMPT = (
  '把题目拆成下一步。next 必须是候选 tool-agent 的 id。'
  + '只输出 JSON，键为 next、args、sub_goal、done。'
  + 'done 为 true 且 next 为空时结束。'
);

export const DEFAULT_VERIFIER_PROMPT = (
  '根据工具结果判断子目标是否完成。'
  + '只输出 JSON，键为 ok、reason、step_conclusion、slot_updates、answer。'
  + 'step_conclusion 只能是 COMPLETE 或 INCOMPLETE。'
  + '完成时 answer 给出简短答案。'
);

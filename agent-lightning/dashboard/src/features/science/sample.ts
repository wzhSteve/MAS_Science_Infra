// Copyright (c) Microsoft. All rights reserved.

/** Compact collect-shaped fixture so Science MAS is usable without a live store. */
export const SAMPLE_SCIENCE_PAYLOAD = {
  batch: {
    trajectories: [
      {
        trajectory_id: 'ok1',
        task: { id: 'demo-1', question: 'What is 1+1?', answer: '2', source: 'gsm8k' },
        events: [
          { event_id: 'e1', agent_id: 'hub', kind: 'task_start', payload: {} },
          { event_id: 'e2', agent_id: 'hub', kind: 'tool_call', payload: { tool: 'execute_python' } },
          { event_id: 'e3', agent_id: 'hub', kind: 'final_answer', payload: { answer: '2' } },
        ],
        messages: [],
        final_answer: '2',
        final_reward: 1.0,
        format_ok: true,
        n_search: 0,
        n_python: 1,
        meta: { format_ok: true, n_search: 0, n_python: 1 },
      },
      {
        trajectory_id: 'err1',
        task: { id: 'demo-2', question: 'Who wrote Hamlet?', answer: 'Shakespeare', source: 'hotpotqa' },
        events: [
          { event_id: 'e4', agent_id: 'hub', kind: 'task_start', payload: {} },
          {
            event_id: 'e5',
            agent_id: 'hub',
            kind: 'error',
            payload: { error: 'timeout contacting wiki' },
          },
        ],
        messages: [],
        final_answer: '',
        final_reward: 0.0,
        format_ok: false,
        n_search: 1,
        n_python: 0,
        meta: { format_ok: false, n_search: 1, n_python: 0 },
      },
      {
        trajectory_id: 'hack1',
        task: { id: 'demo-3', question: 'What is 2+2?', answer: '4', source: 'gsm8k' },
        events: [
          { event_id: 'e6', agent_id: 'hub', kind: 'task_start', payload: {} },
          {
            event_id: 'e7',
            agent_id: 'hub',
            kind: 'error',
            payload: { error: 'timeout contacting wiki' },
          },
          { event_id: 'e8', agent_id: 'hub', kind: 'final_answer', payload: { answer: '4' } },
        ],
        messages: [],
        final_answer: '4',
        final_reward: 1.0,
        format_ok: true,
        n_search: 1,
        n_python: 0,
        meta: { format_ok: true, n_search: 1, n_python: 0 },
      },
    ],
    meta: { mean_reward: 0.6667, n_trajectories: 3 },
  },
  train_signal: {
    advantage: { name: 'grpo' },
    loss: { name: 'grpo' },
    meta: {},
  },
};

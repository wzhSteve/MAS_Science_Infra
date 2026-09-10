// Copyright (c) Microsoft. All rights reserved.

import { describe, expect, it } from 'vitest';
import { buildScienceView, parseSciencePayload } from './parse';
import { SAMPLE_SCIENCE_PAYLOAD } from './sample';

describe('science payload parser', () => {
  it('reads collect JSON (batch + train_signal)', () => {
    const parsed = parseSciencePayload(SAMPLE_SCIENCE_PAYLOAD);
    expect(parsed.trajectories).toHaveLength(3);
    expect(parsed.trainSignal.advantage).toBe('grpo');
    expect(parsed.trainSignal.loss).toBe('grpo');
  });

  it('builds a dashboard view with harness hits', () => {
    const { view, error } = buildScienceView(SAMPLE_SCIENCE_PAYLOAD);
    expect(error).toBeNull();
    expect(view).not.toBeNull();
    expect(view?.n).toBe(3);
    expect(view?.nError).toBe(2);
    expect(view?.rewards.map((point) => point.reward)).toEqual([1, 0, 1]);
    const plugins = new Set(view?.hypotheses.map((hyp) => hyp.plugin));
    expect(plugins.has('log_error')).toBe(true);
    expect(plugins.has('reward_hacking')).toBe(true);
    expect(plugins.has('cognitive_convergence')).toBe(true);
    expect(plugins.has('loss_volatility')).toBe(true);
  });

  it('accepts a bare trajectories document', () => {
    const { view, error } = buildScienceView({
      trajectories: [{ trajectory_id: 't1', final_reward: 0.5, final_answer: 'x', format_ok: true, events: [] }],
    });
    expect(error).toBeNull();
    expect(view?.n).toBe(1);
    expect(view?.meanReward).toBe(0.5);
  });

  it('treats null as an empty batch', () => {
    const { view, error } = buildScienceView(null);
    expect(error).toBeNull();
    expect(view?.n).toBe(0);
  });
});

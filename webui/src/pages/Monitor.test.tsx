import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MonitorPanel } from './Monitor';

vi.mock('../app/providers/RuntimeProvider', () => ({
  useRuntimeCommands: () => ({ viewTraining: vi.fn(), startTrain: vi.fn(), stopTrain: vi.fn(), diagnose: vi.fn() }),
  useAgl: () => ({
    data: { ok: false, dashboard_built: true, ui_path: '/agl/metrics' },
    error: null,
    loading: false,
    refresh: async () => {},
  }),
}));

afterEach(() => cleanup());

const props = {
  expId: 'exp',
  model: { experiment_id: 'exp', rewards: [], empty: true },
  error: null,
  loading: false,
  onRefresh: async () => {},
  trainRunId: 'run-1',
  trainRunning: true,
  aglOnline: false,
  onRefreshLog: async () => {},
  visible: true,
};

describe('MonitorPanel log', () => {
  it('shows the current train log and clears it when the log becomes empty', () => {
    const { rerender } = render(<MonitorPanel {...props} trainLog="alpha-line" />);
    expect(screen.getByText('alpha-line')).toBeTruthy();
    rerender(<MonitorPanel {...props} trainLog="" trainRunId="run-2" />);
    expect(screen.queryByText('alpha-line')).toBeNull();
    expect(screen.getByText('尚无训练 stdout')).toBeTruthy();
  });

  it('shows reward and loss charts and a historical AGL link when the store is offline', () => {
    render(<MonitorPanel {...props} trainLog="" model={{
      rewards: [],
      empty: false,
      step_rewards: [
        { index: 1, reward: 0.6, answer_reward: 0.5, correct_reward: 0.1 },
        { index: 2, reward: 0, answer_reward: 0.4, correct_reward: 0.05 },
      ],
      val_rewards: [{ index: 0, reward: 0.4, answer_reward: 0.3, correct_reward: 0.12 }],
      train_answer_rewards: [{ index: 1, reward: 0.5 }, { index: 2, reward: 0.4 }],
      train_correct_rewards: [{ index: 1, reward: 0.1 }, { index: 2, reward: 0.05 }],
      val_answer_rewards: [{ index: 0, reward: 0.3 }],
      val_correct_rewards: [{ index: 0, reward: 0.12 }],
      step_losses: [{ index: 1, pg_loss: 0, entropy_loss: 0.11 }, { index: 2, pg_loss: 0, entropy_loss: 0.14 }],
      metrics_path: '/tmp/metrics.jsonl',
    }} />);
    expect(screen.getByText('训练集 step reward 记录数').parentElement?.textContent).toContain('2');
    expect(screen.getByText('最新 val Ra').parentElement?.textContent).toContain('0.3000');
    expect(screen.getByText('最新 val Rc').parentElement?.textContent).toContain('0.1200');
    expect(screen.getByRole('heading', { name: '训练集 reward（total / Ra / Rc）' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '验证集 reward（total / Ra / Rc）' })).toBeTruthy();
    expect(screen.getByText('当前没有活动训练')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Loss' })).toBeTruthy();
    const link = screen.getAllByRole('link', { name: /AGL Metrics（历史）/ })[0];
    expect(link.getAttribute('href') || '').toMatch(/^\/agl\/metrics/);
  });
});

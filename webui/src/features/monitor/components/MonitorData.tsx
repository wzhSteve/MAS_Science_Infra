import { memo, useMemo } from 'react';
import type { Hypothesis, MonitorResponse, RewardPoint, Trajectory } from '../../../shared/api/types';
import { EmptyState } from '../../../shared/components/EmptyState';
import { Section } from '../../../shared/components/Section';
import { HarnessDiagnostics } from './HarnessDiagnostics';
import { RewardChart } from './RewardChart';
import { TrajectoryTable } from './TrajectoryTable';

const EMPTY_REWARDS: RewardPoint[] = [];
const EMPTY_TRAJECTORIES: Trajectory[] = [];
const EMPTY_HYPOTHESES: Hypothesis[] = [];

export const MonitorSummary = memo(function MonitorSummary({ model }: { model: MonitorResponse | null }) {
  const trainPoints = model?.step_rewards?.length || model?.train_rewards?.length || 0;
  return (
    <dl className="metrics-grid" aria-label="Monitor 概览">
      <div className="metric"><dt className="metric-label">collect n</dt><dd className="metric-value">{model?.n ?? 0}</dd></div>
      <div className="metric"><dt className="metric-label">collect mean</dt><dd className="metric-value">{model?.mean_reward ?? '—'}</dd></div>
      <div className="metric"><dt className="metric-label">train pts</dt><dd className="metric-value">{trainPoints}</dd></div>
      <div className="metric"><dt className="metric-label">errors</dt><dd className="metric-value">{model?.n_error ?? 0}</dd></div>
    </dl>
  );
});

interface MonitorDataProps {
  model: MonitorResponse | null;
  aglOnline: boolean;
  visible: boolean;
}

export const MonitorData = memo(function MonitorData({ model, aglOnline, visible }: MonitorDataProps) {
  const rewards = model?.rewards ?? EMPTY_REWARDS;
  const trainingRewards = model?.step_rewards?.length
    ? model.step_rewards
    : model?.train_rewards ?? EMPTY_REWARDS;
  const collectChart = useMemo(() => rewards.map(({ index, reward }) => ({ i: index, reward })), [rewards]);
  const trainChart = useMemo(() => trainingRewards.map(({ index, reward }) => ({ i: index, reward })), [trainingRewards]);
  const hasCollect = Boolean(model && !model.empty && (model.n ?? 0) > 0);

  return (
    <>
      <Section title="训练 Reward（AGL）">
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : trainChart.length ? (
          <RewardChart data={trainChart} title="训练 Reward（AGL）" color="var(--chart-train)" />
        ) : (
          <EmptyState title={aglOnline ? '训练已连接，尚无训练 Reward' : 'LightningStore 未就绪'}>
            {aglOnline
              ? '训练已连接，但还没有带 final_reward 的 rollout。等第一批 episode 结束即可。'
              : '无 train / LightningStore 未就绪时这里为空。AGL Metrics 能开时会同步到这里。'}
          </EmptyState>
        )}
      </Section>
      <Section title="Collect Reward">
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : hasCollect && collectChart.length ? (
          <RewardChart data={collectChart} title="Collect Reward" color="var(--chart-collect)" />
        ) : (
          <EmptyState title="尚无 Collect Reward">
            尚无 collect.json。请先在 MAS 面板 Collect。
          </EmptyState>
        )}
      </Section>
      {hasCollect ? <TrajectoryTable trajectories={model?.trajectories ?? EMPTY_TRAJECTORIES} /> : null}
      <HarnessDiagnostics hypotheses={model?.hypotheses ?? EMPTY_HYPOTHESES} trainSignal={model?.train_signal} />
    </>
  );
});

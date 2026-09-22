import { memo, useMemo } from 'react';
import type { MonitorResponse, RewardPoint } from '../../../shared/api/types';
import { EmptyState } from '../../../shared/components/EmptyState';
import { Section } from '../../../shared/components/Section';
import { RewardChart } from './RewardChart';

const EMPTY_REWARDS: RewardPoint[] = [];

export const MonitorSummary = memo(function MonitorSummary({ model }: { model: MonitorResponse | null }) {
  const trainPoints = model?.step_rewards?.length || model?.train_rewards?.length || 0;
  return (
    <dl className="metrics-grid" aria-label="Monitor 概览">
      <div className="metric"><dt className="metric-label">训练 Reward 记录数</dt><dd className="metric-value">{trainPoints}</dd></div>
    </dl>
  );
});

interface MonitorDataProps {
  model: MonitorResponse | null;
  aglOnline: boolean;
  visible: boolean;
}

export const MonitorData = memo(function MonitorData({ model, aglOnline, visible }: MonitorDataProps) {
  const trainingRewards = model?.step_rewards?.length
    ? model.step_rewards
    : model?.train_rewards ?? EMPTY_REWARDS;
  const trainChart = useMemo(() => trainingRewards.map(({ index, reward }) => ({ i: index, reward })), [trainingRewards]);

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
    </>
  );
});

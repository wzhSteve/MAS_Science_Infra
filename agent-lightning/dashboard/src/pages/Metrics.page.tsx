// Copyright (c) Microsoft. All rights reserved.

import { useMemo } from 'react';
import { IconChartLine } from '@tabler/icons-react';
import { Alert, Card, Group, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { selectAutoRefreshMs } from '@/features/config';
import { useGetTrainingMetricsQuery } from '@/features/metrics';
import { useAppSelector } from '@/store/hooks';

type MetricRow = Record<string, number | string | boolean | null>;

function pickLossKey(steps: MetricRow[]): string | null {
  const preferred = ['actor/pg_loss', 'actor/policy_loss', 'actor/loss', 'training/loss'];
  for (const key of preferred) {
    if (steps.some((row) => typeof row[key] === 'number')) {
      return key;
    }
  }
  for (const row of steps) {
    for (const key of Object.keys(row)) {
      if (key.toLowerCase().includes('loss') && typeof row[key] === 'number') {
        return key;
      }
    }
  }
  return null;
}

function formatScalar(value: number | string | boolean | null | undefined): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (value == null) {
    return '—';
  }
  return String(value);
}

export function MetricsPage() {
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const { data, isLoading, isError, error, refetch } = useGetTrainingMetricsQuery(
    { rolloutLimit: 300, stepLimit: 2000 },
    { pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined },
  );

  const episodeSeries = useMemo(() => {
    const points = data?.rolloutRewards ?? [];
    return points
      .filter((p) => typeof p.finalReward === 'number')
      .map((p, index) => ({
        index: index + 1,
        rolloutId: p.rolloutId,
        reward: p.finalReward as number,
        mode: p.mode ?? '',
      }));
  }, [data?.rolloutRewards]);

  const stepRewardSeries = useMemo(() => {
    const byStep = new Map<number, { step: number; trainingReward?: number; valReward?: number }>();
    for (const row of data?.trainingSteps ?? []) {
      if (typeof row.step !== 'number') {
        continue;
      }
      const point = byStep.get(row.step) ?? { step: row.step };
      if (typeof row['training/reward'] === 'number') {
        point.trainingReward = row['training/reward'] as number;
      }
      if (typeof row['val/reward'] === 'number') {
        point.valReward = row['val/reward'] as number;
      }
      byStep.set(row.step, point);
    }
    return [...byStep.values()]
      .filter((row) => typeof row.trainingReward === 'number' || typeof row.valReward === 'number')
      .sort((a, b) => a.step - b.step);
  }, [data?.trainingSteps]);

  const hasTrainingReward = stepRewardSeries.some((row) => typeof row.trainingReward === 'number');
  const hasValReward = stepRewardSeries.some((row) => typeof row.valReward === 'number');

  const lossKey = useMemo(() => pickLossKey(data?.trainingSteps ?? []), [data?.trainingSteps]);

  const lossSeries = useMemo(() => {
    if (!lossKey) {
      return [];
    }
    const steps = data?.trainingSteps ?? [];
    return steps
      .filter((row) => typeof row.step === 'number' && typeof row[lossKey] === 'number')
      .map((row) => ({
        step: row.step as number,
        loss: row[lossKey] as number,
      }));
  }, [data?.trainingSteps, lossKey]);

  const latestStep = data?.trainingSteps?.length ? data.trainingSteps[data.trainingSteps.length - 1] : null;
  const latestScalarEntries = useMemo(() => {
    if (!latestStep) {
      return [];
    }
    return Object.entries(latestStep)
      .filter(([key, value]) => key !== 'step' && (typeof value === 'number' || typeof value === 'boolean'))
      .sort(([a], [b]) => a.localeCompare(b));
  }, [latestStep]);

  const errorMessage =
    isError && error && typeof error === 'object' && 'status' in error
      ? `Request failed (${String((error as { status?: unknown }).status)})`
      : isError
        ? 'Failed to load metrics'
        : null;

  return (
    <Stack gap='md'>
      <Group justify='space-between' align='flex-end'>
        <div>
          <Group gap='xs'>
            <IconChartLine size={22} />
            <Title order={2}>Metrics</Title>
          </Group>
          <Text c='dimmed' size='sm' mt={4}>
            Episode rewards come from completed rollouts in the store. Step reward / loss come from metrics.jsonl
            after validation (`val/reward`) and after each GRPO actor update (`training/reward`, `actor/pg_loss`).
          </Text>
        </div>
        <Text size='sm' c='dimmed'>
          {isLoading
            ? 'Loading…'
            : `${episodeSeries.length} rewarded rollouts · ${data?.trainingSteps?.length ?? 0} jsonl rows`}
        </Text>
      </Group>

      {errorMessage && (
        <Alert color='red' title='Metrics unavailable' withCloseButton={false}>
          {errorMessage}.{' '}
          <Text span inherit style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => refetch()}>
            Retry
          </Text>
        </Alert>
      )}

      {!data?.source.exists && (
        <Alert color='yellow' title='No training step metrics yet'>
          `metrics.jsonl` was not found ({data?.source.metricsJsonl ?? 'AGL_METRICS_JSONL unset'}). Episode rewards
          still appear after rollouts finish. After the first GRPO step, curves for `training/reward` and actor loss
          should populate. You can also open TensorBoard on the experiment `tensorboard/` directory.
        </Alert>
      )}

      {data?.source.exists && !hasTrainingReward && hasValReward && (
        <Alert color='blue' title='Validation logged; first GRPO step still running'>
          `val/reward` is already in metrics.jsonl (step {String(latestStep?.step ?? 0)}). `training/reward` and
          `actor/pg_loss` are written only after the first actor update — after the trainer finishes collecting the
          train batch (`train_batch_size × rollout.n` rollouts) and runs GRPO.
        </Alert>
      )}

      <SimpleGrid cols={{ base: 1, lg: 1 }} spacing='md'>
        <Card withBorder padding='md' radius='md'>
          <Title order={4} mb='sm'>
            Episode reward
          </Title>
          <Text c='dimmed' size='xs' mb='sm'>
            Per-rollout `final_reward` from the store (includes validation rollouts).
          </Text>
          {episodeSeries.length === 0 ? (
            <Text c='dimmed' size='sm'>
              No completed rollouts with rewards yet.
            </Text>
          ) : (
            <ResponsiveContainer width='100%' height={280}>
              <LineChart data={episodeSeries}>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey='index' label={{ value: 'Rollout #', position: 'insideBottom', offset: -2 }} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type='monotone' dataKey='reward' name='final_reward' stroke='#228be6' dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card withBorder padding='md' radius='md'>
          <Title order={4} mb='sm'>
            Step reward
          </Title>
          <Text c='dimmed' size='xs' mb='sm'>
            `val/reward` after validation; `training/reward` after each GRPO step.
          </Text>
          {stepRewardSeries.length === 0 ? (
            <Text c='dimmed' size='sm'>
              Waiting for `val/reward` or `training/reward` in metrics.jsonl.
            </Text>
          ) : (
            <ResponsiveContainer width='100%' height={280}>
              <LineChart data={stepRewardSeries}>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey='step' label={{ value: 'global_step', position: 'insideBottom', offset: -2 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                {hasValReward && (
                  <Line
                    type='monotone'
                    dataKey='valReward'
                    name='val/reward'
                    stroke='#fab005'
                    dot
                    connectNulls
                    strokeWidth={2}
                  />
                )}
                {hasTrainingReward && (
                  <Line
                    type='monotone'
                    dataKey='trainingReward'
                    name='training/reward'
                    stroke='#12b886'
                    dot={false}
                    connectNulls
                    strokeWidth={2}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card withBorder padding='md' radius='md'>
          <Title order={4} mb='sm'>
            Loss {lossKey ? `(${lossKey})` : ''}
          </Title>
          {lossSeries.length === 0 ? (
            <Text c='dimmed' size='sm'>
              No actor loss yet. Validation does not write a loss scalar. After the first GRPO actor update,
              `actor/pg_loss` (or similar) will appear here.
            </Text>
          ) : (
            <ResponsiveContainer width='100%' height={280}>
              <LineChart data={lossSeries}>
                <CartesianGrid strokeDasharray='3 3' />
                <XAxis dataKey='step' label={{ value: 'global_step', position: 'insideBottom', offset: -2 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type='monotone' dataKey='loss' name={lossKey ?? 'loss'} stroke='#fa5252' dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {latestScalarEntries.length > 0 && (
          <Card withBorder padding='md' radius='md'>
            <Title order={4} mb='sm'>
              Latest jsonl scalars {latestStep && typeof latestStep.step === 'number' ? `(step ${latestStep.step})` : ''}
            </Title>
            <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing='xs'>
              {latestScalarEntries.map(([key, value]) => (
                <Group key={key} justify='space-between' gap='sm' wrap='nowrap'>
                  <Text size='sm' c='dimmed' style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {key}
                  </Text>
                  <Text size='sm' ff='monospace'>
                    {formatScalar(value)}
                  </Text>
                </Group>
              ))}
            </SimpleGrid>
          </Card>
        )}
      </SimpleGrid>

      {data?.source.metricsJsonl && (
        <Text size='xs' c='dimmed'>
          metrics.jsonl: {data.source.metricsJsonl} ({data.source.exists ? 'found' : 'missing'})
        </Text>
      )}
    </Stack>
  );
}

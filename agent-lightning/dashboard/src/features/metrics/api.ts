// Copyright (c) Microsoft. All rights reserved.

import { rolloutsApi } from '../rollouts/api';

export type RolloutRewardPoint = {
  rolloutId: string;
  mode: string | null;
  status: string;
  finalReward: number | null;
  startTime: number | null;
  endTime: number | null;
};

export type TrainingMetricsResponse = {
  rolloutRewards: RolloutRewardPoint[];
  trainingSteps: Array<Record<string, number | string | boolean | null>>;
  source: {
    metricsJsonl: string | null;
    exists: boolean;
    candidates: string[];
  };
};

const normalizeMetrics = (value: unknown): TrainingMetricsResponse => {
  const raw = (value ?? {}) as Record<string, any>;
  const rolloutRewardsRaw = Array.isArray(raw.rollout_rewards)
    ? raw.rollout_rewards
    : Array.isArray(raw.rolloutRewards)
      ? raw.rolloutRewards
      : [];
  const trainingStepsRaw = Array.isArray(raw.training_steps)
    ? raw.training_steps
    : Array.isArray(raw.trainingSteps)
      ? raw.trainingSteps
      : [];
  const sourceRaw = (raw.source ?? {}) as Record<string, any>;

  return {
    rolloutRewards: rolloutRewardsRaw.map((item: Record<string, any>) => ({
      rolloutId: String(item.rollout_id ?? item.rolloutId ?? ''),
      mode: (item.mode ?? null) as string | null,
      status: String(item.status ?? ''),
      finalReward:
        typeof item.final_reward === 'number'
          ? item.final_reward
          : typeof item.finalReward === 'number'
            ? item.finalReward
            : null,
      startTime:
        typeof item.start_time === 'number'
          ? item.start_time
          : typeof item.startTime === 'number'
            ? item.startTime
            : null,
      endTime:
        typeof item.end_time === 'number' ? item.end_time : typeof item.endTime === 'number' ? item.endTime : null,
    })),
    trainingSteps: trainingStepsRaw as Array<Record<string, number | string | boolean | null>>,
    source: {
      metricsJsonl: (sourceRaw.metrics_jsonl ?? sourceRaw.metricsJsonl ?? null) as string | null,
      exists: Boolean(sourceRaw.exists),
      candidates: Array.isArray(sourceRaw.candidates) ? sourceRaw.candidates.map(String) : [],
    },
  };
};

export const metricsApi = rolloutsApi.injectEndpoints({
  endpoints: (builder) => ({
    getTrainingMetrics: builder.query<TrainingMetricsResponse, { rolloutLimit?: number; stepLimit?: number } | void>({
      query: (args) => {
        const searchParams = new URLSearchParams();
        const rolloutLimit = args && typeof args === 'object' ? args.rolloutLimit : undefined;
        const stepLimit = args && typeof args === 'object' ? args.stepLimit : undefined;
        if (typeof rolloutLimit === 'number') {
          searchParams.set('rollout_limit', String(rolloutLimit));
        }
        if (typeof stepLimit === 'number') {
          searchParams.set('step_limit', String(stepLimit));
        }
        const qs = searchParams.toString();
        return {
          url: qs.length > 0 ? `v1/agl/metrics?${qs}` : 'v1/agl/metrics',
          method: 'GET',
        };
      },
      transformResponse: (response: unknown) => normalizeMetrics(response),
    }),
  }),
});

export const { useGetTrainingMetricsQuery } = metricsApi;

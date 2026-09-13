import { useCallback } from 'react';
import { experimentApi } from '../../experiment/api';
import type { Bundle } from '../../../shared/api/types';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';

export interface ResourceOverview {
  experimentId: string;
  experimentName: string;
  inference: {
    kind: string; model: string; endpoint: string; credentialSource: string;
    credentialSet: boolean; revision?: string;
  };
  training: { modelPath: string; profile: string; agentIds: string[] };
  data: { training: string[]; validation: string[]; unsupportedTraining: boolean; unsupportedValidation: boolean };
}

function strings(value: unknown) {
  if (value == null || value === '') return { paths: [], unsupported: false };
  if (typeof value === 'string') return { paths: [value], unsupported: false };
  if (Array.isArray(value) && value.every(item => typeof item === 'string')) return { paths: value, unsupported: false };
  return { paths: [], unsupported: true };
}

function overview(bundle: Bundle): ResourceOverview {
  const nestedModel = bundle.rl.actor_rollout_ref?.model;
  const nestedPath = nestedModel && typeof nestedModel === 'object' && 'path' in nestedModel && typeof nestedModel.path === 'string'
    ? nestedModel.path : '';
  const training = strings(bundle.rl.data?.train_files);
  const validation = strings(bundle.rl.data?.val_files);
  const declared = bundle.workflow.agents;
  return {
    experimentId: bundle.id, experimentName: bundle.meta.name || bundle.id,
    inference: {
      kind: bundle.llm.kind || 'api', model: bundle.llm.model || '', endpoint: bundle.llm.base_url || '',
      credentialSource: bundle.llm.credential_source || 'none', credentialSet: Boolean(bundle.llm.api_key_set),
      revision: bundle.llm.config_revision,
    },
    training: {
      modelPath: bundle.rl.model_path || nestedPath, profile: bundle.rl.profile || '',
      agentIds: declared?.length ? declared.filter(agent => agent.trainable !== false).map(agent => agent.id)
        : [bundle.workflow.entry_agent || 'hub'],
    },
    data: {
      training: training.paths, validation: validation.paths,
      unsupportedTraining: training.unsupported, unsupportedValidation: validation.unsupported,
    },
  };
}

export function useResourceOverview(experimentId: string | null, active: boolean) {
  const load = useCallback(async (signal: AbortSignal) => {
    if (!experimentId) throw new Error('请先选择所属实验。');
    // Legacy GET initializes missing experiments: only read an ID still present in the list.
    const listing = await experimentApi.listExperiments(signal);
    if (!listing.experiments.includes(experimentId)) throw new Error('所选实验不存在，请重新选择或刷新列表。');
    const bundle = await experimentApi.getExperiment(experimentId, signal);
    if (bundle.id !== experimentId) throw new Error('资源响应与所选实验不匹配。');
    return overview(bundle);
  }, [experimentId]);
  return usePollingResource(`resource-overview:${experimentId}`, load, undefined, active && Boolean(experimentId));
}

import { memo, useCallback, useMemo, useState } from 'react';
import { modelResourcesApi, type LocalModelCandidate, type ModelResource } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Select } from '../../../shared/ui/select';

const LOCAL_PREFIX = '__local__:';

export interface AgentModelOption {
  id: string;
  name: string;
  source: 'local' | 'api';
  trainable: boolean;
  resource?: ModelResource;
  candidate?: LocalModelCandidate;
}

export interface AgentDefaultModel {
  name: string;
  source: 'local' | 'api';
  available: boolean;
  trainable: boolean;
  loaded: boolean;
}

export interface ResolvedAgentModel {
  name: string;
  source: 'local' | 'api' | 'unknown';
  available: boolean;
  trainable: boolean;
  inherited: boolean;
}

export function resolveAgentModel(
  value: string | undefined,
  options: AgentModelOption[],
  defaultModel: AgentDefaultModel,
): ResolvedAgentModel {
  if (!value || value === 'inherit') {
    return {
      name: defaultModel.name,
      source: defaultModel.source,
      available: defaultModel.available,
      trainable: defaultModel.trainable,
      inherited: true,
    };
  }
  const option = options.find(item => item.id === value);
  return option
    ? {
        name: option.name,
        source: option.source,
        available: true,
        trainable: option.trainable,
        inherited: false,
      }
    : {
        name: value,
        source: 'unknown',
        available: false,
        trainable: false,
        inherited: false,
      };
}

async function loadAgentModels(signal: AbortSignal) {
  const first = await modelResourcesApi.list({ limit: 100 }, signal);
  const resources = [...first.items];
  for (let offset = 100; offset < first.total; offset += 100) {
    const page = await modelResourcesApi.list({ offset, limit: 100 }, signal);
    resources.push(...page.items);
  }
  const discovered = await modelResourcesApi.discoverLocal(signal);
  return { resources, discovered: discovered.items };
}

export function useAgentModelOptions(active = true) {
  const load = useCallback((signal: AbortSignal) => loadAgentModels(signal), []);
  const models = usePollingResource('agent-model-options', load, undefined, active);
  const options = useMemo<AgentModelOption[]>(() => {
    const resources = models.data?.resources || [];
    const result: AgentModelOption[] = resources.map(resource => ({
      id: resource.id,
      name: resource.name,
      source: resource.type === 'training' || resource.config.kind === 'local' ? 'local' : 'api',
      trainable: resource.type === 'training',
      resource,
    }));
    const ids = new Set(result.map(option => option.id));
    for (const candidate of models.data?.discovered || []) {
      if (candidate.registered_resource_id && ids.has(candidate.registered_resource_id)) continue;
      result.push({
        id: candidate.registered_resource_id || `${LOCAL_PREFIX}${encodeURIComponent(candidate.path)}`,
        name: candidate.name,
        source: 'local',
        trainable: true,
        candidate,
      });
    }
    return result;
  }, [models.data]);
  return { ...models, options };
}

export const AgentModelSelect = memo(function AgentModelSelect({
  value,
  defaultModel,
  trainable,
  options,
  loading,
  error,
  onRefresh,
  disabled = false,
  onChange,
}: {
  value: string;
  defaultModel: AgentDefaultModel;
  trainable: boolean;
  options: AgentModelOption[];
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<unknown>;
  disabled?: boolean;
  onChange: (model: string) => void;
}) {
  const [registering, setRegistering] = useState(false);
  const [registrationError, setRegistrationError] = useState<string | null>(null);
  const visible = useMemo(() => options.filter(option => !trainable || option.trainable),
    [options, trainable]);
  const selected = value === 'inherit' ? null : options.find(option => option.id === value);
  const effective = resolveAgentModel(value, options, defaultModel);
  const selectedInvalid = value !== 'inherit' && !loading && (!selected || (trainable && !selected.trainable));
  const trainableInvalid = trainable && defaultModel.loaded && (!effective.available || !effective.trainable);

  const select = useCallback(async (next: string) => {
    setRegistrationError(null);
    if (!next || next === 'inherit') {
      onChange('inherit');
      return;
    }
    if (!next.startsWith(LOCAL_PREFIX)) {
      onChange(next);
      return;
    }
    const path = decodeURIComponent(next.slice(LOCAL_PREFIX.length));
    const candidate = options.find(option => option.candidate?.path === path)?.candidate;
    if (!candidate) {
      setRegistrationError('本地模型列表已变化，请刷新后重试。');
      return;
    }
    setRegistering(true);
    try {
      const resource = await modelResourcesApi.ensureLocal({
        name: candidate.name,
        model_path: candidate.path,
      });
      onChange(resource.id);
      await onRefresh();
    } catch (error) {
      setRegistrationError(error instanceof Error ? error.message : String(error));
    } finally {
      setRegistering(false);
    }
  }, [onChange, onRefresh, options]);

  return <div className="training-model-select">
    <Select value={value} disabled={disabled || registering || loading}
      onChange={event => void select(event.target.value)}>
      <option value="inherit">继承实验默认模型 · {defaultModel.name}</option>
      {selectedInvalid && <option value={value}>
        {selected?.name || value} · {selected && !selected.trainable ? '不可训练' : '不可用'}
      </option>}
      {visible.map(option => <option key={option.id} value={option.id}>
        {option.name} · {option.source === 'api' ? 'API' : option.trainable ? '本地 · 可训练' : '本地'}
      </option>)}
    </Select>
    {trainableInvalid && <InlineNotice tone="warning">
      {effective.available
        ? `${effective.name} 没有可训练权重，请先选择本地可训练模型。`
        : `${effective.inherited ? '实验默认模型' : effective.name} 不可用，请先选择本地可训练模型。`}
    </InlineNotice>}
    {(error || registrationError) &&
      <InlineNotice tone="warning">{registrationError || error}</InlineNotice>}
  </div>;
});

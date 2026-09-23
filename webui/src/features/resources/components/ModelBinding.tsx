import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { modelResourcesApi, type LocalModelCandidate, type ModelBindings, type ModelResource, type ModelResourceType } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { Section } from '../../../shared/components/Section';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useConfirm } from '../../../shared/feedback/useConfirm';

export interface BindingView {
  loaded: boolean;
  bound: boolean;
  resource: ModelResource | null;
  error: string | null;
}
const UNLOADED: BindingView = { loaded: false, bound: false, resource: null, error: null };

export const ModelBinding = memo(function ModelBinding({
  experimentId, purpose, active, onReload, onManage, onState, legacyDirty = false, suggestedId, onSuggestionApplied, children, saveInHeader = false, compact = false, fallbackName,
}: {
  experimentId: string; purpose: ModelResourceType; active: boolean; onReload: () => void;
  onManage: () => void; onState?: (state: BindingView) => void; legacyDirty?: boolean;
  suggestedId?: string; onSuggestionApplied?: () => void; children?: ReactNode;
  saveInHeader?: boolean;
  compact?: boolean; fallbackName?: string;
}) {
  const confirm = useConfirm();
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const loadBindings = useCallback((signal: AbortSignal) => modelResourcesApi.bindings(experimentId, signal), [experimentId]);
  const remote = usePollingResource(`bindings:${experimentId}:${purpose}`, loadBindings, undefined, active);
  const loadList = useCallback(async (signal: AbortSignal) => {
    if (!compact) return modelResourcesApi.list({ type: purpose, q: search, offset, limit: 20 }, signal);
    const first = await modelResourcesApi.list({ type: purpose, limit: 100 }, signal);
    const items = [...first.items];
    for (let next = 100; next < first.total; next += 100) {
      const page = await modelResourcesApi.list({ type: purpose, offset: next, limit: 100 }, signal);
      items.push(...page.items);
    }
    return { ...first, items };
  }, [purpose, search, offset, compact]);
  const list = usePollingResource(`binding-options:${purpose}:${search}:${offset}`, loadList, undefined, active);
  const loadLocalModels = useCallback((signal: AbortSignal) => modelResourcesApi.discoverLocal(signal), []);
  const localModels = usePollingResource('local-training-models', loadLocalModels, undefined, active && purpose === 'training' && !compact);
  const [draft, setDraft] = useState<{ base: ModelBindings; selected: string | null } | null>(null);
  const action = useAction();
  const handledSuggestion = useRef<string>();
  const mounted = useRef(true);
  const latestDraft = useRef(draft);
  latestDraft.current = draft;
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    if (!remote.data) return;
    setDraft(previous => {
      if (previous && previous.selected !== previous.base[purpose].resource_id) return previous;
      return { base: remote.data!, selected: remote.data![purpose].resource_id };
    });
  }, [remote.data, purpose]);
  useEffect(() => {
    if (!suggestedId) { handledSuggestion.current = undefined; return; }
    if (!active || !suggestedId || !draft || handledSuggestion.current === suggestedId) return;
    handledSuggestion.current = suggestedId;
    setDraft(previous => previous && ({ ...previous, selected: suggestedId }));
    onSuggestionApplied?.();
  }, [active, suggestedId, draft, onSuggestionApplied]);
  const dirty = Boolean(draft && draft.selected !== draft.base[purpose].resource_id);
  const saved = draft?.base[purpose];
  const selected = list.data?.items.find(item => item.id === draft?.selected)
    || (draft?.selected === saved?.resource_id ? saved?.resource : null);
  const loadSelected = useCallback((signal: AbortSignal) => {
    if (!draft?.selected) throw new Error('未选择资源');
    return modelResourcesApi.get(draft.selected, signal);
  }, [draft?.selected]);
  const selectedResource = usePollingResource(`binding-selected:${draft?.selected || ''}`, loadSelected, undefined,
    active && Boolean(draft?.selected) && !selected);
  const candidate = selected || (selectedResource.data?.id === draft?.selected ? selectedResource.data : null);
  const candidateValid = !draft?.selected || candidate?.type === purpose;
  const error = remote.error || saved?.error || null;
  useEffect(() => {
    onState?.(draft ? {
      loaded: true, bound: Boolean(compact ? draft.selected : saved?.resource_id), resource: compact ? candidate : saved?.resource || null, error,
    } : { ...UNLOADED, loaded: !remote.loading, error: remote.error });
  }, [draft?.base, compact, compact ? draft?.selected : undefined, compact ? candidate : undefined, purpose, onState, error, remote.error, remote.loading]);

  const save = useCallback(async () => {
    const submitted = latestDraft.current;
    if (!submitted || legacyDirty || !candidateValid) return false;
    if (!submitted.selected && submitted.base[purpose].resource_id && !(await confirm({
      title: '解除模型引用？',
      description: '将恢复本实验原先保存的配置，不会复制个人资源的当前值或密钥。',
      confirmLabel: '解除引用',
      tone: 'danger',
    }))) return false;
    const success = await action.run('bind', async () => {
      const current = await modelResourcesApi.bindings(experimentId);
      if (current[purpose].resource_id !== submitted.base[purpose].resource_id) throw new Error('此用途的绑定已被修改，请刷新后重新选择。');
      const result = await modelResourcesApi.bind(experimentId, current.revision, purpose, submitted.selected);
      if (!mounted.current) return;
      setDraft(previous => ({
        base: result, selected: previous?.selected === submitted.selected ? result[purpose].resource_id : previous?.selected ?? null,
      }));
      onReload();
      return '模型绑定已保存';
    });
    return success;
  }, [confirm, experimentId, purpose, legacyDirty, candidateValid, action.run, onReload]);
  useUnsavedChanges(`model-binding-${purpose}`, {
    label: purpose === 'inference' ? '默认推理模型绑定' : '训练模型来源绑定',
    resource: `model-binding-${purpose}`, dirty, busy: action.pending !== null, save,
  });
  const reload = async () => {
    if (action.pending !== null) return;
    if (dirty && !(await confirm({
      title: '重新载入模型绑定？',
      description: '当前未保存的资源选择将被放弃，其他配置草稿不会改变。',
      confirmLabel: '放弃并载入',
      tone: 'danger',
    }))) return;
    await action.run('reload', async () => {
      const result = await modelResourcesApi.bindings(experimentId);
      if (!mounted.current) return;
      setDraft({ base: result, selected: result[purpose].resource_id });
      void list.refresh();
      return '已重新载入模型绑定。';
    });
  };
  const registerLocal = useCallback(async (candidate: LocalModelCandidate) => {
    const submitted = latestDraft.current;
    if (!submitted || purpose !== 'training' || legacyDirty || submitted.selected !== submitted.base[purpose].resource_id) return;
    await action.run('register-local', async () => {
      const { resource, bindings } = await modelResourcesApi.registerLocal(experimentId, {
        revision: submitted.base.revision,
        name: candidate.name,
        model_path: candidate.path,
      });
      if (!mounted.current) return;
      setDraft({ base: bindings, selected: bindings.training.resource_id });
      await list.refresh();
      onReload();
      return `已绑定本地模型 ${resource.name}`;
    });
  }, [action.run, experimentId, list.refresh, onReload, purpose, legacyDirty]);

  if (compact) return <div className="model-resource-choice">
    <FormField label={purpose === 'training' ? '初始权重' : '默认推理模型'}>
      <Select value={draft?.selected || ''} disabled={!draft || action.pending !== null} onChange={event => {
        if (event.target.value === '__manage__') { onManage(); return; }
        setDraft(previous => previous && ({ ...previous, selected: event.target.value || null }));
      }}>
        <option value="">{saved?.resource_id ? '原有模型配置' : fallbackName || '选择已有模型'}</option>
        {draft?.selected && !list.data?.items.some(item => item.id === draft.selected) &&
          <option value={draft.selected}>{candidate?.name || draft.selected}</option>}
        {list.data?.items.map(resource => <option key={resource.id} value={resource.id}>{resource.name}</option>)}
        <option value="__manage__">在模型与数据中管理…</option>
      </Select>
    </FormField>
    {(error || list.error || selectedResource.error) && <InlineNotice tone="danger">
      {error || list.error || selectedResource.error}
      <Button size="sm" onClick={() => {
        void remote.refresh(); void list.refresh();
        if (draft?.selected && !selected) void selectedResource.refresh();
      }}>重试</Button>
    </InlineNotice>}
    {draft?.selected && !candidateValid && <p className="field-hint">所选模型尚未就绪，无法保存。</p>}
  </div>;

  return <Section title={purpose === 'inference' ? '默认推理模型' : '训练模型来源'} actions={
    <StatusBadge tone={dirty ? 'warning' : 'neutral'}>{action.pending ? '处理中' : !draft ? '读取绑定中' : dirty ? '绑定未保存' : saved?.resource_id ? '个人资源' : '实验内配置'}</StatusBadge>
  }>
    {remote.error && <InlineNotice tone="danger">绑定读取失败：{remote.error}</InlineNotice>}
    {saved?.error && <InlineNotice tone="danger">{saved.error}，不会自动切换模型。</InlineNotice>}
    {(search || query || offset > 0 || (list.data?.total || 0) > 20) && <div className="model-binding-search">
      <Input aria-label="搜索模型资源" placeholder="按资源名称搜索…" value={query} onChange={event => setQuery(event.target.value)} />
      <Button size="sm" onClick={() => { setSearch(query.trim()); setOffset(0); }}>搜索</Button>
    </div>}
    <FormField label="选择模型">
      <Select value={draft?.selected || ''} disabled={!draft || action.pending !== null} onChange={event =>
        setDraft(previous => previous && ({ ...previous, selected: event.target.value || null }))}>
        <option value="">使用实验已保存配置</option>
        {draft?.selected && !list.data?.items.some(item => item.id === draft.selected) &&
          <option value={draft.selected}>{candidate?.name || draft.selected}</option>}
        {list.data?.items.map(resource => <option key={resource.id} value={resource.id}>{resource.name}</option>)}
      </Select>
    </FormField>
    {list.error && <InlineNotice tone="warning">资源列表读取失败：{list.error}</InlineNotice>}
    {list.data && list.data.total > 20 && <div className="model-binding-pages">
      <Button size="sm" disabled={offset === 0} onClick={() => setOffset(value => Math.max(0, value - 20))}>上一页</Button>
      <span className="field-hint">{offset + 1}–{Math.min(offset + 20, list.data.total)} / {list.data.total}</span>
      <Button size="sm" disabled={offset + 20 >= list.data.total} onClick={() => setOffset(value => value + 20)}>下一页</Button>
    </div>}
    {draft?.selected && !candidate && <p className="field-hint">{selectedResource.error || '正在读取所选资源…'}</p>}
    {candidate && <div className="model-binding-summary">
      <p>{purpose === 'inference' ? `${candidate.config.model || '未命名模型'} · ${candidate.config.base_url || ''}` : candidate.config.model_path}</p>
    </div>}
    {purpose === 'training' && localModels.data?.items.length ? <div className="local-model-candidates">
      <p className="field-hint">服务器检测到的本地模型</p>
      {localModels.data.items.map(model => <div className="local-model-candidate" key={model.path}>
        <div><strong>{model.name}</strong><span className="mono">{model.path}</span>
          <small>{model.architectures.join(', ') || model.model_type || 'Hugging Face model'}</small></div>
        <Button size="sm" disabled={!draft || dirty || legacyDirty || action.pending !== null} loading={action.pending === 'register-local'}
          onClick={() => void registerLocal(model)}>
          {model.registered_resource_id ? '绑定' : '登记并绑定'}
        </Button>
      </div>)}
    </div> : null}
    {purpose === 'training' && localModels.error && <InlineNotice tone="warning">本地模型发现失败：{localModels.error}</InlineNotice>}
    {!candidateValid && draft?.selected && <InlineNotice tone="warning">所选资源尚未读取或类型不适用，不能保存绑定。</InlineNotice>}
    {legacyDirty && !saveInHeader && <InlineNotice tone="warning">实验配置还有未保存修改，请先保存后再切换模型来源。</InlineNotice>}
    <div className="action-bar">
      {!saveInHeader && <Button size="sm" variant="primary" loading={action.pending === 'bind'} disabled={!dirty || action.pending !== null || legacyDirty || !candidateValid} onClick={() => void save()}>保存模型绑定</Button>}
      <Button size="sm" variant="ghost" onClick={onManage}>管理模型</Button>
      <Button size="sm" variant="ghost" disabled={action.pending !== null} onClick={() => void reload()}>刷新</Button>
    </div>
    {children}
  </Section>;
});

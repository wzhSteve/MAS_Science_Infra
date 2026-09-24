import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight, LocateFixed, Maximize2, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import type { Bundle, MetaResponse, Palette, SamplingPreviewResponse, WorkflowSpec } from '../../../shared/api/types';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { HYDRA_FIELDS } from '../../rl/model/hydraFields';
import { RlField } from '../../rl/components/RlField';
import { ModelBinding, type BindingView } from '../../resources/components/ModelBinding';
import { SamplingSettings } from '../../sampling/components/SamplingSettings';
import { workflowToFlow } from '../../mas/model/workflowGraph';
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';
import { DatasetSelection } from '../../resources/components/DatasetSelection';
import type { SettingsSection } from '../model/sections';
import type { CanvasMode } from '../../mas/types';
import {
  resolveAgentModel,
  type AgentDefaultModel,
  type AgentModelOption,
} from '../../resources/components/AgentModelSelect';

const BASIC_DATA = HYDRA_FIELDS.filter(field => field.path === 'data.train_files' || field.path === 'data.val_files');
const ENVIRONMENT = HYDRA_FIELDS.filter(field => field.group === 'Rollout' || field.path === 'trainer.nnodes');
const OPTIMIZATION = HYDRA_FIELDS.filter(field => !BASIC_DATA.includes(field) && !ENVIRONMENT.includes(field));

export const TrainingSettings = memo(function TrainingSettings({
  bundle, meta, onReload, active, section, jump, expanded, onExpandedChange, onManageModels, onManageData,
  selectedResource, resourcePurpose, onSuggestionApplied, workflow, palette, onWorkflowChange, onLocate, onDemo,
  canvasMode, onCanvasModeChange, samplingPreview, samplingPreviewError, selectedSamplingOpportunity, onSelectSamplingOpportunity,
  onTrainingModelState, modelOptions,
}: {
  bundle: Bundle; meta: MetaResponse | null; onReload: () => void; active: boolean;
  section: SettingsSection | null; jump: number; expanded: boolean; onExpandedChange: (expanded: boolean) => void;
  onManageModels: () => void; onManageData: () => void; selectedResource?: string; resourcePurpose?: 'training' | 'inference';
  onSuggestionApplied: () => void; onDemo: () => void;
  workflow: WorkflowSpec; palette: Palette; onWorkflowChange: (workflow: WorkflowSpec) => void; onLocate: (agentId: string) => void;
  onCanvasModeChange: (mode: CanvasMode) => void;
  canvasMode: CanvasMode; samplingPreview: SamplingPreviewResponse | null; samplingPreviewError: string | null;
  selectedSamplingOpportunity: string | null; onSelectSamplingOpportunity: (id: string | null) => void;
  onTrainingModelState?: (state: BindingView) => void;
  modelOptions: AgentModelOption[];
}) {
  const { draft, gpu, selectedIds } = useTrainingConfig();
  const { rl, patch, pending } = draft;
  const root = useRef<HTMLDivElement>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [binding, setBinding] = useState<BindingView>({ loaded: false, bound: false, resource: null, error: null });
  const agents = useMemo(() => workflowToFlow(workflow).nodes.filter(node =>
    node.type === 'agent'), [workflow]);
  const scrollToSection = useCallback((id: string) => {
    requestAnimationFrame(() => {
      const element = root.current;
      const target = element?.querySelector<HTMLElement>(`#training-${id}`);
      if (element && target) element.scrollTo({ top: target.getBoundingClientRect().top - element.getBoundingClientRect().top + element.scrollTop - 16 });
    });
  }, []);
  useEffect(() => {
    if (!active || !section || section === 'diagnostics') return;
    setCollapsed(false);
    scrollToSection(section === 'inference' ? 'model' : section);
  }, [active, section, jump, scrollToSection]);
  const legacyModel = rl.model_path || rl.actor_rollout_ref?.model?.path || '';
  const defaultModelName = binding.resource?.name || legacyModel.split(/[\\/]/).pop() || '未选择';
  const defaultModel = useMemo<AgentDefaultModel>(() => ({
    name: defaultModelName,
    source: 'local',
    available: Boolean(binding.resource || legacyModel),
    trainable: Boolean(binding.resource || legacyModel),
    loaded: binding.loaded,
  }), [binding.loaded, binding.resource, defaultModelName, legacyModel]);
  const updateBinding = useCallback((state: BindingView) => {
    setBinding(state);
    onTrainingModelState?.(state);
  }, [onTrainingModelState]);
  const summaries = useMemo(() => agents.map(node => ({
    node,
    model: resolveAgentModel(node.data.model, modelOptions, defaultModel),
  })), [agents, defaultModel, modelOptions]);
  const fields = (items: typeof HYDRA_FIELDS) => <div className="form-grid">{items.map(field =>
    <RlField key={field.path} field={field} value={field.read(rl)} onPatch={patch} compact />)}</div>;

  return <aside className={`training-parameters${expanded ? ' is-expanded' : collapsed ? ' is-collapsed' : ''}`}
    aria-label="训练配置" hidden={canvasMode === 'sampling'}>
    <header className="parameter-panel-heading">
      {collapsed && !expanded ? <Button size="sm" variant="ghost" title="展开训练配置" aria-label="展开训练配置" onClick={() => setCollapsed(false)}>
        <PanelLeftOpen size={16} />
      </Button> : <>
        <h2>训练配置</h2>
        <Button size="sm" variant="ghost" title={expanded ? '返回画布' : '展开编辑'} aria-label={expanded ? '返回画布' : '展开编辑'}
          onClick={() => onExpandedChange(!expanded)}>{expanded ? '返回画布' : <Maximize2 size={14} />}</Button>
        {!expanded && <Button size="sm" variant="ghost" title="收起训练配置" aria-label="收起训练配置" onClick={() => setCollapsed(true)}><PanelLeftClose size={15} /></Button>}
      </>}
    </header>
    <div ref={root} className="parameter-panel-scroll" hidden={collapsed && !expanded}>
      {draft.notice?.tone === 'danger' && <InlineNotice tone="danger">{draft.notice.message}</InlineNotice>}
      <section id="training-model" className="parameter-group">
        <h3><span>01</span>模型与训练范围</h3>
        <ModelBinding experimentId={bundle.id} purpose="training" active={active} compact saveInHeader
          label="实验默认模型"
          fallbackName={legacyModel.split(/[\\/]/).pop()} onReload={onReload} onManage={onManageModels}
          legacyDirty={draft.dirty || pending !== null} suggestedId={resourcePurpose === 'training' ? selectedResource : undefined}
          onSuggestionApplied={onSuggestionApplied} onState={updateBinding} />
        {agents.length > 0 && <div className="training-agent-summary" aria-label="Agent 模型与训练状态">
          <div className="training-agent-summary__heading">
            <span>Agent 模型与训练状态</span>
            <small>{summaries.filter(({ node }) => node.data.trainable !== false).length} 个参与训练</small>
          </div>
          {summaries.map(({ node, model }) => {
            const invalid = !model.available || (node.data.trainable !== false && !model.trainable);
            return <button type="button" key={node.id} className={invalid ? 'is-invalid' : undefined}
            onClick={() => onLocate(node.id)} title={`配置 ${node.id}`}>
            <span>{node.id}</span>
            <span>{model.name} · {model.inherited ? '继承' : model.source === 'api' ? 'API' : model.source === 'local' ? '本地' : '不可用'}</span>
            <span className={node.data.trainable !== false ? 'is-trainable' : ''}>
              {invalid ? '需配置' : node.data.trainable !== false ? '参与训练' : '已冻结'}
            </span>
            <LocateFixed size={12} />
          </button>;
          })}
        </div>}
        {import.meta.env.DEV && <details className="parameter-details"><summary>开发工具</summary>
          <Button size="sm" onClick={onDemo}>模拟调试 · Mock</Button></details>}
      </section>
      <section id="training-data" className="parameter-group">
        <h3><span>02</span>训练数据<Button size="sm" variant="ghost" aria-label="打开数据资源" title="数据资源" onClick={onManageData}><ArrowUpRight size={13} /></Button></h3>
        <DatasetSelection active={active} onManage={onManageData} />
      </section>
      <section id="training-training" className="parameter-group">
        <h3><span>03</span>训练策略</h3>
        <SamplingSettings workflow={workflow} palette={palette} onChange={onWorkflowChange} onLocate={onLocate}
          compact onOpenDesign={() => onCanvasModeChange('sampling')} canvasMode={canvasMode}
          preview={samplingPreview} previewError={samplingPreviewError}
          selectedOpportunityId={selectedSamplingOpportunity} onSelectOpportunity={onSelectSamplingOpportunity} />
        <details className="parameter-details"><summary>训练与优化参数</summary>{fields(OPTIMIZATION)}</details>
      </section>
      <section id="training-environment" className="parameter-group">
        <h3><span>04</span>执行资源</h3>
        <div className="parameter-summary-row"><span>GPU</span><span>{selectedIds.join(', ')}</span></div>
        <FormField label="并行 Runner">
          <Input type="number" min={1} step={1} value={rl.n_runners ?? ''}
            onChange={event => patch(current => ({ ...current, n_runners: event.target.value === '' ? undefined : Number(event.target.value) }))} />
        </FormField>
        <details className="parameter-details"><summary>显存与并行参数</summary>
          <FormField label="运行档位">
            <Select value={rl.profile || 'fast'} onChange={event => patch(current => ({ ...current, profile: event.target.value }))}>
              {Array.from(new Set([rl.profile || 'fast', ...(meta?.profiles || ['fast'])])).map(profile => <option key={profile} value={profile}>{profile}</option>)}
            </Select>
          </FormField>
          {fields(ENVIRONMENT)}
          <Button size="sm" disabled={pending !== null || !gpu.data?.gpus.length} onClick={draft.recommend}>资源建议</Button>
        </details>
      </section>
    </div>
  </aside>;
});

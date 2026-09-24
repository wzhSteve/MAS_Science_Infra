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
import { workflowToFlow, KNOWN_TOOLS } from '../../mas/model/workflowGraph';
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';
import { InferenceSettings } from './InferenceSettings';
import { DatasetSelection } from '../../resources/components/DatasetSelection';
import type { SettingsSection } from '../model/sections';
import type { CanvasMode } from '../../mas/types';

const BASIC_DATA = HYDRA_FIELDS.filter(field => field.path === 'data.train_files' || field.path === 'data.val_files');
const ENVIRONMENT = HYDRA_FIELDS.filter(field => field.group === 'Rollout' || field.path === 'trainer.nnodes');
const OPTIMIZATION = HYDRA_FIELDS.filter(field => !BASIC_DATA.includes(field) && !ENVIRONMENT.includes(field));

export const TrainingSettings = memo(function TrainingSettings({
  bundle, meta, onReload, active, section, jump, expanded, onExpandedChange, onManageModels, onManageData,
  selectedResource, resourcePurpose, onSuggestionApplied, workflow, palette, onWorkflowChange, onLocate, onDemo,
  canvasMode, samplingPreview, samplingPreviewError, selectedSamplingOpportunity, onSelectSamplingOpportunity,
}: {
  bundle: Bundle; meta: MetaResponse | null; onReload: () => void; active: boolean;
  section: SettingsSection | null; jump: number; expanded: boolean; onExpandedChange: (expanded: boolean) => void;
  onManageModels: () => void; onManageData: () => void; selectedResource?: string; resourcePurpose?: 'training' | 'inference';
  onSuggestionApplied: () => void; onDemo: () => void;
  workflow: WorkflowSpec; palette: Palette; onWorkflowChange: (workflow: WorkflowSpec) => void; onLocate: (agentId: string) => void;
  canvasMode: CanvasMode; samplingPreview: SamplingPreviewResponse | null; samplingPreviewError: string | null;
  selectedSamplingOpportunity: string | null; onSelectSamplingOpportunity: (id: string) => void;
}) {
  const { draft, gpu, selectedIds } = useTrainingConfig();
  const { rl, patch, pending } = draft;
  const root = useRef<HTMLDivElement>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [executionOpen, setExecutionOpen] = useState(false);
  const [binding, setBinding] = useState<BindingView>({ loaded: false, bound: false, resource: null, error: null });
  const agents = useMemo(() => workflowToFlow(workflow).nodes.filter(node =>
    node.type === 'agent' && node.data.kind !== 'tool' && !KNOWN_TOOLS.includes(node.id)), [workflow]);
  const trainable = useMemo(() => agents.filter(node => node.data.trainable), [agents]);
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
    if (section === 'inference') setExecutionOpen(true);
    scrollToSection(section === 'inference' ? 'model' : section);
  }, [active, section, jump, scrollToSection]);
  useEffect(() => {
    if (canvasMode !== 'sampling') return;
    setCollapsed(false);
    scrollToSection('training');
  }, [canvasMode, scrollToSection]);
  const expandBranches = useCallback(() => {
    onExpandedChange(true);
    scrollToSection('training');
  }, [onExpandedChange, scrollToSection]);
  const legacyModel = rl.model_path || rl.actor_rollout_ref?.model?.path || '';
  const fields = (items: typeof HYDRA_FIELDS) => <div className="form-grid">{items.map(field =>
    <RlField key={field.path} field={field} value={field.read(rl)} onPatch={patch} compact />)}</div>;

  return <aside className={`training-parameters${expanded ? ' is-expanded' : collapsed ? ' is-collapsed' : ''}`} aria-label="训练配置">
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
        <h3><span>01</span>训练对象</h3>
        <div className="parameter-agent-list"><span>Agent</span>
          <div>{trainable.map(node => <button type="button" key={node.id} onClick={() => onLocate(node.id)} title={`定位 ${node.id}`}>
            {node.id}<LocateFixed size={12} />
          </button>)}</div>
        </div>
        {!trainable.length && <InlineNotice tone="warning">未指定训练对象，后端将使用入口 {workflow.entry_agent || 'hub'}。</InlineNotice>}
        {trainable.length > 1 && <InlineNotice tone="warning">当前后端仅训练首个可训练 Agent，实际对象以启动检查为准。</InlineNotice>}
        <ModelBinding experimentId={bundle.id} purpose="training" active={active} compact saveInHeader
          fallbackName={legacyModel.split(/[\\/]/).pop()} onReload={onReload} onManage={onManageModels} onState={setBinding}
          legacyDirty={draft.dirty || pending !== null} suggestedId={resourcePurpose === 'training' ? selectedResource : undefined} onSuggestionApplied={onSuggestionApplied} />
        {binding.loaded && !binding.bound && !binding.error && <details className="parameter-details">
          <summary>现有权重路径</summary>
          <FormField label="模型路径"><Input value={legacyModel} onChange={event => patch(current => ({
            ...current, model_path: event.target.value,
            actor_rollout_ref: { ...current.actor_rollout_ref, model: { ...current.actor_rollout_ref?.model, path: event.target.value } },
          }))} /></FormField>
        </details>}
        <details className="parameter-details" open={executionOpen} onToggle={event => setExecutionOpen(event.currentTarget.open)}>
          <summary>执行模型</summary>
          <InferenceSettings bundle={bundle} active={active && executionOpen} view="all" compact onReload={onReload} onManage={onManageModels}
            suggestedId={resourcePurpose === 'inference' ? selectedResource : undefined} onSuggestionApplied={onSuggestionApplied} />
          {agents.filter(node => !node.data.trainable).map(node => <div className="parameter-execution-agent" key={node.id}>
            <span>{node.id}</span><span>{node.data.model === 'inherit' ? '默认推理模型' : node.data.model}</span>
            <Button size="sm" variant="ghost" aria-label={`配置 ${node.id}`} onClick={() => onLocate(node.id)}><LocateFixed size={12} /></Button>
          </div>)}
          {import.meta.env.DEV && <details className="parameter-details"><summary>开发工具</summary>
            <Button size="sm" onClick={onDemo}>模拟调试 · Mock</Button></details>}
        </details>
      </section>
      <section id="training-data" className="parameter-group">
        <h3><span>02</span>训练数据<Button size="sm" variant="ghost" aria-label="打开数据资源" title="数据资源" onClick={onManageData}><ArrowUpRight size={13} /></Button></h3>
        <DatasetSelection active={active} onManage={onManageData} />
      </section>
      <section id="training-training" className="parameter-group">
        <h3><span>03</span>训练策略</h3>
        <SamplingSettings workflow={workflow} palette={palette} onChange={onWorkflowChange} onLocate={onLocate}
          compact expanded={expanded} onExpand={expandBranches} canvasMode={canvasMode}
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

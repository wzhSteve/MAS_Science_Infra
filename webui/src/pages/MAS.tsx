import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { Bundle, MetaResponse } from '../shared/api/types';
import { MasGraphEditor, useMasDraft } from '../features/mas';
import { WorkflowDebug } from '../features/mas/components/WorkflowDebug';
import { executableInfo } from '../features/mas/model/workflowGraph';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { Button } from '../shared/ui/button';
import { useModelReadiness } from '../features/mas/model/useModelReadiness';
import type { TraceFocusRequest } from '../features/mas/types';
import { TrainingSettings } from '../features/settings/components/TrainingSettings';
import { HarnessPanel } from '../features/harness/components/HarnessPanel';
import type { SettingsSection } from '../features/settings/model/sections';
import type { ResourceCategory } from '../app/navigation';
import { useSamplingPreview } from '../features/sampling/model/useSamplingPreview';
import type { CanvasMode } from '../features/mas/types';
import type { BindingView } from '../features/resources/components/ModelBinding';
import {
  useAgentModelOptions,
  type AgentDefaultModel,
} from '../features/resources/components/AgentModelSelect';

export type MASPanelProps = {
  expId: string; bundle: Bundle | null; onReload: () => void; meta: MetaResponse | null;
  requestedSettings: SettingsSection | null; active?: boolean;
  onWorkspace: () => void; onResources: (category: ResourceCategory) => void; onSettings: (section: SettingsSection) => void;
  selectedResource?: string; resourcePurpose?: 'inference' | 'training';
  trainingJump: number;
};

function RetainedView({ active, children, className }: { active: boolean; children: ReactNode; className?: string }) {
  const [visited, setVisited] = useState(active);
  useEffect(() => { if (active) setVisited(true); }, [active]);
  return <div className={className} hidden={!active}>{(active || visited) && children}</div>;
}

export const MASPanel = memo(function MASPanel(props: MASPanelProps) {
  const draft = useMasDraft(props);
  const active = props.active ?? true;
  const section = props.requestedSettings;
  const [expanded, setExpanded] = useState(false);
  const [canvasMode, setCanvasMode] = useState<CanvasMode>('workflow');
  const [selectedSamplingOpportunity, setSelectedSamplingOpportunity] = useState<string | null>(null);
  const workflowExpanded = useRef(false);
  const [modelRequest, setModelRequest] = useState(0);
  const editing = active && section !== 'diagnostics';
  const workflowVisible = editing && !expanded;
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugVisited, setDebugVisited] = useState(false);
  const [debugMode, setDebugMode] = useState<'live' | 'mock'>('live');
  const readiness = useModelReadiness(props.expId, props.bundle?.llm.config_revision, workflowVisible && debugOpen);
  const [traceFocus, setTraceFocus] = useState<TraceFocusRequest | null>(null);
  const [defaultModel, setDefaultModel] = useState<AgentDefaultModel>({
    name: '读取中',
    source: 'local',
    available: false,
    trainable: false,
    loaded: false,
  });
  const modelOptions = useAgentModelOptions(editing);
  const focusSequence = useRef(0);
  const configureModel = useCallback(() => {
    setModelRequest(value => value + 1);
    props.onSettings('inference');
  }, [props.onSettings]);
  const modelResources = useCallback(() => props.onResources('models'), [props.onResources]);
  const dataResources = useCallback(() => props.onResources('datasets'), [props.onResources]);
  const consumeResourceSelection = useCallback(() => props.onSettings(section || 'model'), [props.onSettings, section]);
  const openDebug = useCallback(() => {
    setLibraryOpen(false); setDebugMode('live'); setDebugVisited(true); setDebugOpen(true);
  }, []);
  const closeDebug = useCallback(() => setDebugOpen(false), []);
  const openDemo = useCallback(() => {
    props.onWorkspace(); setExpanded(false); setLibraryOpen(false); setDebugMode('mock'); setDebugVisited(true); setDebugOpen(true);
  }, [props.onWorkspace]);
  const locate = useCallback((target: Omit<TraceFocusRequest, 'token'>) => {
    setExpanded(false);
    setDebugOpen(false);
    setTraceFocus({ ...target, token: ++focusSequence.current });
  }, []);
  const locateAgent = useCallback((agentId: string) => {
    locate({ agentId }); props.onWorkspace();
  }, [locate, props.onWorkspace]);
  const captureTrainingModel = useCallback((state: BindingView) => {
    const legacyPath = props.bundle?.rl.model_path || props.bundle?.rl.actor_rollout_ref?.model?.path || '';
    const available = Boolean(state.resource || legacyPath);
    setDefaultModel({
      name: state.resource?.name || legacyPath.split(/[\\/]/).pop() || (state.loaded ? '未选择' : '读取中'),
      source: 'local',
      available,
      trainable: available,
      loaded: state.loaded,
    });
  }, [props.bundle?.rl.actor_rollout_ref?.model?.path, props.bundle?.rl.model_path]);
  const { workflow } = draft;
  const samplingPreview = useSamplingPreview(workflow, editing);
  const executable = useMemo(() => workflow ? executableInfo(workflow) : { ok: false, reason: '加载中…' }, [workflow]);
  const selectCanvasMode = useCallback((mode: CanvasMode) => {
    setCanvasMode(mode);
    setLibraryOpen(false);
    setDebugOpen(false);
    if (mode === 'workflow') {
      setSelectedSamplingOpportunity(null);
      if (workflowExpanded.current) {
        setExpanded(true);
        workflowExpanded.current = false;
      }
    }
    if (mode === 'sampling') {
      workflowExpanded.current = expanded;
      setExpanded(false);
      props.onWorkspace();
    }
  }, [expanded, props.onWorkspace]);
  if (!workflow || !props.bundle) return <LoadingState label="加载工作流…" />;
  return <div className="mas-workspace mas-workspace--parameters">
    {(draft.paletteError || draft.saveError) && <div className="mas-workspace-notice">
      <InlineNotice tone="danger">{draft.paletteError ? <>节点库加载失败：{draft.paletteError}
        <Button size="sm" onClick={() => void draft.loadPalette()}>重试</Button>
      </> : draft.saveError}</InlineNotice>
    </div>}
    <div className={`parameter-panel-host${expanded ? ' is-expanded' : ''}`} hidden={!editing}>
      <TrainingSettings bundle={props.bundle} meta={props.meta} onReload={props.onReload} active={editing}
        section={section} jump={props.trainingJump + modelRequest} expanded={expanded} onExpandedChange={setExpanded}
        onManageModels={modelResources} onManageData={dataResources} selectedResource={props.selectedResource} resourcePurpose={props.resourcePurpose}
        onSuggestionApplied={consumeResourceSelection} workflow={workflow} palette={draft.palette}
        onWorkflowChange={draft.onWorkflowChange} onLocate={locateAgent} onDemo={openDemo}
        onTrainingModelState={captureTrainingModel}
        modelOptions={modelOptions.options}
        onCanvasModeChange={selectCanvasMode}
        canvasMode={canvasMode} samplingPreview={samplingPreview.data} samplingPreviewError={samplingPreview.error}
        selectedSamplingOpportunity={selectedSamplingOpportunity}
        onSelectSamplingOpportunity={setSelectedSamplingOpportunity} />
    </div>
    <div id="workflow-view" className="workflow-view" hidden={!workflowVisible} role="region" aria-label="工作流">
      <div className="workflow-canvas-column">
        <MasGraphEditor workflow={workflow} palette={draft.palette} onChange={draft.onWorkflowChange}
          active={workflowVisible} libraryOpen={libraryOpen} onLibraryOpenChange={setLibraryOpen} traceFocus={traceFocus}
          inspectorVisible={!debugOpen} onInspect={closeDebug} onModelResources={modelResources}
          defaultModel={defaultModel}
          modelOptions={modelOptions.options} modelOptionsLoading={modelOptions.loading}
          modelOptionsError={modelOptions.error} onRefreshModelOptions={modelOptions.refresh}
          mode={canvasMode} samplingPreview={samplingPreview.data}
          selectedSamplingOpportunity={selectedSamplingOpportunity}
          onSelectSamplingOpportunity={setSelectedSamplingOpportunity}
          onModeChange={selectCanvasMode} onDebug={debugOpen ? closeDebug : openDebug} debugOpen={debugOpen} />
      </div>
      {debugVisited && <WorkflowDebug experimentId={props.expId} draft={draft} active={workflowVisible}
        open={debugOpen} mode={debugMode} executable={executable} readiness={readiness}
        onClose={closeDebug} onLive={openDebug} onConfigureModel={configureModel} onLocate={locate} />}
    </div>
    <RetainedView active={active && section === 'diagnostics'} className="workspace-document-scroll">
      <div className="workspace-document"><h1>诊断工具</h1>
        <HarnessPanel expId={props.expId} bundle={props.bundle} meta={props.meta} onReload={props.onReload} embedded />
      </div>
    </RetainedView>
  </div>;
});

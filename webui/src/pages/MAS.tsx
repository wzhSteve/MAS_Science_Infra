import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { FlaskConical, Plus } from 'lucide-react';
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
  const [modelRequest, setModelRequest] = useState(0);
  const editing = active && section !== 'diagnostics';
  const workflowVisible = editing && !expanded;
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugVisited, setDebugVisited] = useState(false);
  const [debugMode, setDebugMode] = useState<'live' | 'mock'>('live');
  const readiness = useModelReadiness(props.expId, props.bundle?.llm.config_revision, workflowVisible && debugOpen);
  const [traceFocus, setTraceFocus] = useState<TraceFocusRequest | null>(null);
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
  const { workflow } = draft;
  const executable = useMemo(() => workflow ? executableInfo(workflow) : { ok: false, reason: '加载中…' }, [workflow]);
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
        onWorkflowChange={draft.onWorkflowChange} onLocate={locateAgent} onDemo={openDemo} />
    </div>
    <div id="workflow-view" className="workflow-view" hidden={!workflowVisible} role="region" aria-label="工作流">
      <div className="workflow-canvas-column">
        <div className="workflow-floating-tools">
          <div><Button size="sm" aria-expanded={libraryOpen} aria-controls="mas-node-library" onClick={() => { setDebugOpen(false); setLibraryOpen(value => !value); }}>
            <Plus size={15} />添加节点
          </Button></div>
          <div>
            <Button id="workflow-debug-trigger" size="sm" aria-expanded={debugOpen} aria-controls="workflow-debug" onClick={debugOpen ? closeDebug : openDebug}>
              <FlaskConical size={14} />单题调试
            </Button>
          </div>
        </div>
        <MasGraphEditor workflow={workflow} palette={draft.palette} onChange={draft.onWorkflowChange}
          active={workflowVisible} libraryOpen={libraryOpen} onLibraryOpenChange={setLibraryOpen} traceFocus={traceFocus}
          inspectorVisible={!debugOpen} onInspect={closeDebug} onModelResources={modelResources} />
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

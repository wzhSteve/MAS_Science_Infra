import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Bundle, MetaResponse } from '../shared/api/types';
import { MasGraphEditor, WorkflowBar, RunConsole, useMasDraft, type ConsoleTab, type EditorPanel } from '../features/mas';
import { executableInfo } from '../features/mas/model/workflowGraph';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { Button } from '../shared/ui/button';
import { useModelReadiness } from '../features/mas/model/useModelReadiness';
import type { TraceFocusRequest } from '../features/mas/types';
import { ExperimentSettings } from '../features/settings/components/ExperimentSettings';
import type { SettingsSection } from '../features/settings/model/sections';
import type { ResourceCategory } from '../app/navigation';

export type MASPanelProps = {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  meta: MetaResponse | null;
  requestedSettings: SettingsSection | null;
  active?: boolean;
  onWorkspace: () => void;
  onResources: (category: ResourceCategory) => void;
  onSettings: (section: SettingsSection) => void;
  selectedResource?: string;
};

export function MASPanel(props: MASPanelProps) {
  return <MASWorkspace key={props.expId} {...props} />;
}

function MASWorkspace(props: MASPanelProps) {
  const draft = useMasDraft(props);
  const readiness = useModelReadiness(props.expId, props.bundle?.llm.config_revision, props.active ?? true);
  const [panel, setPanel] = useState<EditorPanel>(props.requestedSettings ? 'settings' : null);
  const [settingsSection, setSettingsSection] = useState<SettingsSection>(props.requestedSettings || 'model');
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [consoleTab, setConsoleTab] = useState<ConsoleTab>('config');
  const [consoleMode, setConsoleMode] = useState<'rollout' | 'collect' | 'demo'>('rollout');
  const [historyRequest, setHistoryRequest] = useState(0);
  const [traceFocus, setTraceFocus] = useState<TraceFocusRequest | null>(null);
  const focusSequence = useRef(0);
  const lastRequested = useRef(props.requestedSettings);
  const settingsTrigger = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!props.active || lastRequested.current === props.requestedSettings) return;
    lastRequested.current = props.requestedSettings;
    if (props.requestedSettings) {
      setSettingsSection(props.requestedSettings);
      setPanel('settings');
    } else setPanel(null);
  }, [props.active, props.requestedSettings]);
  const onPanelChange = useCallback((next: EditorPanel) => {
    setPanel(next);
    if (!next && props.requestedSettings) props.onWorkspace();
  }, [props.requestedSettings, props.onWorkspace]);
  const openSettings = useCallback((section: SettingsSection) => {
    settingsTrigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setSettingsSection(section);
    setPanel('settings');
    props.onSettings(section);
  }, [props.onSettings]);
  const changeSettingsSection = useCallback((section: SettingsSection) => {
    setSettingsSection(section);
    props.onSettings(section);
  }, [props.onSettings]);
  const closeSettings = useCallback(() => {
    onPanelChange(null);
    const target = settingsTrigger.current?.isConnected ? settingsTrigger.current : document.getElementById('mas-settings-trigger');
    target?.focus({ preventScroll: true });
  }, [onPanelChange]);
  const configureModel = useCallback(() => openSettings('model'), [openSettings]);
  const modelResources = useCallback(() => props.onResources('models'), [props.onResources]);
  const consumeResourceSelection = useCallback(() => {
    props.onSettings(props.requestedSettings || 'model');
  }, [props.onSettings, props.requestedSettings]);
  const openCollect = useCallback(() => {
    onPanelChange(null);
    setConsoleMode('collect');
    setConsoleTab('config');
    setConsoleOpen(true);
  }, [onPanelChange]);
  const openDebug = useCallback(() => {
    onPanelChange(null);
    setConsoleMode('rollout');
    setConsoleTab('config');
    setConsoleOpen(true);
  }, [onPanelChange]);
  const openDemo = useCallback(() => {
    onPanelChange(null);
    setConsoleMode('demo');
    setConsoleTab('config');
    setConsoleOpen(true);
  }, [onPanelChange]);
  const openHistory = useCallback(() => {
    onPanelChange(null);
    setConsoleMode('rollout');
    setConsoleTab('results');
    setConsoleOpen(true);
    setHistoryRequest(value => value + 1);
  }, [onPanelChange]);
  const { workflow } = draft;
  const executable = useMemo(() => workflow ? executableInfo(workflow) : { ok: false, reason: '加载中…' }, [workflow]);
  const settings = useMemo(() => props.bundle && <ExperimentSettings bundle={props.bundle} meta={props.meta} onReload={props.onReload}
    open={panel === 'settings'} active={props.active ?? true} section={settingsSection} onSectionChange={changeSettingsSection}
    onClose={closeSettings} onCollect={openCollect} onDemo={openDemo} onResources={props.onResources} readiness={readiness.data} readinessError={readiness.error}
    selectedResource={props.selectedResource} suggestedPurpose={props.requestedSettings === 'model' ? 'inference' : 'training'} onSuggestionApplied={consumeResourceSelection} />,
  [props.bundle, props.meta, props.onReload, props.active, props.onResources, props.selectedResource, props.requestedSettings, consumeResourceSelection, panel, settingsSection, changeSettingsSection, closeSettings, openCollect, openDemo, readiness.data, readiness.error]);
  const locate = useCallback((target: Omit<TraceFocusRequest, 'token'>) => {
    setTraceFocus({ ...target, token: ++focusSequence.current });
  }, []);
  if (!workflow) return <LoadingState label="加载 workflow…" />;

  return <div className="mas-workspace">
    <WorkflowBar expId={props.expId} dirty={draft.workflowDirty} pending={draft.pending}
      saving={draft.savingWorkflow} saveError={Boolean(draft.saveError)}
      panel={panel} onSettings={openSettings} onCloseSettings={closeSettings} onSave={draft.save}
      libraryOpen={libraryOpen} onOpenLibrary={() => setLibraryOpen(true)}
      onDebug={openDebug} onHistory={openHistory} />
    {(draft.paletteError || draft.saveError) && <div className="mas-workspace-notice">
      <InlineNotice tone="danger">{draft.paletteError ? <>节点库加载失败：{draft.paletteError}
        <Button size="sm" onClick={() => void draft.loadPalette()}>重试</Button>
      </> : draft.saveError}</InlineNotice>
    </div>}
    <MasGraphEditor workflow={workflow} palette={draft.palette} onChange={draft.onWorkflowChange}
      active={props.active ?? true} panel={panel} onPanelChange={onPanelChange}
      libraryOpen={libraryOpen} onLibraryOpenChange={setLibraryOpen}
      traceFocus={traceFocus}
      settingsContent={settings} onModelResources={modelResources} />
    <RunConsole draft={draft} experimentId={props.expId} active={props.active ?? true} open={consoleOpen} onOpenChange={setConsoleOpen}
      tab={consoleTab} onTabChange={setConsoleTab} executable={executable}
      readiness={readiness} onConfigureModel={configureModel}
      mode={consoleMode} onModeChange={setConsoleMode} historyRequest={historyRequest}
      onLocate={locate} />
  </div>;
}

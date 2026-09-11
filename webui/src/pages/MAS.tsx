import { useMemo, useState } from 'react';
import type { Bundle, RlConfig } from '../shared/api/types';
import { MasGraphEditor, WorkflowBar, RunConsole, useMasDraft, type RlSettingsSlot, type ConsoleTab, type EditorPanel } from '../features/mas';
import { executableInfo } from '../features/mas/model/workflowGraph';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { Button } from '../shared/ui/button';

export type MASPanelProps = {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  normalizeRl: (rl: RlConfig) => RlConfig;
  renderRlSettings?: RlSettingsSlot;
  active?: boolean;
};

export function MASPanel(props: MASPanelProps) {
  const draft = useMasDraft(props);
  const [panel, setPanel] = useState<EditorPanel>(null);
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [consoleTab, setConsoleTab] = useState<ConsoleTab>('config');
  const { workflow } = draft;
  const executable = useMemo(() => workflow ? executableInfo(workflow) : { ok: false, reason: '加载中…' }, [workflow]);
  const rlSettings = useMemo(() => props.renderRlSettings?.({
    rl: draft.rl, onPatch: draft.onRlPatch, onSave: draft.saveRl, saving: draft.savingRl,
  }), [props.renderRlSettings, draft.rl, draft.onRlPatch, draft.saveRl, draft.savingRl]);
  if (!workflow) return <LoadingState label="加载 workflow…" />;

  return <div className="mas-workspace">
    <WorkflowBar expId={props.expId} dirty={draft.workflowDirty} pending={draft.pending}
      saving={draft.savingWorkflow} saveError={Boolean(draft.saveError)} rlDirty={draft.rlDirty}
      panel={panel} onPanelChange={setPanel} onSave={draft.save}
      libraryOpen={libraryOpen} onOpenLibrary={() => setLibraryOpen(true)}
      onRun={() => { setConsoleOpen(true); setConsoleTab('config'); }} />
    {(draft.paletteError || draft.saveError) && <div className="mas-workspace-notice">
      <InlineNotice tone="danger">{draft.paletteError ? <>节点库加载失败：{draft.paletteError}
        <Button size="sm" onClick={() => void draft.loadPalette()}>重试</Button>
      </> : draft.saveError}</InlineNotice>
    </div>}
    <MasGraphEditor workflow={workflow} palette={draft.palette} onChange={draft.onWorkflowChange}
      active={props.active ?? true} panel={panel} onPanelChange={setPanel}
      libraryOpen={libraryOpen} onLibraryOpenChange={setLibraryOpen}
      rlSettings={rlSettings} rlDirty={draft.rlDirty} rlNotice={draft.rlNotice} />
    <RunConsole draft={draft} open={consoleOpen} onOpenChange={setConsoleOpen}
      tab={consoleTab} onTabChange={setConsoleTab} executable={executable} />
  </div>;
}

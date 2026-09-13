import { memo, useEffect, useState, type ReactNode } from 'react';
import type { Bundle, MetaResponse } from '../../shared/api/types';
import { ExperimentPanel } from '../../pages/Experiment';
import { MASPanel } from '../../pages/MAS';
import { MonitorPanel } from '../../pages/Monitor';
import { useAgl, useMonitor, useTraining } from '../providers/RuntimeProvider';
import { isCanvasPanel, type PanelId, type ResourceCategory } from '../navigation';
import type { SettingsSection } from '../../features/settings/model/sections';

const Experiment = memo(ExperimentPanel);
const Mas = memo(MASPanel);

function RetainedPanel({ active, id, label, children }: {
  active: boolean; id: PanelId; label: string; children: ReactNode;
}) {
  const [visited, setVisited] = useState(active);
  useEffect(() => { if (active) setVisited(true); }, [active]);
  return <div id={`panel-${id}`} hidden={!active} role="region" aria-label={`${label} 面板`}>
    {(active || visited) && children}
  </div>;
}

const MonitorConnection = memo(function MonitorConnection({ expId, visible }: { expId: string; visible: boolean }) {
  const { data, refresh } = useTraining();
  const agl = useAgl();
  const monitor = useMonitor();
  return <MonitorPanel expId={expId} visible={visible}
    model={monitor.data} error={monitor.error} loading={monitor.loading} onRefresh={monitor.refresh}
    trainRunId={data?.runId ?? null} trainRunning={data?.running ?? false} trainLog={data?.log ?? ''}
    aglOnline={!!agl.data?.ok && !agl.error} onRefreshLog={refresh} />;
});

export const WorkspacePanels = memo(function WorkspacePanels({ active, visible, bundle, meta, onReload, setExpId, onWorkspace, settings, onResources, onSettings, selectedResource }: {
  active: PanelId; visible: boolean; bundle: Bundle; meta: MetaResponse | null; onReload: () => Promise<void>; setExpId: (id: string) => void;
  onWorkspace: () => void;
  settings?: SettingsSection; onResources: (category: ResourceCategory) => void;
  onSettings: (section: SettingsSection) => void;
  selectedResource?: string;
}) {
  const common = { expId: bundle.id, bundle, onReload };
  const canvas = isCanvasPanel(active);
  const requestedSettings = settings || (active === 'llm' ? 'model' : active === 'rl' ? 'training' : active === 'harness' ? 'diagnostics' : null);
  return <div className={`workspace-content${canvas ? ' workspace-content--mas' : ''}`} id="workspace-content" tabIndex={-1}>
    <RetainedPanel id="mas" label="MAS" active={visible && canvas}>
      <Mas key={bundle.id} {...common} meta={meta} active={visible && canvas} requestedSettings={requestedSettings}
        onWorkspace={onWorkspace} onResources={onResources} onSettings={onSettings} selectedResource={selectedResource} />
    </RetainedPanel>
    <RetainedPanel id="experiment" label="实验配置" active={visible && active === 'experiment'}>
      <Experiment {...common} setExpId={setExpId} />
    </RetainedPanel>
    <RetainedPanel id="monitor" label="实验监控" active={visible && active === 'monitor'}>
      <MonitorConnection expId={bundle.id} visible={visible && active === 'monitor'} />
    </RetainedPanel>
  </div>;
});

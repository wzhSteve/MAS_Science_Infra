import { memo, type ComponentProps } from 'react';
import type { Bundle, MetaResponse } from '../../shared/api/types';
import { ExperimentPanel } from '../../pages/Experiment';
import { LLMPanel } from '../../pages/LLM';
import { MASPanel } from '../../pages/MAS';
import { RLPanel } from '../../pages/RL';
import { HarnessPanel } from '../../pages/Harness';
import { MonitorPanel } from '../../pages/Monitor';
import { RlQuickSettings } from '../../features/rl/components/RlQuickSettings';
import { normalizeRlPayload } from '../../features/rl/model/normalizeRlPayload';
import { useAgl, useMonitor, useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { NAVIGATION, type PanelId } from '../navigation';

const renderRlSettings = (props: ComponentProps<typeof RlQuickSettings>) => <RlQuickSettings {...props} />;
const Experiment = memo(ExperimentPanel);
const Llm = memo(LLMPanel);
const Mas = memo(MASPanel);
const Harness = memo(HarnessPanel);

const RlConnection = memo(function RlConnection({ bundle, meta, onReload }: {
  bundle: Bundle; meta: MetaResponse | null; onReload: () => Promise<void>;
}) {
  const { data, refresh } = useTraining();
  const agl = useAgl();
  const { startTrain, stopTrain } = useRuntimeCommands();
  return <RLPanel expId={bundle.id} bundle={bundle} meta={meta} onReload={onReload}
    trainRunId={data?.runId ?? null} trainRunning={data?.running ?? false} trainLog={data?.log ?? ''}
    aglOnline={!!agl.data?.ok && !agl.error} onStartTrain={startTrain} onStopTrain={stopTrain} onRefreshLog={refresh} />;
});

const MonitorConnection = memo(function MonitorConnection({ expId, visible }: { expId: string; visible: boolean }) {
  const { data, refresh } = useTraining();
  const agl = useAgl();
  const monitor = useMonitor();
  return <MonitorPanel expId={expId} visible={visible}
    model={monitor.data} error={monitor.error} loading={monitor.loading} onRefresh={monitor.refresh}
    trainRunId={data?.runId ?? null} trainRunning={data?.running ?? false} trainLog={data?.log ?? ''}
    aglOnline={!!agl.data?.ok && !agl.error} onRefreshLog={refresh} />;
});

export const WorkspacePanels = memo(function WorkspacePanels({ active, bundle, meta, onReload, setExpId }: {
  active: PanelId; bundle: Bundle; meta: MetaResponse | null; onReload: () => Promise<void>; setExpId: (id: string) => void;
}) {
  const common = { expId: bundle.id, bundle, onReload };
  return <div className={`workspace-content${active === 'mas' ? ' workspace-content--mas' : ''}`} id="workspace-content" tabIndex={-1}>
    {NAVIGATION.map(({ id, label }) => <div key={id} id={`panel-${id}`} hidden={active !== id} role="region" aria-label={`${label} 面板`}>
      {id === 'experiment' && <Experiment {...common} setExpId={setExpId} />}
      {id === 'llm' && <Llm {...common} />}
      {id === 'mas' && <Mas key={bundle.id} {...common} active={active === 'mas'} renderRlSettings={renderRlSettings} normalizeRl={normalizeRlPayload} />}
      {id === 'rl' && <RlConnection bundle={bundle} meta={meta} onReload={onReload} />}
      {id === 'harness' && <Harness {...common} meta={meta} />}
      {id === 'monitor' && <MonitorConnection expId={bundle.id} visible={active === 'monitor'} />}
    </div>)}
  </div>;
});

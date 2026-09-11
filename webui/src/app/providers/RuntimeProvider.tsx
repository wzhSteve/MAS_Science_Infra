import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { masApi } from '../../features/mas/api';
import { harnessApi } from '../../features/harness/api';
import { rlApi } from '../../features/rl/api';
import { monitorApi } from '../../features/monitor/api';
import { runtimeApi } from '../../features/runtime/api';
import { useTrainingRun, type TrainingSnapshot } from '../../features/runtime/model/useTrainingRun';
import { useExperimentEvents } from '../../features/runtime/model/useExperimentEvents';
import { usePollingResource, type Resource } from '../../shared/hooks/usePollingResource';
import { useAction } from '../../shared/hooks/useAction';
import { useTrainConfirmation } from '../../shared/ui/alert-dialog';
import type { AglHealth, MonitorResponse } from '../../shared/api/types';
import type { Notice } from '../../shared/components/InlineNotice';

interface RuntimeCommands {
  startTrain: () => Promise<void>;
  stopTrain: () => Promise<void>;
  collect: () => Promise<void>;
  diagnose: () => Promise<void>;
}
interface CommandStatus { pending: string | null; notice: Notice | null; status: string }
const CommandsContext = createContext<RuntimeCommands | null>(null);
const CommandStatusContext = createContext<CommandStatus | null>(null);
const TrainingContext = createContext<Resource<TrainingSnapshot> | null>(null);
const AglContext = createContext<Resource<AglHealth> | null>(null);
const MonitorContext = createContext<Resource<MonitorResponse> | null>(null);
const EventsContext = createContext<{ latest: string; error: string | null } | null>(null);

function required<T>(value: T | null): T {
  if (value === null) throw new Error('RuntimeProvider is required');
  return value;
}
export const useRuntimeCommands = () => required(useContext(CommandsContext));
export const useCommandStatus = () => required(useContext(CommandStatusContext));
export const useTraining = () => required(useContext(TrainingContext));
export const useAgl = () => required(useContext(AglContext));
export const useMonitor = () => required(useContext(MonitorContext));
export const useRuntimeEvents = () => required(useContext(EventsContext));

export function RuntimeProvider({ expId, onReload, children }: {
  expId: string; onReload: () => Promise<void>; children: ReactNode;
}) {
  const training = useTrainingRun(expId);
  const agl = usePollingResource('agl', runtimeApi.aglHealth, 5000);
  const monitorLoad = useCallback((signal: AbortSignal) => monitorApi.monitor(expId, signal), [expId]);
  const monitor = usePollingResource(`monitor:${expId}`, monitorLoad, 4000);
  const events = useExperimentEvents(expId, training.refresh, monitor.refresh);
  const { run, pending, notice } = useAction();
  const stopAction = useAction();
  const { confirm, dialog } = useTrainConfirmation();
  const [status, setStatus] = useState('idle');
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  const startTrain = useCallback(async () => {
    if (!(await confirm()) || !alive.current) return;
    await run('train', async () => {
      await rlApi.train(expId, { stop_llm: true, confirm_gpu: true });
      if (!alive.current) return;
      setStatus('train_started');
      await training.refresh();
      await monitor.refresh();
      return '训练已启动';
    });
  }, [confirm, expId, run, training.refresh, monitor.refresh]);
  const stopTrain = useCallback(async () => {
    await stopAction.run('stop', async () => {
      await rlApi.trainStop(expId);
      if (!alive.current) return;
      setStatus('train_stopped');
      await training.refresh();
      await monitor.refresh();
      return '已发送停止指令';
    });
  }, [expId, stopAction.run, training.refresh, monitor.refresh]);
  const collect = useCallback(async () => {
    await run('collect', async () => {
      await masApi.collect(expId, { mock: true, n: 1 });
      if (!alive.current) return;
      setStatus('collect_done');
      await onReload();
      await monitor.refresh();
      return 'Mock 采集完成';
    });
  }, [expId, run, onReload, monitor.refresh]);
  const diagnose = useCallback(async () => {
    await run('diagnose', async () => {
      await harnessApi.diagnose(expId);
      if (!alive.current) return;
      setStatus('diagnose_done');
      await onReload();
      await monitor.refresh();
      return '诊断完成';
    });
  }, [expId, run, onReload, monitor.refresh]);
  const commands = useMemo(() => ({ startTrain, stopTrain, collect, diagnose }), [startTrain, stopTrain, collect, diagnose]);
  const commandStatus = useMemo(() => ({
    pending, notice: stopAction.notice?.tone === 'danger' ? stopAction.notice : notice,
    status: events.latest || status,
  }), [pending, notice, stopAction.notice, status, events.latest]);

  return <CommandsContext.Provider value={commands}>
    <CommandStatusContext.Provider value={commandStatus}>
      <TrainingContext.Provider value={training}>
        <AglContext.Provider value={agl}>
          <MonitorContext.Provider value={monitor}>
            <EventsContext.Provider value={events}>{children}{dialog}</EventsContext.Provider>
          </MonitorContext.Provider>
        </AglContext.Provider>
      </TrainingContext.Provider>
    </CommandStatusContext.Provider>
  </CommandsContext.Provider>;
}

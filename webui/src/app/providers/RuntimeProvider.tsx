import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { harnessApi } from '../../features/harness/api';
import { rlApi } from '../../features/rl/api';
import { monitorApi } from '../../features/monitor/api';
import { runtimeApi } from '../../features/runtime/api';
import { trainingApi } from '../../features/training/api';
import { useTrainingRun, type TrainingSnapshot } from '../../features/runtime/model/useTrainingRun';
import { useExperimentEvents, type ExperimentEvents } from '../../features/runtime/model/useExperimentEvents';
import { usePollingResource, type Resource } from '../../shared/hooks/usePollingResource';
import { useAction } from '../../shared/hooks/useAction';
import { useTrainConfirmation } from '../../shared/ui/alert-dialog';
import type { AglHealth, MonitorResponse } from '../../shared/api/types';
import { DraftRegistryContext, useUnsavedChanges } from '../../shared/hooks/useUnsavedChanges';
import { ApiError, errorMessage } from '../../shared/api/http';
import type { SettingsSection } from '../../features/settings/model/sections';
import { isExperimentDraft, saveExperimentDrafts } from '../../features/experiment/model/saveExperiment';
import { useConfirm } from '../../shared/feedback/useConfirm';
import { useNotify } from '../../shared/feedback/useNotify';

interface RuntimeCommands {
  startTrain: () => Promise<void>;
  stopTrain: (runId: string) => Promise<void>;
  viewTraining: (runId?: string) => void;
  diagnose: () => Promise<void>;
}
interface CommandStatus { pending: string | null; status: string }
const CommandsContext = createContext<RuntimeCommands | null>(null);
const CommandStatusContext = createContext<CommandStatus | null>(null);
const TrainingContext = createContext<Resource<TrainingSnapshot> | null>(null);
const AglContext = createContext<Resource<AglHealth> | null>(null);
const MonitorContext = createContext<Resource<MonitorResponse> | null>(null);
const EventsContext = createContext<ExperimentEvents | null>(null);
const TrainingViewContext = createContext(0);

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
export const useTrainingViewRequest = () => useContext(TrainingViewContext);

export function RuntimeProvider({ expId, onReload, active = true, children, onViewTraining, onConfigure }: {
  expId: string; onReload: () => Promise<void>; active?: boolean; children: ReactNode;
  onViewTraining: (runId?: string) => void;
  onConfigure: (section: SettingsSection) => void;
}) {
  const registry = useContext(DraftRegistryContext);
  const confirmAction = useConfirm();
  const notify = useNotify();
  const [viewRequest, setViewRequest] = useState(0);
  const viewTraining = useCallback((runId?: string) => {
    setViewRequest(value => value + 1);
    onViewTraining(runId);
  }, [onViewTraining]);
  const training = useTrainingRun(expId, active);
  const agl = usePollingResource('agl', runtimeApi.aglHealth, 5000, active);
  const monitorLoad = useCallback((signal: AbortSignal) => monitorApi.monitor(expId, signal), [expId]);
  const monitor = usePollingResource(`monitor:${expId}`, monitorLoad, 4000, active);
  const events = useExperimentEvents(expId, training.refresh, monitor.refresh, active);
  const { run, pending } = useAction();
  const stopAction = useAction();
  const { confirm, dialog } = useTrainConfirmation(onConfigure);
  const [status, setStatus] = useState('idle');
  useUnsavedChanges('runtime-operation', {
    label: '实验执行操作', resource: 'runtime', dirty: false, busy: pending !== null || stopAction.pending !== null,
  });
  const alive = useRef(true);
  const pendingTrainRequest = useRef<Parameters<typeof rlApi.createRun>[1] | null>(null);
  const previousTraining = useRef<TrainingSnapshot | null>(null);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    if (pendingTrainRequest.current && training.data?.requestId === pendingTrainRequest.current.request_id && training.data.runId) {
      pendingTrainRequest.current = null;
      viewTraining(training.data.runId);
    }
  }, [training.data?.requestId, training.data?.runId, viewTraining]);
  useEffect(() => {
    const current = training.data;
    const previous = previousTraining.current;
    previousTraining.current = current;
    if (!current?.runId || !previous?.runId || current.runId !== previous.runId || !previous.running || current.running) return;
    const action = { label: '查看日志', run: () => viewTraining(current.runId || undefined) };
    const options = { dedupeKey: `training:${expId}:${current.runId}:${current.state}`, action };
    if (current.state === 'succeeded') notify.success(`训练已完成 · ${current.runId}`, options);
    else if (current.state === 'cancelled') notify.info(`训练已取消 · ${current.runId}`, options);
    else if (current.state === 'failed') notify.error(
      current.error || `训练失败 · ${current.runId}`,
      { ...options, title: `训练失败 · ${current.runId}`, duration: null },
    );
    else if (current.state === 'interrupted') notify.warning(
      `训练托管中断 · ${current.runId}`,
      { ...options, duration: null },
    );
  }, [expId, notify, training.data, viewTraining]);

  const startTrain = useCallback(async () => {
    await run('train', async () => {
      if (!pendingTrainRequest.current) {
        const relevant = () => (registry?.getSnapshot() || []).filter(isExperimentDraft);
        const drafts = relevant();
        if (drafts.some(item => item.busy)) throw new Error('配置正在保存或执行，请稍后再启动训练。');
        const dirty = drafts.filter(item => item.dirty);
        if (dirty.length && !(await confirmAction({
          title: '保存修改并检查训练？',
          description: `将保存以下修改：\n${dirty.map(item => `• ${item.label}`).join('\n')}\n\n已保存内容不会因后续检查失败而撤销。`,
          confirmLabel: '保存并检查',
        }))) return;
        if (registry) await saveExperimentDrafts(registry, onReload);
        await onReload();
        if (!alive.current) return;
        if (relevant().some(item => item.dirty || item.busy)) throw new Error('保存期间配置又有修改，请保存后重新检查。');
        const preflight = await trainingApi.preflight(expId);
        if (!(await confirm(preflight)) || !alive.current) return;
        if (relevant().some(item => item.dirty || item.busy)) throw new Error('检查期间配置已修改，请重新检查。');
        pendingTrainRequest.current = {
          request_id: crypto.randomUUID(), preflight_revision: preflight.revision, stop_local_llm: true,
        };
      }
      let created;
      try {
        created = await rlApi.createRun(expId, pendingTrainRequest.current);
      } catch (error) {
        if (error instanceof ApiError) {
          // A gateway/server error can follow a successful launch; retain the idempotency key.
          if (error.status && error.status < 500) pendingTrainRequest.current = null;
          const detail = error.detail;
          if (detail && typeof detail === 'object' && 'run_id' in detail && typeof detail.run_id === 'string') {
            pendingTrainRequest.current = null;
            viewTraining(detail.run_id);
          }
        }
        if (pendingTrainRequest.current) throw new Error(`${errorMessage(error)}。启动结果尚未确认；将复查活动运行，再次提交会复用原请求编号，不会重复启动。`);
        throw error;
      }
      if (!alive.current) return;
      pendingTrainRequest.current = null;
      viewTraining(created.run_id);
      setStatus('train_started');
      await monitor.refresh();
      return created.reused ? `已恢复训练进程：${created.run_id}` : `训练进程已启动：${created.run_id}`;
    });
  }, [confirm, confirmAction, expId, run, monitor.refresh, registry, onReload, viewTraining]);
  const stopTrain = useCallback(async (runId: string) => {
    if (!(await confirmAction({
      title: '停止训练？',
      description: `将停止实验 ${expId} 的运行 ${runId} 及其子进程。`,
      confirmLabel: '停止训练',
      tone: 'danger',
    }))) return;
    await stopAction.run('stop', async () => {
      await rlApi.stopRun(expId, runId);
      if (!alive.current) return;
      setStatus('train_stopped');
      await training.refresh();
      await monitor.refresh();
      return '已发送停止指令';
    });
  }, [confirmAction, expId, stopAction.run, training.refresh, monitor.refresh]);
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
  const commands = useMemo(() => ({ startTrain, stopTrain, diagnose, viewTraining }),
    [startTrain, stopTrain, diagnose, viewTraining]);
  const commandStatus = useMemo(() => ({
    pending,
    status: events.latest || status,
  }), [pending, status, events.latest]);

  return <CommandsContext.Provider value={commands}>
    <CommandStatusContext.Provider value={commandStatus}>
      <TrainingContext.Provider value={training}>
        <AglContext.Provider value={agl}>
          <MonitorContext.Provider value={monitor}>
            <EventsContext.Provider value={events}><TrainingViewContext.Provider value={viewRequest}>
              {children}{dialog}
            </TrainingViewContext.Provider></EventsContext.Provider>
          </MonitorContext.Provider>
        </AglContext.Provider>
      </TrainingContext.Provider>
    </CommandStatusContext.Provider>
  </CommandsContext.Provider>;
}

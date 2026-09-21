import { useCallback, useEffect, useRef, useState } from 'react';
import type { RolloutExecution, RolloutRunSummary, RolloutTrajectoryResponse, WorkflowSpec } from '../../../shared/api/types';
import { ApiError, errorMessage } from '../../../shared/api/http';
import { masApi } from '../api';
import type { useMasDraft } from './useMasDraft';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';

export type RolloutStatus = 'saving' | 'running' | 'succeeded' | 'failed' | 'interrupted' | 'unknown';
export const rolloutStatusLabel: Record<RolloutStatus, string> = {
  saving: '正在保存 Workflow', running: '正在运行', succeeded: '运行成功',
  failed: '运行失败', interrupted: '运行已中断', unknown: '结果未知',
};
interface RolloutAttempt {
  workflow: WorkflowSpec | null;
  question: string;
  execution: RolloutExecution;
  status: RolloutStatus;
  startedAt: number;
  endedAt?: number;
  summary?: RolloutRunSummary;
  error?: string;
}
interface RolloutLog {
  at: number;
  message: string;
  tone: 'neutral' | 'success' | 'danger';
}

export function useMasRolloutRun(expId: string, executeWithSavedWorkflow: ReturnType<typeof useMasDraft>['executeWithSavedWorkflow'], execution: RolloutExecution = 'live') {
  const [question, setQuestion] = useState('');
  const [attempt, setAttempt] = useState<RolloutAttempt | null>(null);
  const [logs, setLogs] = useState<RolloutLog[]>([]);
  const [trajectory, setTrajectory] = useState<RolloutTrajectoryResponse | null>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState(false);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const generation = useRef(0);
  const mounted = useRef(false);
  const reading = useRef<AbortController | null>(null);
  useUnsavedChanges(`rollout-input-${execution}`, {
    label: execution === 'live' ? '未提交的真实调试输入' : '未提交的模拟演示输入', resource: `rollout-input-${execution}`,
    dirty: Boolean(question.trim()) && (!attempt?.summary || question !== attempt.question || execution !== attempt.execution),
    busy: attempt?.status === 'saving' || attempt?.status === 'running',
  });

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      generation.current++;
      reading.current?.abort();
    };
  }, [expId]);

  const start = useCallback(async () => {
    if (!question.trim()) return;
    let token = 0;
    let run: RolloutAttempt;
    const current = () => mounted.current && generation.current === token;
    const log = (message: string, tone: RolloutLog['tone'] = 'neutral') => {
      if (current()) setLogs(previous => [...previous, { at: Date.now(), message, tone }]);
    };
    await executeWithSavedWorkflow('rollout', {
      onSaving: workflow => {
        token = ++generation.current;
        reading.current?.abort();
        reading.current = null;
        setTrajectory(null);
        setTrajectoryError(null);
        setTrajectoryLoading(false);
        setLogs([]);
        run = { workflow, question, execution, status: 'saving', startedAt: Date.now() };
        setAttempt(run);
        log(`${execution === 'mock' ? '模拟演示' : '真实调试'} · 正在保存本次 Workflow`);
      },
      onSaveError: error => {
        if (!current()) return;
        const message = `保存 Workflow 失败，未提交运行：${errorMessage(error)}`;
        setAttempt({ ...run, status: 'failed', endedAt: Date.now(), error: message });
        log(message, 'danger');
      },
      execute: async workflow => {
        if (!current()) return;
        run = { ...run, workflow, status: 'running' };
        setAttempt(run);
        log('Workflow 已保存；提交同一份快照，新编辑不加入本次运行', 'success');
        log('已提交单次 Rollout 请求，等待服务端返回；不提供实时调用进度');
        try {
          const summary = await masApi.rollout(expId, { workflow, task: { question }, execution });
          if (!current()) return;
          setAttempt({ ...run, summary, status: summary.status,
            endedAt: summary.status === 'running' ? undefined : Date.now() });
          log(`${rolloutStatusLabel[summary.status]} · ${summary.run_id} · 模型调用 ${summary.model_call_count} 次 · 工具调用 ${summary.tool_call_count} 次`,
            summary.status === 'succeeded' ? 'success' : summary.status === 'running' ? 'neutral' : 'danger');
          if (summary.error) log(`${summary.error.stage} / ${summary.error.code}：执行失败，完整原因见结果详情。`, 'danger');
        } catch (error) {
          if (!current()) return;
          const rejected = error instanceof ApiError && (error.status === 400 || error.status === 422);
          const message = rejected ? `请求被拒绝（HTTP ${error.status}），未启动运行：${errorMessage(error)}`
            : `${error instanceof ApiError ? `HTTP ${error.status}：` : '请求中断：'}${errorMessage(error)}。未取得运行 ID，执行结果未知；不会自动重试，请先核实服务端记录。`;
          setAttempt({ ...run, status: rejected ? 'failed' : 'unknown', endedAt: Date.now(), error: message });
          log(message, 'danger');
        }
      },
    });
  }, [question, execution, expId, executeWithSavedWorkflow]);

  const readTrajectory = useCallback(async () => {
    const runId = attempt?.summary?.run_id;
    if (!runId || reading.current) return;
    const token = generation.current;
    const controller = new AbortController();
    reading.current = controller;
    setTrajectoryLoading(true);
    setTrajectoryError(null);
    try {
      const data = await masApi.rolloutTrajectory(expId, runId, controller.signal);
      if (!mounted.current || token !== generation.current) return;
      if (data.run.run_id !== runId) throw new Error('轨迹响应与本次运行身份不匹配，请重新读取。');
      setTrajectory(data);
      setAttempt(previous => previous && ({ ...previous, summary: data.run, status: data.run.status,
        endedAt: data.run.status === 'running' ? undefined : previous.endedAt ?? Date.now() }));
    } catch (error) {
      if (mounted.current && token === generation.current && !controller.signal.aborted) setTrajectoryError(errorMessage(error));
    } finally {
      if (reading.current === controller) {
        reading.current = null;
        if (mounted.current) setTrajectoryLoading(false);
      }
    }
  }, [attempt?.summary?.run_id, expId]);

  return { experimentId: expId, question, setQuestion, execution, attempt, logs, start,
    trajectory, trajectoryLoading, trajectoryError, readTrajectory,
    running: attempt?.status === 'saving' || attempt?.status === 'running' };
}

export type MasRolloutRun = ReturnType<typeof useMasRolloutRun>;

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, CollectBody, CollectResponse, LlmConfig, Palette, WorkflowSpec } from '../../../shared/api/types';
import { errorMessage } from '../../../shared/api/http';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import type { SampleDataInput, SampleDataResponse } from '../api';

interface ConsoleRun {
  id: number;
  mock: boolean;
  input: 'demo' | 'parquet';
  status: 'saving' | 'running' | 'succeeded' | 'failed';
  startedAt: number;
  endedAt?: number;
  error?: string;
  workflow: WorkflowSpec | null;
}

interface ConsoleResult {
  run: ConsoleRun;
  data: CollectResponse;
}

interface OperationLog {
  id: number;
  at: number;
  message: string;
  tone: 'neutral' | 'success' | 'danger';
}

function bindSavedModel(workflow: WorkflowSpec, llm: LlmConfig): WorkflowSpec {
  const binding = Object.fromEntries(['kind', 'model', 'base_url']
    .filter(key => llm[key] !== undefined).map(key => [key, llm[key]]));
  return JSON.stringify(workflow.llm) === JSON.stringify(binding) ? workflow : { ...workflow, llm: binding };
}

export function useMasDraft({ expId, bundle, onReload }: {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
}) {
  const [workflow, setWorkflow] = useState<WorkflowSpec | null>(() =>
    bundle?.id === expId && bundle.workflow ? bindSavedModel(bundle.workflow, bundle.llm) : null);
  const [palette, setPalette] = useState<Palette>({});
  const [workflowDirty, setWorkflowDirty] = useState(false);
  const [paletteError, setPaletteError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savingWorkflow, setSavingWorkflow] = useState(false);
  const [lastRun, setLastRun] = useState<ConsoleRun | null>(null);
  const [lastResult, setLastResult] = useState<ConsoleResult | null>(null);
  const [previewData, setPreviewData] = useState<{ input: SampleDataInput; data: SampleDataResponse } | null>(null);
  const [logs, setLogs] = useState<OperationLog[]>([]);
  const sequence = useRef(0);
  const synced = useRef(bundle?.id === expId && bundle.workflow ? expId : null);
  const syncedModelRevision = useRef(bundle?.llm.config_revision);
  const savedLlm = useRef(bundle?.llm || {});
  if (bundle?.id === expId) savedLlm.current = bundle.llm;
  const workflowRef = useRef(workflow);
  const action = useAction();
  const log = useCallback((message: string, tone: OperationLog['tone'] = 'neutral') => {
    const entry = { id: ++sequence.current, at: Date.now(), message, tone };
    setLogs(previous => [...previous.slice(-199), entry]);
  }, []);

  const loadPalette = useCallback(async (signal?: AbortSignal) => {
    setPaletteError(null);
    await api.palette(signal).then(setPalette).catch((error: unknown) => {
      if (!signal?.aborted) setPaletteError(errorMessage(error));
    });
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void loadPalette(controller.signal);
    return () => controller.abort();
  }, [loadPalette]);

  useEffect(() => {
    if (!bundle?.workflow || bundle.id !== expId) return;
    if (synced.current === expId) {
      if (syncedModelRevision.current !== bundle.llm.config_revision && workflowRef.current) {
        const next = bindSavedModel(workflowRef.current, bundle.llm);
        workflowRef.current = next;
        setWorkflow(next);
        syncedModelRevision.current = bundle.llm.config_revision;
      }
      return;
    }
    const nextWorkflow = bindSavedModel(bundle.workflow, bundle.llm);
    setWorkflow(nextWorkflow);
    workflowRef.current = nextWorkflow;
    setWorkflowDirty(false);
    synced.current = expId;
    syncedModelRevision.current = bundle.llm.config_revision;
  }, [expId, bundle]);

  const onWorkflowChange = useCallback((next: WorkflowSpec) => {
    const bound = bindSavedModel(next, savedLlm.current);
    workflowRef.current = bound;
    setWorkflow(bound);
    setWorkflowDirty(true);
  }, []);

  const saveWorkflow = useCallback(async () => {
    const draft = workflowRef.current;
    if (!draft) throw new Error('Workflow 尚未加载');
    const submitted = bindSavedModel(draft, savedLlm.current);
    workflowRef.current = submitted;
    if (submitted !== draft) setWorkflow(submitted);
    setSavingWorkflow(true);
    setSaveError(null);
    try {
      await api.putWorkflow(expId, submitted);
      if (workflowRef.current === submitted) setWorkflowDirty(false);
      onReload();
      return submitted;
    } catch (error) {
      setSaveError(errorMessage(error));
      throw error;
    } finally {
      setSavingWorkflow(false);
    }
  }, [expId, onReload]);

  const saveForNavigation = useCallback(async () => {
    let currentSaved = false;
    const success = await action.run('save', async () => {
      const submitted = await saveWorkflow();
      currentSaved = workflowRef.current === submitted;
      log('Workflow 已保存', 'success');
      return 'Workflow 已保存';
    });
    return success && currentSaved;
  }, [action.run, saveWorkflow, log]);
  const save = useCallback(async () => { await saveForNavigation(); }, [saveForNavigation]);

  const executeWithSavedWorkflow = useCallback((name: string, operation: {
    onSaving: (workflow: WorkflowSpec | null) => void;
    onSaveError: (error: unknown) => void;
    execute: (workflow: WorkflowSpec) => Promise<void>;
  }) => action.run(name, async () => {
    operation.onSaving(workflowRef.current);
    let submitted: WorkflowSpec;
    try {
      submitted = await saveWorkflow();
    } catch (error) {
      operation.onSaveError(error);
      throw error;
    }
    await operation.execute(submitted);
  }), [action.run, saveWorkflow]);

  const collect = useCallback(async (name: string, body: CollectBody) => {
    await action.run(name, async () => {
      let run: ConsoleRun = {
        id: ++sequence.current, mock: body.mock ?? true, input: body.parquet ? 'parquet' : 'demo',
        status: 'saving', startedAt: Date.now(),
        workflow: workflowRef.current,
      };
      setLastRun(run);
      log(`${run.mock ? '模拟' : '真实'}运行 · ${run.input === 'parquet' ? 'parquet 数据集' : '示例任务'} · 正在保存 Workflow`);
      let stage = '保存 Workflow';
      try {
        // A failed save must never fall through to collection of an older workflow.
        const submitted = await saveWorkflow();
        run = { ...run, workflow: submitted };
        log('Workflow 已保存；准备提交采集请求', 'success');
        if (workflowRef.current !== submitted) log('保存期间有新的编辑；本次使用刚保存的版本，新编辑仍待保存');
        stage = '采集请求';
        setLastRun({ ...run, status: 'running' });
        log('已提交采集请求，等待后端返回');
        const result = await api.collect(expId, body);
        const completed: ConsoleRun = { ...run, status: 'succeeded', endedAt: Date.now() };
        setLastRun(completed);
        setLastResult({ run: completed, data: result });
        log(`采集完成 · ${result.n} 条 · 平均奖励 ${result.mean_reward ?? '未返回'}`, 'success');
        onReload();
        return `运行完成 · ${result.n} 条`;
      } catch (error) {
        const message = `${stage}失败：${errorMessage(error)}`;
        setLastRun({ ...run, status: 'failed', endedAt: Date.now(), error: message });
        log(message, 'danger');
        throw new Error(message);
      }
    });
  }, [action.run, expId, saveWorkflow, onReload, log]);

  const preview = useCallback(async (body: SampleDataInput) => {
    await action.run('preview', async () => {
      setPreviewData(null);
      log(`读取预览 · ${body.parquet} · 来源 ${body.source || '全部'} · 前 ${body.data_n} 条`);
      try {
        const result = await api.sampleData(body);
        setPreviewData({ input: body, data: result });
        log(`预览已返回 ${result.n} 道题目；未保存 Workflow，未调用模型`, 'success');
        return `已预览 ${result.n} 道题目`;
      } catch (error) {
        log(`预览失败：${errorMessage(error)}`, 'danger');
        throw error;
      }
    });
  }, [action.run, log]);

  useUnsavedChanges('mas-workflow', {
    label: 'Workflow 设计', resource: 'workflow', dirty: workflowDirty,
    busy: savingWorkflow || (action.pending !== null && action.pending !== 'preview'), save: saveForNavigation,
  });
  return {
    workflow, palette, workflowDirty, rows: lastResult?.data.rows || [], onWorkflowChange,
    save, collect, preview, executeWithSavedWorkflow, pending: action.pending, notice: action.notice,
    lastRun, lastResult, previewData, logs,
    paletteError, loadPalette, saveError, savingWorkflow,
    running: lastRun?.status === 'saving' || lastRun?.status === 'running',
  };
}

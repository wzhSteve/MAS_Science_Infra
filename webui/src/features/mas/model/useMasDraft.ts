import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, CollectBody, CollectRow, Palette, RlConfig, WorkflowSpec } from '../../../shared/api/types';
import { useAction } from '../../../shared/hooks/useAction';

export function useMasDraft({ expId, bundle, onReload, normalizeRl }: {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  normalizeRl: (rl: RlConfig) => RlConfig;
}) {
  const [workflow, setWorkflow] = useState<WorkflowSpec | null>(() => bundle?.workflow || null);
  const [rl, setRl] = useState<RlConfig>(() => structuredClone(bundle?.rl || {}));
  const [palette, setPalette] = useState<Palette>({});
  const [workflowDirty, setWorkflowDirty] = useState(false);
  const [rlDirty, setRlDirty] = useState(false);
  const [rows, setRows] = useState<CollectRow[]>([]);
  const synced = useRef(bundle?.workflow ? expId : null);
  const workflowRef = useRef(workflow);
  const rlRef = useRef(rl);
  const action = useAction();
  const rlAction = useAction();

  useEffect(() => {
    const controller = new AbortController();
    api.palette(controller.signal).then(setPalette).catch((error: unknown) => {
      if (!controller.signal.aborted) action.setNotice({ message: String(error), tone: 'danger' });
    });
    return () => controller.abort();
  }, [action.setNotice]);

  useEffect(() => {
    if (!bundle?.workflow || synced.current === expId) return;
    const nextRl = structuredClone(bundle.rl || {});
    setWorkflow(bundle.workflow);
    workflowRef.current = bundle.workflow;
    setRl(nextRl);
    rlRef.current = nextRl;
    setWorkflowDirty(false);
    setRlDirty(false);
    setRows([]);
    synced.current = expId;
  }, [expId, bundle]);

  const onWorkflowChange = useCallback((next: WorkflowSpec) => {
    workflowRef.current = next;
    setWorkflow(next);
    setWorkflowDirty(true);
  }, []);

  const onRlPatch = useCallback((patch: Partial<RlConfig>) => {
    const next = { ...rlRef.current, ...patch };
    rlRef.current = next;
    setRl(next);
    setRlDirty(true);
  }, []);

  const saveWorkflow = useCallback(async () => {
    const submitted = workflowRef.current;
    if (!submitted) throw new Error('Workflow 尚未加载');
    await api.putWorkflow(expId, submitted);
    if (workflowRef.current === submitted) setWorkflowDirty(false);
    onReload();
  }, [expId, onReload]);

  const save = useCallback(async () => {
    await action.run('save', async () => {
      await saveWorkflow();
      return 'saved workflow.yaml';
    });
  }, [action.run, saveWorkflow]);

  const saveRl = useCallback(async () => {
    await rlAction.run('save-rl', async () => {
      const submitted = rlRef.current;
      const payload = normalizeRl(submitted);
      await api.putSection(expId, 'rl', payload);
      if (rlRef.current === submitted) {
        rlRef.current = payload;
        setRl(payload);
        setRlDirty(false);
      }
      onReload();
      return 'saved rl.yaml（训练简参）';
    });
  }, [expId, normalizeRl, onReload, rlAction.run]);

  const collect = useCallback(async (name: string, body: CollectBody) => {
    await action.run(name, async () => {
      // Let save failures reject this action: collecting an unsaved draft is unsafe.
      await saveWorkflow();
      const result = await api.collect(expId, body);
      setRows(result.rows || []);
      onReload();
      if (name === 'parquet') {
        const passed = (result.rows || []).filter((row) => row.tool && Number(row.reward) > 0).length;
        return `live-api-data via UI: ${passed}/${result.n} pass · mean_reward=${result.mean_reward} · ${result.path}`;
      }
      return `collect done n=${result.n} mean_reward=${result.mean_reward}`;
    });
  }, [action.run, expId, saveWorkflow, onReload]);

  const preview = useCallback(async (body: { parquet: string; data_n: number; source: string }) => {
    await action.run('preview', async () => {
      const result = await api.sampleData(body);
      return `sampled ${result.n}: ${(result.tasks || []).map((task) => task.id).join(', ')}`;
    });
  }, [action.run]);

  return {
    workflow, rl, palette, workflowDirty, rlDirty, rows, onWorkflowChange, onRlPatch,
    save, saveRl, collect, preview, pending: action.pending, notice: action.notice,
    savingRl: Boolean(rlAction.pending), rlNotice: rlAction.notice,
  };
}

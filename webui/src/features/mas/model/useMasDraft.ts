import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, LlmConfig, Palette, WorkflowSpec } from '../../../shared/api/types';
import { errorMessage } from '../../../shared/api/http';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';

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
  const synced = useRef(bundle?.id === expId && bundle.workflow ? expId : null);
  const syncedModelRevision = useRef(bundle?.llm.config_revision);
  const savedLlm = useRef(bundle?.llm || {});
  if (bundle?.id === expId) savedLlm.current = bundle.llm;
  const workflowRef = useRef(workflow);
  const action = useAction({ errorFeedback: 'silent' });

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
      return 'Workflow 已保存';
    });
    return success && currentSaved;
  }, [action.run, saveWorkflow]);
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

  useUnsavedChanges('mas-workflow', {
    label: 'Workflow 设计', resource: 'workflow', dirty: workflowDirty,
    busy: savingWorkflow || action.pending !== null, save: saveForNavigation,
  });
  return {
    workflow, palette, workflowDirty, onWorkflowChange,
    save, executeWithSavedWorkflow, pending: action.pending,
    paletteError, loadPalette, saveError, savingWorkflow,
  };
}

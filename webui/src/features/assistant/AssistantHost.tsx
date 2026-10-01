import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { AssistantDock } from './AssistantDock';
import { AssistantCanvasProvider, type AssistantCanvasApi } from './context';

export function AssistantHost({ expId, children }: { expId?: string | null; children: ReactNode }) {
  const [bound, setBound] = useState<Omit<AssistantCanvasApi, 'bind'>>({
    expId: expId ?? null,
    workflow: null,
    selectedNodeId: null,
    applyWorkflow: () => undefined,
    reloadPalette: () => undefined,
  });
  const bind = useCallback((next: Partial<Omit<AssistantCanvasApi, 'bind'>>) => {
    setBound((current) => ({ ...current, ...next }));
  }, []);
  const value = useMemo<AssistantCanvasApi>(() => ({
    ...bound,
    expId: expId ?? bound.expId,
    bind,
  }), [bind, bound, expId]);
  return <AssistantCanvasProvider value={value}>{children}<AssistantDock /></AssistantCanvasProvider>;
}

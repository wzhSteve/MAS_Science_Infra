import { createContext, useContext, type ReactNode } from 'react';
import type { WorkflowSpec } from '../../shared/api/types';

export type AssistantCanvasApi = {
  expId: string | null;
  workflow: WorkflowSpec | null;
  selectedNodeId: string | null;
  applyWorkflow: (workflow: WorkflowSpec) => void;
  reloadPalette: () => void;
  bind: (next: Partial<Omit<AssistantCanvasApi, 'bind'>>) => void;
};

const defaults: AssistantCanvasApi = {
  expId: null,
  workflow: null,
  selectedNodeId: null,
  applyWorkflow: () => undefined,
  reloadPalette: () => undefined,
  bind: () => undefined,
};

const AssistantCanvasContext = createContext<AssistantCanvasApi>(defaults);

export function AssistantCanvasProvider({ value, children }: { value: AssistantCanvasApi; children: ReactNode }) {
  return <AssistantCanvasContext.Provider value={value}>{children}</AssistantCanvasContext.Provider>;
}

export function useAssistantCanvas() {
  return useContext(AssistantCanvasContext);
}

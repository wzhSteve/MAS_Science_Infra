import type { WorkflowSpec } from '../../shared/api/types';

export interface UserProject {
  id: string;
  title: string;
  mode: 'io_module' | 'native_mas' | string;
  status?: string;
  agent_ids?: string[];
  tool_ids?: string[];
  experiment_id?: string;
  validation?: { ok?: boolean; issues?: string[]; n_sites?: number };
}

export interface AssistantApplyResult {
  workflow: WorkflowSpec;
  mode: string;
  project: UserProject;
}

export type WorkflowSummary = {
  agents: Array<{ id: string; kind?: string }>;
  routers: string[];
  n_sites: number;
  entry_agent?: string;
};

export type ChatRequest = {
  message: string;
  experiment_id?: string;
  project_id?: string;
  history?: Array<{ role: string; content: string }>;
  workflow_summary?: WorkflowSummary;
  selected_node_id?: string;
  current_workflow?: WorkflowSpec;
};

export type ChatEventName = 'status' | 'token' | 'tool_start' | 'tool_end' | 'done' | 'error';

export type ChatEvent = {
  event: ChatEventName | string;
  data: Record<string, unknown>;
};

async function readError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const body = JSON.parse(text) as { detail?: string };
    if (body.detail) return body.detail;
  } catch {
    // keep text
  }
  return text || response.statusText;
}

export function workflowSummary(workflow: WorkflowSpec | null | undefined): WorkflowSummary | undefined {
  if (!workflow) return undefined;
  return {
    agents: (workflow.agents || []).map((agent) => ({ id: agent.id, kind: agent.kind })),
    routers: (workflow.routers || []).map((router) => router.id),
    n_sites: workflow.sampling?.sites?.length || 0,
    entry_agent: workflow.entry_agent || 'planner',
  };
}

export const assistantApi = {
  listProjects: (experimentId?: string) =>
    fetch(`/api/assistant/projects${experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : ''}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(await readError(r));
        return r.json() as Promise<{ projects: UserProject[] }>;
      }),
  upload: async (form: FormData) => {
    const response = await fetch('/api/assistant/projects', { method: 'POST', body: form });
    if (!response.ok) throw new Error(await readError(response));
    return response.json() as Promise<{ project: UserProject; workflow?: WorkflowSpec }>;
  },
  apply: async (projectId: string, current: WorkflowSpec, replace = false) => {
    const response = await fetch(`/api/assistant/projects/${encodeURIComponent(projectId)}/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_workflow: current, replace }),
    });
    if (!response.ok) throw new Error(await readError(response));
    return response.json() as Promise<AssistantApplyResult>;
  },
  chat: async (body: ChatRequest, onEvent: (event: ChatEvent) => void, signal?: AbortSignal) => {
    const response = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok) throw new Error(await readError(response));
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop() || '';
      for (const chunk of chunks) {
        let eventName = 'token';
        let payload: Record<string, unknown> = {};
        for (const line of chunk.split('\n')) {
          if (line.startsWith('event: ')) eventName = line.slice(7).trim();
          if (line.startsWith('data: ')) {
            try {
              payload = JSON.parse(line.slice(6)) as Record<string, unknown>;
            } catch {
              payload = { text: line.slice(6) };
            }
          }
        }
        onEvent({ event: eventName, data: payload });
      }
    }
  },
};

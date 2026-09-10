import type { Node, Edge } from '@xyflow/react';

export const KNOWN_TOOLS = ['web_search', 'wikipedia_search', 'execute_python'];

export type AgentSpec = {
  id: string;
  role?: string;
  skills?: string[];
  tools?: string[];
  memory_scope?: string;
  system_prompt?: string;
  model?: string;
  trainable?: boolean;
};

export type WorkflowSpec = {
  schema_version?: string;
  topology: string;
  entry_agent?: string;
  hub: {
    role?: string;
    skills?: string[];
    verify?: string | null;
    max_feedback_hops?: number;
    system_prompt?: string;
  };
  tools: string[];
  llm?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  archive?: Record<string, unknown>;
  agents?: AgentSpec[];
  edges?: Array<{ from: string; to: string; kind?: string }>;
};

export function workflowToFlow(wf: WorkflowSpec): { nodes: Node[]; edges: Edge[] } {
  const entry = wf.entry_agent || 'hub';
  const agents =
    wf.agents && wf.agents.length > 0
      ? wf.agents
      : [
          {
            id: 'hub',
            role: wf.hub?.role || 'orchestrator',
            skills: wf.hub?.skills || ['react_loop'],
            tools: wf.tools || [],
            system_prompt: wf.hub?.system_prompt || '',
            trainable: true,
          },
        ];

  const nodes: Node[] = agents.map((a, i) => ({
    id: a.id,
    type: 'agent',
    position: { x: 80 + (i % 3) * 240, y: 80 + Math.floor(i / 3) * 150 },
    data: {
      label: a.id,
      role: a.role || 'agent',
      skills: a.skills || [],
      tools: a.tools || (a.id === 'hub' ? wf.tools : []),
      system_prompt: a.system_prompt || (a.id === 'hub' ? wf.hub?.system_prompt : '') || '',
      trainable: a.trainable !== false,
      entry: a.id === entry,
      verify: a.id === 'hub' ? wf.hub?.verify : undefined,
      max_feedback_hops: a.id === 'hub' ? wf.hub?.max_feedback_hops : undefined,
    },
  }));

  if (wf.hub?.verify && !nodes.find((n) => n.id === 'verifier')) {
    nodes.push({
      id: 'verifier',
      type: 'agent',
      position: { x: 300, y: 280 },
      data: {
        label: 'verifier',
        role: 'verifier',
        skills: ['verifier'],
        tools: [],
        trainable: false,
        entry: false,
      },
    });
  }

  const toolSet = new Set<string>();
  for (const a of agents) {
    for (const t of a.tools || []) toolSet.add(t);
  }
  for (const t of wf.tools || []) toolSet.add(t);
  for (const e of wf.edges || []) {
    if (e.kind === 'tool_call') toolSet.add(e.to);
  }
  let ti = 0;
  for (const t of toolSet) {
    if (nodes.some((n) => n.id === t)) continue;
    nodes.push({
      id: t,
      type: 'tool',
      position: { x: 80 + (ti % 3) * 220, y: 360 },
      data: { label: t, role: 'tool' },
    });
    ti += 1;
  }

  const edges: Edge[] = (wf.edges || []).map((e, i) => ({
    id: `e-${e.from}-${e.to}-${i}`,
    source: e.from,
    target: e.to,
    label: e.kind || 'message',
    data: { kind: e.kind || 'message' },
  }));

  if (wf.hub?.verify && !edges.some((e) => e.source === 'verifier')) {
    edges.push({
      id: 'e-verifier-hub',
      source: 'verifier',
      target: 'hub',
      label: 'feedback',
      data: { kind: 'feedback' },
    });
  }

  return { nodes, edges };
}

export function flowToWorkflow(
  nodes: Node[],
  edges: Edge[],
  base: WorkflowSpec,
): WorkflowSpec {
  const agentNodes = nodes.filter((n) => n.type !== 'tool');
  const toolNodes = nodes.filter((n) => n.type === 'tool');
  const hubNode = agentNodes.find((n) => n.id === 'hub') || agentNodes[0];
  const entry =
    agentNodes.find((n) => n.data?.entry)?.id ||
    base.entry_agent ||
    hubNode?.id ||
    'hub';

  const agents: AgentSpec[] = agentNodes.map((n) => ({
    id: n.id,
    role: String(n.data?.role || 'agent'),
    skills: (n.data?.skills as string[]) || [],
    tools: (n.data?.tools as string[]) || [],
    memory_scope: 'agent',
    system_prompt: String(n.data?.system_prompt || ''),
    model: String(n.data?.model || 'inherit'),
    trainable: n.data?.trainable !== false,
  }));

  for (const e of edges) {
    if (String(e.data?.kind || e.label) !== 'tool_call') continue;
    const agent = agents.find((a) => a.id === e.source);
    if (agent && !agent.tools.includes(e.target)) agent.tools.push(e.target);
  }

  const toolsFromHub = (hubNode?.data?.tools as string[]) || [];
  const tools = Array.from(
    new Set([
      ...toolsFromHub,
      ...agents.flatMap((a) => a.tools),
      ...toolNodes.map((n) => n.id),
      ...(base.tools || []),
    ]),
  );

  const skills = (hubNode?.data?.skills as string[]) || base.hub?.skills || ['react_loop'];
  const verify = (hubNode?.data?.verify as string | null | undefined) ?? base.hub?.verify ?? null;
  const maxHops =
    (hubNode?.data?.max_feedback_hops as number | undefined) ?? base.hub?.max_feedback_hops ?? 1;
  const hubPrompt =
    String(hubNode?.data?.system_prompt || '') || base.hub?.system_prompt || '';

  const edgeSpecs = edges.map((e) => ({
    from: e.source,
    to: e.target,
    kind: String(e.data?.kind || e.label || 'message'),
  }));

  const onlyHubish = agents.every((a) => a.id === 'hub' || a.id === 'verifier');
  const topology =
    agents.length <= 2 && onlyHubish && !agents.some((a) => a.id === 'planner')
      ? verify || agents.length > 1
        ? base.topology === 'graph'
          ? 'graph'
          : 'hub_react'
        : 'hub_react'
      : 'graph';

  return {
    ...base,
    schema_version: topology === 'graph' ? '0.2.0' : base.schema_version || '0.1.0',
    topology,
    entry_agent: entry,
    hub: {
      role: String(hubNode?.data?.role || 'orchestrator'),
      skills,
      verify: verify || null,
      max_feedback_hops: maxHops,
      system_prompt: hubPrompt,
    },
    tools,
    agents,
    edges: edgeSpecs,
  };
}

export function executableInfo(wf: WorkflowSpec): { ok: boolean; reason: string } {
  if (wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology) {
    return { ok: true, reason: 'hub_react' };
  }
  if (wf.topology === 'graph') {
    const agentIds = new Set((wf.agents || []).map((a) => a.id));
    const inboundRoute: Record<string, number> = {};
    for (const e of wf.edges || []) {
      const kind = e.kind || 'message';
      const toIsTool = KNOWN_TOOLS.includes(e.to);
      const fromIsTool = KNOWN_TOOLS.includes(e.from);
      if (kind === 'tool_call') {
        if (!agentIds.has(e.from)) {
          return { ok: false, reason: `tool_call source ${e.from} must be an agent` };
        }
        if (agentIds.has(e.to)) {
          return { ok: false, reason: `tool_call target ${e.to} is an agent` };
        }
        continue;
      }
      if (toIsTool || fromIsTool) {
        return { ok: false, reason: `${kind} cannot involve tool node` };
      }
      if (kind === 'route') {
        inboundRoute[e.to] = (inboundRoute[e.to] || 0) + 1;
        if (inboundRoute[e.to] > 1) {
          return { ok: false, reason: `agent ${e.to} has more than one inbound route` };
        }
      }
    }
    return { ok: true, reason: 'graph_compiled' };
  }
  return { ok: false, reason: `unknown topology ${wf.topology}` };
}

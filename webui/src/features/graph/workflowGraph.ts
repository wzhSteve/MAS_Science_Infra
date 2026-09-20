import type { Node, Edge } from '@xyflow/react';

export const KNOWN_TOOLS = ['web_search', 'wikipedia_search', 'execute_python'];

export type AgentSpec = {
  id: string;
  kind?: string; // schema 0.3: hub | planner | tool | verifier | blank
  role?: string;
  skills?: string[];
  tools?: string[];
  memory_scope?: string;
  system_prompt?: string;
  model?: string;
  trainable?: boolean;
  profile?: Record<string, unknown>;
};

export type RouterSpec = {
  id: string;
  candidates?: string[];
  strategy?: string;
  scorer?: string | null;
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
  routers?: RouterSpec[];
  edges?: Array<{ from: string; to: string; kind?: string }>;
  sampling?: {
    mode?: string;
    group_n?: number;
    beam_size?: number;
    initial_rollouts?: number;
    barriers?: string[];
    sites?: Array<Record<string, unknown>>;
  };
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

  const nodes: Node[] = agents.map((a, i) => {
    // W3: blank agents carry no role semantics — prefer profile.skills as label
    const skills = (a.profile?.skills as string[] | undefined) || a.skills || [];
    const label = a.kind === 'blank' && skills.length > 0 ? `${a.id} (${skills[0]})` : a.id;
    return {
    id: a.id,
    type: 'agent',
    position: { x: 80 + (i % 3) * 240, y: 80 + Math.floor(i / 3) * 150 },
    data: {
      label,
      kind: a.kind || (a.id === 'hub' ? 'hub' : 'blank'), // schema 0.3 kind badge
      role: a.role || 'agent',
      skills: a.skills || [],
      tools: a.tools || (a.id === 'hub' ? wf.tools : []),
      system_prompt: a.system_prompt || (a.id === 'hub' ? wf.hub?.system_prompt : '') || '',
      memory_scope: a.memory_scope || 'agent', // W3: fidelity round-trip
      profile: a.profile || {}, // W3: blank agent custom profile
      trainable: a.trainable !== false,
      entry: a.id === entry,
      verify: a.id === 'hub' ? wf.hub?.verify : undefined,
      max_feedback_hops: a.id === 'hub' ? wf.hub?.max_feedback_hops : undefined,
    },
  }});

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

  // agent-framework A5: routers render as diamond routing nodes showing
  // strategy + candidate count (edges to candidates handled below).
  const routerIds = new Set<string>((wf.routers || []).map((r) => r.id));
  for (const r of wf.routers || []) {
    if (nodes.some((n) => n.id === r.id)) continue;
    nodes.push({
      id: r.id,
      type: 'router',
      position: { x: 420 + (nodes.length % 2) * 200, y: 240 + Math.floor(nodes.length / 2) * 140 },
      data: {
        label: r.id,
        strategy: r.strategy || 'llm_choice',
        candidates: r.candidates || [],
        n_candidates: (r.candidates || []).length,
      },
    });
  }

  const edges: Edge[] = (wf.edges || []).map((e, i) => ({
    id: `e-${e.from}-${e.to}-${i}`,
    source: e.from,
    target: e.to,
    label: e.kind || 'message',
    data: { kind: e.kind || 'message' },
  }));

  // Router candidate edges (route from router node to each candidate).
  for (const r of wf.routers || []) {
    for (const c of r.candidates || []) {
      if (edges.some((e) => e.source === r.id && e.target === c)) continue;
      edges.push({
        id: `e-router-${r.id}-${c}`,
        source: r.id,
        target: c,
        label: 'candidate',
        data: { kind: 'candidate' },
      });
    }
  }

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
  const agentNodes = nodes.filter((n) => n.type !== 'tool' && n.type !== 'router');
  const toolNodes = nodes.filter((n) => n.type === 'tool');
  const routerNodes = nodes.filter((n) => n.type === 'router');
  const hubNode = agentNodes.find((n) => n.id === 'hub') || agentNodes[0];
  const entry =
    agentNodes.find((n) => n.data?.entry)?.id ||
    base.entry_agent ||
    hubNode?.id ||
    'hub';

  const agents: AgentSpec[] = agentNodes.map((n) => ({
    id: n.id,
    kind: String(n.data?.kind || (n.id === 'hub' ? 'hub' : 'blank')), // W3: schema 0.3 fidelity
    role: String(n.data?.role || 'agent'),
    skills: (n.data?.skills as string[]) || [],
    tools: (n.data?.tools as string[]) || [],
    memory_scope: String(n.data?.memory_scope || 'agent'), // W3: agent/shared
    system_prompt: String(n.data?.system_prompt || ''),
    model: String(n.data?.model || 'inherit'),
    trainable: n.data?.trainable !== false,
    profile: (n.data?.profile as Record<string, unknown>) || {}, // W3: blank profile fidelity
  }));

  for (const e of edges) {
    if (String(e.data?.kind || e.label) !== 'tool_call') continue;
    const agent = agents.find((a) => a.id === e.source);
    if (agent && !agent.tools?.includes(e.target)) (agent.tools ||= []).push(e.target);
  }

  const toolsFromHub = (hubNode?.data?.tools as string[]) || [];
  const tools = Array.from(
    new Set([
      ...toolsFromHub,
      ...agents.flatMap((a) => a.tools || []),
      ...toolNodes.map((n) => n.id),
      ...(base.tools || []),
    ]),
  );

  const skills =
    (hubNode?.data?.skills as string[] | undefined) || base.hub?.skills || ['react_loop'];
  const verify = (hubNode?.data?.verify as string | null | undefined) ?? base.hub?.verify ?? null;
  const maxHops =
    (hubNode?.data?.max_feedback_hops as number | undefined) ?? base.hub?.max_feedback_hops ?? 1;
  const hubPrompt =
    String(hubNode?.data?.system_prompt || '') || base.hub?.system_prompt || '';

  const edgeSpecs = edges
    .filter((e) => e.source !== e.target && e.data?.kind !== 'candidate')
    .map((e) => ({
      from: e.source,
      to: e.target,
      kind: String(e.data?.kind || e.label || 'message'),
    }));

  // W5: rebuild routers[] from canvas router nodes + candidate edges. Canvas
  // edges with kind 'candidate' (router -> agent) define membership; strategy
  // and scorer live on the router node data. Router ids not present on the
  // canvas are dropped (deleted) — deletions round-trip.
  const routers: RouterSpec[] = routerNodes.map((n) => {
    const cands = edges
      .filter((e) => e.source === n.id && e.data?.kind === 'candidate')
      .map((e) => e.target);
    return {
      id: n.id,
      candidates: cands,
      strategy: String(n.data?.strategy || 'llm_choice'),
      scorer: (n.data?.scorer as string | null | undefined) || null,
    };
  });
  // preserve strategy/candidates of spec-level routers whose nodes are absent
  // from the canvas (defensive: workflowToFlow always creates them, so this
  // only matters for programmatically-built specs).
  for (const r of base.routers || []) {
    if (routers.some((x) => x.id === r.id)) continue;
    routers.push({
      id: r.id,
      candidates: r.candidates || [],
      strategy: r.strategy || 'llm_choice',
      scorer: r.scorer ?? null,
    });
  }

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
    schema_version: '0.3', // W3: schema 0.3 round-trip (routers/kind/profile)
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
    // W5: routers rebuilt from canvas router nodes + candidate edges.
    routers,
    edges: edgeSpecs,
  };
}

export function executableInfo(wf: WorkflowSpec): { ok: boolean; reason: string } {
  if (wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology) {
    return { ok: true, reason: 'hub_react' };
  }
  if (wf.topology === 'graph') {
    const agentIds = new Set((wf.agents || []).map((a) => a.id));
    const routerIds = new Set((wf.routers || []).map((r) => r.id));
    const inboundRoute: Record<string, number> = {};
    for (const e of wf.edges || []) {
      const kind = e.kind || 'message';
      // agent-framework A5: router edges are routing sugar, not graph edges.
      if (routerIds.has(e.from) || routerIds.has(e.to)) continue;
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
    // W5: router candidates must exist among agents (compiler mirrors this).
    // blank:<id> candidates are accepted (bare agent id underneath).
    for (const r of wf.routers || []) {
      for (const c of r.candidates || []) {
        const bare = c.startsWith('blank:') ? c.slice('blank:'.length) : c;
        if (!agentIds.has(bare)) {
          return { ok: false, reason: `router ${r.id} candidate ${c} is not an agent node` };
        }
      }
      if (r.strategy === 'score' && r.scorer && !agentIds.has(r.scorer)) {
        return { ok: false, reason: `router ${r.id} scorer ${r.scorer} is not an agent node` };
      }
    }
    return { ok: true, reason: 'graph_compiled' };
  }
  return { ok: false, reason: `unknown topology ${wf.topology}` };
}

import type { AgentKind, AgentSpec, RouterSpec, WorkflowSpec } from '../../../shared/api/types';
import type { EdgeKind, GraphNode, GraphEdge } from '../types';
import { EDGE_KINDS, edgeDefinition } from './edgeDefinitions';
import { resolveHandles, serializeEdges } from './edgeGeometry';
export type { AgentSpec, WorkflowSpec } from '../../../shared/api/types';

export { KNOWN_TOOLS } from './edgeDefinitions';

export const EDGE_LABELS: Record<string, string> = Object.fromEntries(EDGE_KINDS.map((kind) => [kind, edgeDefinition(kind).title]));

export function graphEdge(source: string, target: string, kind: EdgeKind, id = `edge-${crypto.randomUUID()}`): GraphEdge {
  return { id, source, target, type: 'workflow', data: { kind } };
}

export function workflowGraphRevision(workflow: WorkflowSpec): string {
  return JSON.stringify({
    topology: workflow.topology,
    entry_agent: workflow.entry_agent,
    hub: workflow.hub,
    tools: workflow.tools,
    agents: workflow.agents,
    routers: workflow.routers,
    edges: workflow.edges,
  });
}

function inferredKind(agent: AgentSpec): AgentKind {
  const role = (agent.role || '').toLowerCase();
  if (agent.id === 'hub' || role === 'hub' || role === 'orchestrator') return 'hub';
  if (role === 'planner' || role === 'executor') return 'planner';
  if (role === 'verifier') return 'verifier';
  if (role === 'tool') return 'tool';
  return 'blank';
}

export function workflowToFlow(
  wf: WorkflowSpec,
  modelNames: ReadonlyMap<string, string> = new Map(),
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const entry = wf.entry_agent || 'hub';
  const agents =
    wf.agents && (wf.agents.length > 0 || wf.topology === 'graph')
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

  const nodes: GraphNode[] = agents.map((a, i) => ({
    id: a.id,
    type: 'agent',
    position: { x: 80 + (i % 3) * 240, y: 80 + Math.floor(i / 3) * 150 },
    data: {
      label: a.id,
      kind: a.kind || inferredKind(a),
      role: a.role || 'agent',
      skills: a.skills || [],
      tools: a.tools || (a.id === 'hub' ? wf.tools : []),
      memory_scope: a.memory_scope || 'agent',
      system_prompt: a.system_prompt ?? (a.id === 'hub' ? wf.hub?.system_prompt : '') ?? '',
      model: a.model || 'inherit',
      model_name: a.model && a.model !== 'inherit' ? modelNames.get(a.model) || a.model : undefined,
      trainable: a.trainable !== false,
      profile: a.profile || {},
      meta: a.meta || {},
      backend: typeof a.profile?.backend === 'string' ? a.profile.backend : undefined,
      llm_required: a.profile?.llm_required === true,
      entry: a.id === entry,
      verify: a.id === 'hub' ? wf.hub?.verify : undefined,
      max_feedback_hops: a.id === 'hub' ? wf.hub?.max_feedback_hops : undefined,
    },
  }));

  if (wf.hub?.verify && nodes.some((n) => n.id === 'hub') && !nodes.find((n) => n.id === 'verifier')) {
    nodes.push({
      id: 'verifier',
      type: 'agent',
      position: { x: 300, y: 280 },
      data: {
        label: 'verifier',
        kind: 'verifier',
        role: 'verifier',
        skills: ['verifier'],
        tools: [],
        memory_scope: 'agent',
        system_prompt: '',
        model: 'inherit',
        trainable: false,
        profile: {},
        meta: {},
        entry: false,
      },
    });
  }

  for (const [index, router] of (wf.routers || []).entries()) {
    nodes.push({
      id: router.id,
      type: 'router',
      position: { x: 420 + (index % 2) * 240, y: 100 + Math.floor(index / 2) * 160 },
      data: {
        label: router.id,
        candidates: [...(router.candidates || [])],
        strategy: router.strategy || 'llm_choice',
        scorer: router.scorer ?? null,
        meta: router.meta || {},
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

  const edges: GraphEdge[] = (wf.edges || []).map((e, i) => ({
    ...graphEdge(e.from, e.to, e.kind || 'message', `e-${e.from}-${e.to}-${i}`),
    data: { kind: e.kind || 'message', meta: e.meta },
  }));

  for (const node of nodes.filter((n) => n.type === 'agent')) {
    for (const tool of node.data.tools || []) {
      if (!edges.some((e) => e.source === node.id && e.target === tool && e.data?.kind === 'tool_call')) {
        edges.push(graphEdge(node.id, tool, 'tool_call'));
      }
    }
    node.data.tools = edges.filter((e) => e.source === node.id && e.data?.kind === 'tool_call').map((e) => e.target);
  }
  if (wf.hub?.verify && nodes.some((n) => n.id === 'hub') && !edges.some((e) => e.source === 'verifier' && e.target === 'hub' && e.data?.kind === 'feedback')) {
    edges.push(graphEdge('verifier', 'hub', 'feedback'));
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  return { nodes, edges: edges.map((edge) => resolveHandles(edge, nodeById)) };
}

export function flowToWorkflow(
  nodes: GraphNode[],
  edges: GraphEdge[],
  base: WorkflowSpec,
): WorkflowSpec {
  const agentNodes = nodes.filter((n) => n.type === 'agent');
  const toolNodes = nodes.filter((n) => n.type === 'tool');
  const routerNodes = nodes.filter((n) => n.type === 'router');
  const hubNode = agentNodes.find((n) => n.id === 'hub') || agentNodes[0];
  const entry =
    agentNodes.find((n) => n.data?.entry)?.id ||
    hubNode?.id ||
    '';

  const previousAgents = new Map((base.agents || []).map((a) => [a.id, a]));
  const agents: Array<AgentSpec & { tools: string[] }> = agentNodes.map((n) => {
    const previous = previousAgents.get(n.id);
    return {
      ...previous,
      id: n.id,
      kind: n.data.kind || previous?.kind || inferredKind({ id: n.id, role: n.data.role }),
      role: n.data.role || 'agent',
      skills: n.data.skills || [],
      tools: [...(n.data.tools || [])],
      memory_scope: n.data.memory_scope || 'agent',
      system_prompt: n.data.system_prompt || '',
      model: n.data.model || 'inherit',
      trainable: n.data.trainable !== false,
      profile: n.data.profile || {},
      meta: n.data.meta || {},
    };
  });

  const agentById = new Map(agents.map((agent) => [agent.id, agent]));
  for (const e of edges) {
    if (String(e.data?.kind || e.label) !== 'tool_call') continue;
    const agent = agentById.get(e.source);
    if (agent && !agent.tools.includes(e.target)) agent.tools.push(e.target);
  }

  const tools = Array.from(
    new Set([
      ...(base.tools || []),
      ...agents.flatMap((a) => a.tools),
      ...toolNodes.map((n) => n.id),
    ]),
  );

  const previousRouters = new Map((base.routers || []).map((router) => [router.id, router]));
  const routers: RouterSpec[] = routerNodes.map((node) => ({
    ...previousRouters.get(node.id),
    id: node.id,
    candidates: [...(node.data.candidates || [])],
    strategy: node.data.strategy || 'llm_choice',
    scorer: node.data.scorer || null,
    meta: node.data.meta || {},
  }));

  const skills = hubNode?.data.skills || [];
  const verify = hubNode?.data.verify ?? null;
  const maxHops =
    hubNode?.data.max_feedback_hops ?? base.hub?.max_feedback_hops ?? 1;
  const hubPrompt =
    hubNode?.data.system_prompt || '';

  const onlyHubish = agents.every((a) => a.kind === 'hub' || a.kind === 'verifier');
  const topology =
    !routers.length && hubNode?.id === 'hub' && agents.length <= 2 && onlyHubish
      ? verify || agents.length > 1
        ? base.topology === 'graph'
          ? 'graph'
          : 'hub_react'
        : 'hub_react'
      : 'graph';

  return {
    ...base,
    schema_version: '0.3',
    topology,
    entry_agent: entry,
    hub: {
      ...base.hub,
      role: String(hubNode?.data?.role || 'orchestrator'),
      skills,
      verify: verify || null,
      max_feedback_hops: maxHops,
      system_prompt: hubPrompt,
    },
    tools,
    agents,
    routers,
    edges: serializeEdges(edges),
  };
}

export function executableInfo(wf: WorkflowSpec): { ok: boolean; reason: string; nodeId?: string } {
  const agents = wf.agents && (wf.agents.length > 0 || wf.topology === 'graph') ? wf.agents : [{ id: 'hub' }];
  if (!agents.length) return { ok: false, reason: '请先添加一个 Agent，并设置运行入口。' };
  const agentIds = new Set(agents.map((a) => a.id));
  const toolIds = new Set([
    ...(wf.tools || []),
    ...agents.filter((agent) => agent.kind === 'tool').map((agent) => agent.id),
  ]);
  const routerIds = new Set((wf.routers || []).map((router) => router.id));
  const entry = wf.entry_agent || 'hub';
  if (!agentIds.has(entry)) return { ok: false, reason: `入口 ${entry} 不存在，请重新设置运行入口。` };
  if (wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology) {
    return { ok: true, reason: 'hub_react' };
  }
  if (wf.topology === 'graph') {
    const inboundRoute: Record<string, number> = {};
    for (const e of wf.edges || []) {
      const kind = e.kind || 'message';
      if (routerIds.has(e.from) || routerIds.has(e.to) || kind === 'sample_barrier') continue;
      const toIsTool = toolIds.has(e.to);
      const fromIsTool = toolIds.has(e.from);
      if (kind === 'tool_call') {
        if (!agentIds.has(e.from)) {
          return { ok: false, reason: `工具调用的起点 ${e.from} 必须是 Agent。`, nodeId: e.from };
        }
        if (!toIsTool) {
          return { ok: false, reason: `工具调用的终点 ${e.to} 必须是 Tool 或 tool-agent。`, nodeId: e.to };
        }
        continue;
      }
      if (toIsTool || fromIsTool) {
        return { ok: false, reason: `${EDGE_LABELS[kind] || kind} 不能连接 Tool。`, nodeId: toIsTool ? e.to : e.from };
      }
      if (!agentIds.has(e.from) || !agentIds.has(e.to)) {
        return { ok: false, reason: '连线引用了不存在的 Agent。', nodeId: agentIds.has(e.from) ? e.from : e.to };
      }
      if (kind === 'route') {
        inboundRoute[e.to] = (inboundRoute[e.to] || 0) + 1;
        if (inboundRoute[e.to] > 1) {
          return { ok: false, reason: `${e.to} 只能有一条输入路由。`, nodeId: e.to };
        }
      }
    }
    for (const router of wf.routers || []) {
      for (const candidate of router.candidates) {
        const id = candidate.startsWith('blank:') ? candidate.slice('blank:'.length) : candidate;
        if (!agentIds.has(id) && !toolIds.has(id)) {
          return { ok: false, reason: `Router ${router.id} 的候选 ${candidate} 不存在。`, nodeId: router.id };
        }
      }
      if (router.strategy === 'score' && router.scorer && !agentIds.has(router.scorer)) {
        return { ok: false, reason: `Router ${router.id} 的 scorer ${router.scorer} 不存在。`, nodeId: router.id };
      }
    }
    return { ok: true, reason: 'graph_compiled' };
  }
  return { ok: false, reason: `暂不支持拓扑 ${wf.topology}。` };
}

import type { AgentKind, AgentSpec, RouterSpec, WorkflowSpec } from '../../../shared/api/types';
import type { EdgeKind, GraphNode, GraphEdge } from '../types';
import { EDGE_KINDS, edgeDefinition } from './edgeDefinitions';
import { resolveHandles, serializeEdges } from './edgeGeometry';
export type { AgentSpec, WorkflowSpec } from '../../../shared/api/types';

export { KNOWN_TOOLS } from './edgeDefinitions';

export const EDGE_LABELS: Record<string, string> = Object.fromEntries(EDGE_KINDS.map((kind) => [kind, edgeDefinition(kind).title]));

export function poolNodeId(routerId: string): string {
  return `pool_${routerId}`;
}

export function routerIdFromPool(poolId: string): string {
  return poolId.startsWith('pool_') ? poolId.slice('pool_'.length) : '';
}

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

function isUserWrap(agent: { profile?: { backend?: string }; meta?: { wraps?: string } }): boolean {
  const wraps = typeof agent.meta?.wraps === 'string' ? agent.meta.wraps : '';
  return agent.profile?.backend === 'user_space' || wraps.startsWith('EPC-AW.');
}

function isHubAgent(agent: { id: string; kind?: string; role?: string; profile?: { backend?: string }; meta?: { wraps?: string } }): boolean {
  if (isUserWrap(agent)) return false;
  const role = (agent.role || '').toLowerCase();
  const kind = (agent.kind || '').toLowerCase();
  return agent.id === 'hub' || kind === 'hub' || role === 'hub' || role === 'orchestrator';
}

function isExecutorAgent(agent: { id: string; kind?: string; role?: string; profile?: { backend?: string }; meta?: { wraps?: string } }): boolean {
  if (isUserWrap(agent)) return false;
  const role = (agent.role || '').toLowerCase();
  const kind = (agent.kind || '').toLowerCase();
  return agent.id === 'executor' || kind === 'executor' || role === 'executor';
}

function inferredKind(agent: AgentSpec): AgentKind {
  const role = (agent.role || '').toLowerCase();
  const kind = (agent.kind || '').toLowerCase();
  if (kind === 'planner' || kind === 'verifier' || kind === 'tool' || kind === 'blank') return kind;
  if (role === 'planner' || agent.id === 'planner') return 'planner';
  if (role === 'verifier') return 'verifier';
  if (role === 'tool') return 'tool';
  return 'blank';
}

export function workflowToFlow(
  wf: WorkflowSpec,
  modelNames: ReadonlyMap<string, string> = new Map(),
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const entry = wf.entry_agent || 'planner';
  const declared =
    wf.agents && (wf.agents.length > 0 || wf.topology === 'graph')
      ? wf.agents.map((agent) => ({ ...agent }))
      : [
          {
            id: 'planner',
            kind: 'planner' as const,
            role: wf.hub?.role || 'planner',
            skills: wf.hub?.skills || [],
            tools: wf.tools || [],
            system_prompt: wf.hub?.system_prompt || '',
            trainable: true,
          },
        ];

  const hubAgents = declared.filter(isHubAgent);
  const executorAgents = declared.filter(isExecutorAgent);
  const dropped = new Set([...hubAgents, ...executorAgents].map((agent) => agent.id));
  const agents = declared.filter((agent) => !dropped.has(agent.id));
  let plannerAgent = agents.find((agent) => agent.kind === 'planner' || agent.id === 'planner');
  if (!plannerAgent && hubAgents.length) {
    const hub = hubAgents[0];
    plannerAgent = {
      id: 'planner',
      kind: 'planner',
      role: 'planner',
      skills: [...(hub.skills || [])],
      tools: [...(hub.tools || [])],
      system_prompt: hub.system_prompt || '',
      trainable: true,
    };
    agents.unshift(plannerAgent);
  } else if (plannerAgent) {
    for (const hub of hubAgents) {
      if (!plannerAgent.system_prompt && hub.system_prompt) plannerAgent.system_prompt = hub.system_prompt;
      if (!(plannerAgent.skills || []).length && hub.skills?.length) plannerAgent.skills = [...(hub.skills || [])];
    }
  }

  const executorTools = executorAgents.flatMap((agent) => agent.tools || []);
  let routers = (wf.routers || []).map((router) => ({
    ...router,
    candidates: [...(router.candidates || [])].filter((id) => id !== 'hub' && id !== 'executor' && !dropped.has(id)),
  }));
  let synthesizedRouter = false;
  if (executorTools.length) {
    if (!routers.length) {
      synthesizedRouter = true;
      routers = [{ id: 'route_exec', candidates: [], strategy: 'from_plan' }];
    }
    const [first, ...rest] = routers;
    const candidates = [...(first.candidates || [])];
    for (const tool of executorTools) {
      if (!candidates.includes(tool)) candidates.push(tool);
    }
    routers = [{ ...first, candidates }, ...rest];
  }

  const nodes: GraphNode[] = agents.filter((agent) => (agent.kind || inferredKind(agent)) !== 'tool').map((a, i) => ({
    id: a.id,
    type: 'agent',
    position: { x: 80 + (i % 3) * 260, y: 80 + Math.floor(i / 3) * 170 },
    data: {
      label: (typeof a.label === 'string' && a.label.trim()) ? a.label.trim() : a.id,
      kind: a.kind || inferredKind(a),
      role: a.role || 'agent',
      skills: a.skills || [],
      tools: a.tools || (a.id === 'planner' || a.kind === 'planner' ? wf.tools : []),
      memory_scope: a.memory_scope || 'agent',
      system_prompt: a.system_prompt ?? (a.id === 'planner' ? wf.hub?.system_prompt : '') ?? '',
      model: a.model || 'inherit',
      model_name: a.model && a.model !== 'inherit' ? modelNames.get(a.model) || a.model : undefined,
      trainable: a.trainable !== false,
      profile: a.profile || {},
      meta: a.meta || {},
      backend: typeof a.profile?.backend === 'string' ? a.profile.backend : undefined,
      llm_required: a.profile?.llm_required === true,
      entry: a.id === entry,
      verify: a.kind === 'planner' || a.id === 'planner' ? wf.hub?.verify : undefined,
      max_feedback_hops: a.kind === 'planner' || a.id === 'planner' ? wf.hub?.max_feedback_hops : undefined,
    },
  }));

  if (wf.hub?.verify && nodes.some((n) => n.data.kind === 'planner') && !nodes.find((n) => n.id === 'verifier')) {
    nodes.push({
      id: 'verifier',
      type: 'agent',
      position: { x: 860, y: 80 },
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

  routers.forEach((router, index) => {
    nodes.push({
      id: router.id,
      type: 'router',
      position: { x: 360 + (index % 2) * 280, y: 80 + Math.floor(index / 2) * 200 },
      data: {
        label: router.id,
        candidates: [...(router.candidates || [])],
        strategy: router.strategy || 'from_plan',
        scorer: router.scorer ?? null,
        meta: router.meta || {},
      },
    });
    nodes.push({
      id: poolNodeId(router.id),
      type: 'pool',
      position: { x: 620 + (index % 2) * 280, y: 80 + Math.floor(index / 2) * 200 },
      data: {
        label: 'tool-agent pool',
        members: [...(router.candidates || [])],
        memberLabels: Object.fromEntries(
          (router.candidates || []).map((id) => {
            const agent = declared.find((item) => item.id === id);
            const label = typeof agent?.label === 'string' && agent.label.trim() ? agent.label.trim() : id;
            return [id, label];
          }),
        ),
        memberTiers: Object.fromEntries(
          (router.candidates || []).flatMap((id) => {
            const tier = declared.find((agent) => agent.id === id)?.profile?.tier;
            return tier === 'lite' || tier === 'pro' ? [[id, tier]] : [];
          }),
        ),
        routerId: router.id,
        candidates: [...(router.candidates || [])],
      },
    });
  });

  const routerIds = new Set(routers.map((router) => router.id));
  const executorNext = (wf.edges || []).find((edge) =>
    executorAgents.some((agent) => agent.id === edge.from) && (edge.kind || 'message') === 'message' && !dropped.has(edge.to));
  const rawEdges = (wf.edges || [])
    .filter((edge) => !dropped.has(edge.from) && !dropped.has(edge.to))
    .map((edge) => edge.to === 'hub' && (edge.kind || 'message') === 'feedback' ? { ...edge, to: plannerAgent?.id || 'planner' } : edge);
  if (synthesizedRouter && routers[0] && !rawEdges.some((edge) => edge.to === routers[0].id)) {
    const plannerId = nodes.find((node) => node.data.kind === 'planner')?.id || 'planner';
    rawEdges.unshift({ from: plannerId, to: routers[0].id, kind: 'route' });
  }
  if (synthesizedRouter && routers[0] && executorNext && !rawEdges.some((edge) => edge.from === routers[0].id)) {
    rawEdges.push({ from: routers[0].id, to: executorNext.to, kind: 'message' });
  }

  const edges: GraphEdge[] = [];
  rawEdges.forEach((edge, index) => {
    const kind = edge.kind || 'message';
    if (routerIds.has(edge.from) && !edge.to.startsWith('pool_') && (kind === 'message' || kind === 'route')) {
      const poolId = poolNodeId(edge.from);
      if (!edges.some((item) => item.source === edge.from && item.target === poolId)) {
        edges.push(graphEdge(edge.from, poolId, 'route', `e-${edge.from}-${poolId}`));
      }
      if (!edges.some((item) => item.source === poolId && item.target === edge.to)) {
        edges.push(graphEdge(poolId, edge.to, 'message', `e-${poolId}-${edge.to}-${index}`));
      }
      return;
    }
    edges.push({
      ...graphEdge(edge.from, edge.to, kind, `e-${edge.from}-${edge.to}-${index}`),
      data: { kind, meta: edge.meta },
    });
  });
  for (const router of routers) {
    const poolId = poolNodeId(router.id);
    if (!edges.some((edge) => edge.source === router.id && edge.target === poolId)) {
      edges.push(graphEdge(router.id, poolId, 'route', `e-${router.id}-${poolId}`));
    }
  }

  const skipToolCallEdges = routers.length > 0
    || wf.topology === 'centralized' || wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology;
  if (!skipToolCallEdges) {
    for (const node of nodes.filter((n) => n.type === 'agent' && n.data.kind !== 'tool')) {
      for (const tool of node.data.tools || []) {
        if (!nodes.some((item) => item.id === tool)) continue;
        if (!edges.some((e) => e.source === node.id && e.target === tool && e.data?.kind === 'tool_call')) {
          edges.push(graphEdge(node.id, tool, 'tool_call'));
        }
      }
      node.data.tools = edges.filter((e) => e.source === node.id && e.data?.kind === 'tool_call').map((e) => e.target);
    }
  }
  const plannerId = nodes.find((n) => n.data.kind === 'planner')?.id || 'planner';
  if (wf.hub?.verify && nodes.some((n) => n.data.kind === 'planner') && !edges.some((e) => e.source === 'verifier' && e.target === plannerId && e.data?.kind === 'feedback')) {
    edges.push(graphEdge('verifier', plannerId, 'feedback'));
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  return { nodes, edges: edges.map((edge) => resolveHandles(edge, nodeById)) };
}

export function flowToWorkflow(
  nodes: GraphNode[],
  edges: GraphEdge[],
  base: WorkflowSpec,
): WorkflowSpec {
  const agentNodes = nodes.filter((n) => n.type === 'agent' && n.data.kind !== 'tool');
  const toolNodes = nodes.filter((n) => n.type === 'tool' || (n.type === 'agent' && n.data.kind === 'tool'));
  const routerNodes = nodes.filter((n) => n.type === 'router');
  const poolNodes = nodes.filter((n) => n.type === 'pool');
  const plannerNode = agentNodes.find((n) => n.data.kind === 'planner' || n.id === 'planner') || agentNodes[0];
  const entry =
    agentNodes.find((n) => n.data?.entry)?.id ||
    plannerNode?.id ||
    '';

  const previousAgents = new Map((base.agents || []).map((a) => [a.id, a]));
  const agents: Array<AgentSpec & { tools: string[] }> = agentNodes.map((n) => {
    const previous = previousAgents.get(n.id);
    const kind = n.data.kind || previous?.kind || inferredKind({ id: n.id, role: n.data.role });
    return {
      ...previous,
      id: n.id,
      kind,
      role: n.data.role || 'agent',
      label: (typeof n.data.label === 'string' && n.data.label.trim() && n.data.label !== n.id)
        ? n.data.label.trim()
        : (typeof previous?.label === 'string' ? previous.label : undefined),
      skills: n.data.skills || [],
      tools: [...(n.data.tools || [])],
      memory_scope: n.data.memory_scope || 'agent',
      system_prompt: n.data.system_prompt || '',
      model: n.data.model || 'inherit',
      trainable: kind === 'tool' ? false : n.data.trainable !== false,
      profile: n.data.profile || {},
      meta: n.data.meta || {},
    };
  });

  const poolByRouter = new Map(poolNodes.map((pool) => [pool.data.routerId || routerIdFromPool(pool.id), pool]));
  for (const pool of poolNodes) {
    for (const member of pool.data.members || []) {
      const existing = agents.find((agent) => agent.id === member);
      const tier = pool.data.memberTiers?.[member];
      if (existing) {
        if (existing.kind === 'tool') {
          existing.trainable = false;
          if (tier === 'lite' || tier === 'pro') {
            existing.profile = { ...(existing.profile || {}), tier };
          }
        }
        continue;
      }
      const previous = previousAgents.get(member);
      const kind = previous?.kind === 'blank' || previous?.kind === 'planner' || previous?.kind === 'verifier'
        ? previous.kind
        : 'tool';
      agents.push({
        ...previous,
        id: member,
        kind,
        role: previous?.role || (kind === 'tool' ? 'tool' : 'agent'),
        label: typeof previous?.label === 'string' ? previous.label : undefined,
        skills: previous?.skills || [],
        tools: [...(previous?.tools || [])],
        memory_scope: previous?.memory_scope || (kind === 'tool' ? 'none' : 'agent'),
        system_prompt: previous?.system_prompt || '',
        model: previous?.model || 'inherit',
        trainable: kind === 'tool' ? false : previous?.trainable !== false,
        profile: {
          ...(previous?.profile || {}),
          ...(kind === 'tool' && (tier === 'lite' || tier === 'pro') ? { tier } : {}),
        },
        meta: previous?.meta || {},
      });
    }
  }

  const agentById = new Map(agents.map((agent) => [agent.id, agent]));
  for (const e of edges) {
    if (String(e.data?.kind || e.label) !== 'tool_call') continue;
    const agent = agentById.get(e.source);
    if (agent && !agent.tools.includes(e.target)) agent.tools.push(e.target);
  }

  const tools = Array.from(
    new Set([
      ...(base.tools || []),
      ...agents.filter((agent) => agent.kind === 'tool').map((agent) => agent.id),
      ...agents.flatMap((a) => a.tools),
      ...toolNodes.map((n) => n.id),
    ]),
  );

  const previousRouters = new Map((base.routers || []).map((router) => [router.id, router]));
  const routers: RouterSpec[] = routerNodes.map((node) => {
    const pool = poolByRouter.get(node.id);
    return {
      ...previousRouters.get(node.id),
      id: node.id,
      candidates: [...(pool?.data.members || node.data.candidates || [])],
      strategy: node.data.strategy || 'from_plan',
      scorer: node.data.scorer || null,
      meta: node.data.meta || {},
    };
  });

  const collapsed: GraphEdge[] = [];
  for (const edge of edges) {
    if (edge.target.startsWith('pool_')) continue;
    if (edge.source.startsWith('pool_')) {
      const routerId = routerIdFromPool(edge.source);
      if (!routerId || !edge.target || edge.target.startsWith('pool_')) continue;
      if (!collapsed.some((item) => item.source === routerId && item.target === edge.target && item.data?.kind === 'message')) {
        collapsed.push({ ...edge, source: routerId, data: { ...edge.data, kind: 'message' } });
      }
      continue;
    }
    const target = edge.target === 'hub' && edge.data?.kind === 'feedback' ? (plannerNode?.id || 'planner') : edge.target;
    collapsed.push(target === edge.target ? edge : { ...edge, target });
  }

  const skills = plannerNode?.data.skills || [];
  const verify = plannerNode?.data.verify ?? null;
  const maxHops =
    plannerNode?.data.max_feedback_hops ?? base.hub?.max_feedback_hops ?? 1;
  const hubPrompt = plannerNode?.data.system_prompt || '';

  const topology = routers.length || agents.some((a) => a.kind === 'tool' || a.kind === 'blank')
    ? (base.topology === 'graph' ? 'graph' : 'centralized')
    : 'centralized';

  return {
    ...base,
    schema_version: '0.3',
    topology,
    entry_agent: entry || 'planner',
    hub: {
      ...base.hub,
      role: 'planner',
      skills,
      verify: verify || null,
      max_feedback_hops: maxHops,
      system_prompt: hubPrompt,
    },
    tools,
    agents,
    routers,
    edges: serializeEdges(collapsed),
  };
}

export function executableInfo(wf: WorkflowSpec): { ok: boolean; reason: string; nodeId?: string } {
  const agents = wf.agents && (wf.agents.length > 0 || wf.topology === 'graph') ? wf.agents : [{ id: 'planner', kind: 'planner' as const }];
  if (!agents.length) return { ok: false, reason: '请先添加一个 Agent，并设置运行入口。' };
  const agentIds = new Set(agents.map((a) => a.id));
  const toolIds = new Set([
    ...(wf.tools || []),
    ...agents.filter((agent) => agent.kind === 'tool').map((agent) => agent.id),
  ]);
  const routerIds = new Set((wf.routers || []).map((router) => router.id));
  const entry = wf.entry_agent || 'planner';
  if (!agentIds.has(entry)) return { ok: false, reason: `入口 ${entry} 不存在，请重新设置运行入口。` };
  if (wf.topology === 'centralized' || wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology) {
    return { ok: true, reason: 'centralized' };
  }
  if (wf.topology === 'graph') {
    const inboundRoute: Record<string, number> = {};
    for (const e of wf.edges || []) {
      const kind = e.kind || 'message';
      if (e.from.startsWith('pool_') || e.to.startsWith('pool_')) continue;
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

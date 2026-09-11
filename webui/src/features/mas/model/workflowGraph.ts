import { MarkerType } from '@xyflow/react';
import type { AgentSpec, WorkflowSpec } from '../../../shared/api/types';
import type { GraphNode, GraphEdge } from '../types';
export type { AgentSpec, WorkflowSpec } from '../../../shared/api/types';

export const KNOWN_TOOLS = ['web_search', 'wikipedia_search', 'execute_python'];

export const EDGE_LABELS: Record<string, string> = {
  message: '传递消息', route: '任务路由', feedback: '反馈', tool_call: '调用工具',
};

export function graphEdge(source: string, target: string, kind: string, id = `e-${source}-${target}-${kind}`): GraphEdge {
  return { id, source, target, label: EDGE_LABELS[kind] || kind, data: { kind },
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 } };
}

export function workflowToFlow(wf: WorkflowSpec): { nodes: GraphNode[]; edges: GraphEdge[] } {
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
      role: a.role || 'agent',
      skills: a.skills || [],
      tools: a.tools || (a.id === 'hub' ? wf.tools : []),
      system_prompt: a.system_prompt ?? (a.id === 'hub' ? wf.hub?.system_prompt : '') ?? '',
      trainable: a.trainable !== false,
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

  const edges = (wf.edges || []).map((e, i) =>
    graphEdge(e.from, e.to, e.kind || 'message', `e-${e.from}-${e.to}-${i}`));

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

  return { nodes, edges };
}

export function flowToWorkflow(
  nodes: GraphNode[],
  edges: GraphEdge[],
  base: WorkflowSpec,
): WorkflowSpec {
  const agentNodes = nodes.filter((n) => n.type !== 'tool');
  const toolNodes = nodes.filter((n) => n.type === 'tool');
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
      role: n.data.role || 'agent',
      skills: n.data.skills || [],
      tools: [...(n.data.tools || [])],
      memory_scope: previous?.memory_scope || 'agent',
      system_prompt: n.data.system_prompt || '',
      model: previous?.model || 'inherit',
      trainable: n.data.trainable !== false,
    };
  });

  for (const e of edges) {
    if (String(e.data?.kind || e.label) !== 'tool_call') continue;
    const agent = agents.find((a) => a.id === e.source);
    if (agent && !agent.tools.includes(e.target)) agent.tools.push(e.target);
  }

  const tools = Array.from(
    new Set([
      ...agents.flatMap((a) => a.tools),
      ...toolNodes.map((n) => n.id),
    ]),
  );

  const skills = hubNode?.data.skills || [];
  const verify = hubNode?.data.verify ?? null;
  const maxHops =
    hubNode?.data.max_feedback_hops ?? base.hub?.max_feedback_hops ?? 1;
  const hubPrompt =
    hubNode?.data.system_prompt || '';

  const edgeSpecs = edges.map((e) => ({
    from: e.source,
    to: e.target,
    kind: String(e.data?.kind || e.label || 'message'),
  }));

  const onlyHubish = agents.every((a) => a.id === 'hub' || a.id === 'verifier');
  const topology =
    hubNode?.id === 'hub' && agents.length <= 2 && onlyHubish
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
      ...base.hub,
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

export function executableInfo(wf: WorkflowSpec): { ok: boolean; reason: string; nodeId?: string } {
  const agents = wf.agents && (wf.agents.length > 0 || wf.topology === 'graph') ? wf.agents : [{ id: 'hub' }];
  if (!agents.length) return { ok: false, reason: '请先添加一个 Agent，并设置运行入口。' };
  const agentIds = new Set(agents.map((a) => a.id));
  const entry = wf.entry_agent || 'hub';
  if (!agentIds.has(entry)) return { ok: false, reason: `入口 ${entry} 不存在，请重新设置运行入口。` };
  if (wf.topology === 'hub_react' || wf.topology === 'single' || !wf.topology) {
    return { ok: true, reason: 'hub_react' };
  }
  if (wf.topology === 'graph') {
    const inboundRoute: Record<string, number> = {};
    for (const e of wf.edges || []) {
      const kind = e.kind || 'message';
      const toIsTool = wf.tools.includes(e.to);
      const fromIsTool = wf.tools.includes(e.from);
      if (kind === 'tool_call') {
        if (!agentIds.has(e.from)) {
          return { ok: false, reason: `工具调用的起点 ${e.from} 必须是 Agent。`, nodeId: e.from };
        }
        if (agentIds.has(e.to)) {
          return { ok: false, reason: '工具调用的终点不能是 Agent。', nodeId: e.to };
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
    return { ok: true, reason: 'graph_compiled' };
  }
  return { ok: false, reason: `暂不支持拓扑 ${wf.topology}。` };
}

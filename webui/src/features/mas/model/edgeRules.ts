import type { Connection } from '@xyflow/react';
import type { Palette } from '../../../shared/api/types';
import type { EdgeKind, GraphEdge, GraphNode } from '../types';
import { EDGE_KINDS, KNOWN_TOOLS, edgeDefinition, isEdgeKind } from './edgeDefinitions';

export interface RelationOption { kind: EdgeKind; reason: string }
export type EdgeRules = ReturnType<typeof createEdgeRules>;

export function edgeConnection(edge: GraphEdge): Connection {
  return {
    source: edge.source, target: edge.target,
    sourceHandle: edge.sourceHandle ?? null, targetHandle: edge.targetHandle ?? null,
  };
}

export function reverseConnection(connection: Connection): Connection {
  return {
    source: connection.target, target: connection.source,
    sourceHandle: connection.targetHandle, targetHandle: connection.sourceHandle,
  };
}

export function createEdgeRules(nodes: GraphNode[], edges: GraphEdge[], palette: Palette = {}) {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const outgoing = new Map<string, GraphEdge[]>();
  const incomingRoutes = new Map<string, GraphEdge[]>();
  const tools = new Set(palette.tools ?? KNOWN_TOOLS);
  const supported = new Set(palette.edge_kinds ?? EDGE_KINDS);
  for (const edge of edges) {
    const list = outgoing.get(edge.source) || [];
    list.push(edge);
    outgoing.set(edge.source, list);
    if (edge.data?.kind === 'route') {
      const routes = incomingRoutes.get(edge.target) || [];
      routes.push(edge);
      incomingRoutes.set(edge.target, routes);
    }
  }

  const normalize = (connection: Connection): Connection =>
    nodeById.get(connection.source)?.type === 'tool' && nodeById.get(connection.target)?.type === 'agent'
      ? reverseConnection(connection) : connection;

  const normalizeEdit = (connection: Connection, edge: GraphEdge, kind = edge.data?.kind || 'message'): Connection => {
    const geometryOnly = connection.source === edge.source && connection.target === edge.target && kind === edge.data?.kind;
    return geometryOnly || kind !== 'tool_call' ? connection : normalize(connection);
  };

  const error = (connection: Connection, kind: string, replacing?: string): string => {
    const source = nodeById.get(connection.source);
    const target = nodeById.get(connection.target);
    if (!source || !target) return '连接对象已不存在。';
    if (source.id === target.id) return '请连接两个不同的实体。';
    if (!isEdgeKind(kind) || !supported.has(kind)) return '当前不支持此关系类型。';
    if (source.type === 'tool' && target.type === 'tool') return '工具之间不能直接连接。';
    if (kind === 'tool_call') {
      if (source.type !== 'agent' || target.type !== 'tool') return '工具关系必须由 Agent 指向 Tool。';
      if (!tools.has(target.id)) return '当前运行时不支持该工具。';
    } else if (source.type !== 'agent' || target.type !== 'agent') {
      return 'Agent 协作不能连接 Tool。';
    }
    const other = (outgoing.get(source.id) || []).filter((edge) => edge.id !== replacing);
    if (other.some((edge) => edge.target === target.id && edge.data?.kind === kind)) {
      return kind === 'tool_call' ? '该 Agent 已具备此工具。' : '该关系已经存在。';
    }
    if (kind === 'tool_call') return '';
    if (kind === 'feedback') {
      const matches = (node: GraphNode, roles: string[]) => roles.includes(node.id) || roles.includes(node.data.role || '');
      if (!matches(source, ['verifier', 'critic'])) return '反馈方必须是 verifier 或 critic。';
      if (!matches(target, ['hub', 'planner', 'orchestrator'])) return '反馈接收方必须是 hub、planner 或 orchestrator。';
    }
    if (kind === 'route' && (incomingRoutes.get(target.id) || []).some((edge) => edge.id !== replacing)) {
      return '接收方已有输入路由，最多只能接受一条。';
    }
    if (other.some((edge) => edge.data?.kind === kind)) {
      return `发送方已有${edgeDefinition(kind).title}目标，当前运行时不支持同类型多目标输出。`;
    }
    if ((kind === 'route' || kind === 'message') && other.some((edge) =>
      edge.data?.kind === (kind === 'route' ? 'message' : 'route'))) {
      return '同一发送方不能同时配置消息与路由：路由会遮蔽消息，不会并行执行。';
    }
    return '';
  };

  const options = (raw: Connection, replacing?: string): RelationOption[] => {
    const connection = normalize(raw);
    const hasTool = [connection.source, connection.target].some((id) => nodeById.get(id)?.type === 'tool');
    const kinds = hasTool ? ['tool_call'] as const : EDGE_KINDS.filter((kind) => kind !== 'tool_call');
    return kinds.map((kind) => ({ kind, reason: error(connection, kind, replacing) }));
  };

  return { nodeById, normalize, normalizeEdit, error, options };
}

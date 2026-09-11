import type { Config, WorkflowEdgeSpec } from '../../../shared/api/types';
import type { GraphEdge, GraphNode, HandleId } from '../types';

function handle(value: unknown): HandleId | undefined {
  return value === 'top' || value === 'right' || value === 'bottom' || value === 'left' ? value : undefined;
}

function uiMeta(meta?: Config): Config {
  return meta?.ui && typeof meta.ui === 'object' && !Array.isArray(meta.ui) ? meta.ui as Config : {};
}

export function defaultHandles(source?: GraphNode, target?: GraphNode) {
  const dx = (target?.position.x ?? 1) - (source?.position.x ?? 0);
  const dy = (target?.position.y ?? 0) - (source?.position.y ?? 0);
  if (Math.abs(dx) >= Math.abs(dy)) return {
    sourceHandle: dx >= 0 ? 'right' : 'left', targetHandle: dx >= 0 ? 'left' : 'right',
  };
  return { sourceHandle: dy >= 0 ? 'bottom' : 'top', targetHandle: dy >= 0 ? 'top' : 'bottom' };
}

export function resolveHandles(edge: GraphEdge, nodes: Map<string, GraphNode>): GraphEdge {
  const defaults = defaultHandles(nodes.get(edge.source), nodes.get(edge.target));
  const ui = uiMeta(edge.data?.meta);
  return {
    ...edge,
    sourceHandle: handle(edge.sourceHandle) || handle(ui.sourceHandle) || defaults.sourceHandle,
    targetHandle: handle(edge.targetHandle) || handle(ui.targetHandle) || defaults.targetHandle,
  };
}

export function serializeEdges(edges: GraphEdge[]): WorkflowEdgeSpec[] {
  return edges.map((edge) => ({
    from: edge.source, to: edge.target, kind: edge.data?.kind || 'message',
    meta: {
      ...edge.data?.meta,
      ui: { ...uiMeta(edge.data?.meta), sourceHandle: edge.sourceHandle, targetHandle: edge.targetHandle },
    },
  }));
}

export function edgeLanes(edges: GraphEdge[]): Map<string, number> {
  const groups = new Map<string, GraphEdge[]>();
  for (const edge of edges) {
    const key = JSON.stringify([edge.source, edge.target].sort());
    const group = groups.get(key) || [];
    group.push(edge);
    groups.set(key, group);
  }
  const lanes = new Map<string, number>();
  for (const group of groups.values()) {
    group.sort((a, b) => `${a.source}:${a.data?.kind}:${a.id}`.localeCompare(`${b.source}:${b.data?.kind}:${b.id}`));
    group.forEach((edge, index) => lanes.set(edge.id, (index - (group.length - 1) / 2) * 38));
  }
  return lanes;
}

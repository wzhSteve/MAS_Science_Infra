import type { SamplingOpportunity } from '../../../shared/api/types';
import type { GraphEdge } from '../../mas/types';

export function opportunitiesByEdge(
  opportunities: SamplingOpportunity[],
  edges: GraphEdge[],
): Map<string, SamplingOpportunity> {
  const byId = new Map(edges.map(edge => [edge.id, edge]));
  const output = new Map<string, SamplingOpportunity>();
  for (const opportunity of opportunities) {
    const edge = opportunity.edge_id ? byId.get(opportunity.edge_id) : undefined;
    const relation = edge || edges.find(item =>
      item.source === opportunity.edge_source
      && item.target === opportunity.edge_target
      && item.data?.kind === 'tool_call');
    if (relation) output.set(relation.id, opportunity);
  }
  return output;
}

export function edgeForOpportunity(
  mapped: Map<string, SamplingOpportunity>,
  opportunityId?: string | null,
): string | null {
  if (!opportunityId) return null;
  for (const [edgeId, opportunity] of mapped) {
    if (opportunity.id === opportunityId) return edgeId;
  }
  return null;
}

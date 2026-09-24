import { describe, expect, it } from 'vitest';
import type { SamplingOpportunity } from '../../../shared/api/types';
import type { GraphEdge } from '../../mas/types';
import { edgeForOpportunity, opportunitiesByEdge } from './samplingCanvas';

const edge = (id: string, source: string, target: string): GraphEdge => ({
  id, source, target, type: 'workflow', data: { kind: 'tool_call' },
});

const opportunity = (patch: Partial<SamplingOpportunity> = {}): SamplingOpportunity => ({
  id: 'tool_result:hub:tool_id=execute_python',
  node_id: 'execute_python',
  edge_source: 'hub',
  edge_target: 'execute_python',
  anchor: { kind: 'after_tool', agent_id: 'hub', tool_id: 'execute_python' },
  label: 'execute_python 结果返回 hub 后',
  support: 'native',
  message: 'Tool Result Window',
  allowed_gates: ['entropy_delta', 'always'],
  configured: false,
  enabled: false,
  ...patch,
});

describe('sampling canvas opportunity mapping', () => {
  it('maps explicit and implicit tool relations without changing runtime identity', () => {
    const explicit = edge('e-hub-execute_python-0', 'hub', 'execute_python');
    const implicit = edge('edge-runtime-generated', 'planner', 'expert');
    const mapped = opportunitiesByEdge([
      opportunity({ edge_id: explicit.id }),
      opportunity({
        id: 'tool_result:planner:tool_id=blank:expert',
        node_id: 'expert',
        edge_source: 'planner',
        edge_target: 'expert',
        anchor: { kind: 'after_tool', agent_id: 'planner', tool_id: 'blank:expert' },
      }),
    ], [explicit, implicit]);

    expect(mapped.get(explicit.id)?.anchor.tool_id).toBe('execute_python');
    expect(mapped.get(implicit.id)?.anchor.tool_id).toBe('blank:expert');
    expect(edgeForOpportunity(mapped, 'tool_result:planner:tool_id=blank:expert')).toBe(implicit.id);
  });
});

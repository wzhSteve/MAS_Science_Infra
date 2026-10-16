import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';
import type { WorkflowSpec } from '../../../shared/api/types';
import { DEFAULT_PLANNER_PROMPT, DEFAULT_VERIFIER_PROMPT } from './defaultPrompts';
import { useGraphEditor } from './useGraphEditor';
import { flowToWorkflow } from './workflowGraph';

function useEditor(initial: WorkflowSpec) {
  const [workflow, setWorkflow] = useState(initial);
  return useGraphEditor(workflow, setWorkflow);
}

const empty: WorkflowSpec = {
  schema_version: '0.3',
  topology: 'centralized',
  entry_agent: 'planner',
  hub: { role: 'planner', skills: [], system_prompt: '' },
  tools: [],
  agents: [{ id: 'planner', kind: 'planner', role: 'planner', system_prompt: '', trainable: true }],
  routers: [],
  edges: [],
};

describe('dragged centralized graph', () => {
  it('wires planner, router, one tool and verifier with default prompts', () => {
    const { result } = renderHook(() => useEditor(empty));
    act(() => {
      result.current.addNode({ nodeType: 'agent', id: 'planner', agentKind: 'planner', role: 'planner' }, { x: 0, y: 0 });
      result.current.addNode({ nodeType: 'router', id: 'router' }, { x: 200, y: 0 });
      result.current.addNode({ nodeType: 'tool', id: 'base_generator', agentKind: 'tool' }, { x: 400, y: 0 });
      result.current.addNode({ nodeType: 'agent', id: 'verifier', agentKind: 'verifier', role: 'verifier' }, { x: 600, y: 0 });
    });
    const { nodes, edges } = result.current;
    const planner = nodes.find((node) => node.data.kind === 'planner');
    const verifier = nodes.find((node) => node.data.kind === 'verifier');
    expect(planner?.data.system_prompt).toBe(DEFAULT_PLANNER_PROMPT);
    expect(verifier?.data.system_prompt).toBe(DEFAULT_VERIFIER_PROMPT);
    expect(edges.some((edge) => edge.source === 'planner' && edge.target === 'router' && edge.data?.kind === 'route')).toBe(true);
    expect(edges.some((edge) => edge.source === 'pool_router' && edge.target === 'verifier' && edge.data?.kind === 'message')).toBe(true);
    expect(edges.some((edge) => edge.source === 'verifier' && edge.target === 'planner' && edge.data?.kind === 'feedback')).toBe(true);

    const saved = flowToWorkflow(nodes, edges, empty);
    expect(saved.routers?.[0]?.candidates).toEqual(['base_generator']);
    expect(saved.edges?.map((edge) => edge.kind).sort()).toEqual(['feedback', 'message', 'route']);
    expect(saved.agents?.find((agent) => agent.id === 'planner')?.system_prompt).toBe(DEFAULT_PLANNER_PROMPT);
    expect(saved.agents?.find((agent) => agent.id === 'verifier')?.system_prompt).toBe(DEFAULT_VERIFIER_PROMPT);
  });

  it('does not replace a written prompt or an existing edge', () => {
    const written: WorkflowSpec = {
      ...empty,
      hub: { role: 'planner', skills: [], system_prompt: 'keep planner' },
      agents: [{ id: 'planner', kind: 'planner', role: 'planner', system_prompt: 'keep planner', trainable: true }],
      routers: [{ id: 'router', candidates: ['base_generator'], strategy: 'from_plan' }],
      edges: [{ from: 'planner', to: 'router', kind: 'route' }],
    };
    const { result } = renderHook(() => useEditor(written));
    act(() => {
      result.current.addNode({ nodeType: 'agent', id: 'planner', agentKind: 'planner', role: 'planner' }, { x: 0, y: 0 });
      result.current.addNode({ nodeType: 'agent', id: 'verifier', agentKind: 'verifier', role: 'verifier' }, { x: 600, y: 0 });
    });
    const planner = result.current.nodes.find((node) => node.data.kind === 'planner');
    expect(planner?.data.system_prompt).toBe('keep planner');
    const routes = result.current.edges.filter((edge) => edge.source === 'planner' && edge.data?.kind === 'route');
    expect(routes).toHaveLength(1);
    expect(routes[0]?.target).toBe('router');
  });
});

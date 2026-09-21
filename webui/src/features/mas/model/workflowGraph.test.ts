import { describe, expect, it } from 'vitest';
import type { WorkflowSpec } from '../../../shared/api/types';
import { createEdgeRules } from './edgeRules';
import { executableInfo, flowToWorkflow, workflowToFlow } from './workflowGraph';

const sampling = {
  mode: 'arpo',
  group_n: 4,
  sites: [{
    id: 'after-hub',
    anchor: { kind: 'after_agent_turn', agent_id: 'hub' },
    gate: { type: 'entropy_delta', params: { threshold: 0.2 } },
  }],
};

describe('workflowGraph schema 0.3', () => {
  it('keeps legacy workflow fields and sampling when the graph is serialized', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.1.0',
      topology: 'hub_react',
      entry_agent: 'hub',
      hub: { role: 'orchestrator', skills: ['react_loop'] },
      tools: ['web_search'],
      agents: [{
        id: 'hub',
        role: 'orchestrator',
        skills: ['react_loop'],
        tools: ['web_search'],
        trainable: true,
        extension: { keep: true },
      }],
      edges: [],
      sampling,
      extension: { owner: 'research' },
    };

    const original = structuredClone(workflow);
    const graph = workflowToFlow(workflow);
    const serialized = flowToWorkflow(graph.nodes, graph.edges, workflow);

    expect(workflow).toEqual(original);
    expect(serialized.schema_version).toBe('0.3');
    expect(serialized.tools).toContain('web_search');
    expect(serialized.sampling).toEqual(sampling);
    expect(serialized.extension).toEqual({ owner: 'research' });
    expect(serialized.agents?.[0].extension).toEqual({ keep: true });
  });

  it('round-trips complete agent and router fields', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'planner',
      hub: { role: 'orchestrator', skills: ['react_loop'] },
      tools: ['execute_python'],
      agents: [
        {
          id: 'planner',
          kind: 'planner',
          role: 'research_planner',
          skills: ['planning'],
          tools: ['execute_python'],
          memory_scope: 'shared',
          system_prompt: 'Plan carefully',
          model: 'inherit',
          trainable: true,
          profile: { temperature: 0.2 },
          meta: { owner: 'team-a' },
        },
        {
          id: 'execute_python',
          kind: 'tool',
          role: 'tool',
          trainable: false,
          profile: { backend: 'pure' },
          meta: { capability: 'python' },
        },
      ],
      routers: [{
        id: 'expert_router',
        candidates: ['execute_python', 'planner'],
        strategy: 'score',
        scorer: 'planner',
        meta: { purpose: 'expert-choice' },
        extension: { keep: true },
      }],
      edges: [{ from: 'planner', to: 'expert_router', kind: 'route', meta: { channel: 'decision' } }],
      sampling,
    };

    const graph = workflowToFlow(workflow);
    const router = graph.nodes.find((node) => node.type === 'router');
    expect(router?.data.candidates).toEqual(['execute_python', 'planner']);

    const editedNodes = graph.nodes.map((node) => node.id === 'expert_router'
      ? { ...node, data: { ...node.data, strategy: 'round_robin' as const } }
      : node);
    const serialized = flowToWorkflow(editedNodes, graph.edges, workflow);

    expect(serialized.entry_agent).toBe('planner');
    expect(serialized.agents?.find((agent) => agent.id === 'planner')).toMatchObject({
      kind: 'planner',
      role: 'research_planner',
      memory_scope: 'shared',
      model: 'inherit',
      profile: { temperature: 0.2 },
      meta: { owner: 'team-a' },
    });
    expect(serialized.agents?.find((agent) => agent.id === 'execute_python')).toMatchObject({
      kind: 'tool',
      profile: { backend: 'pure' },
      meta: { capability: 'python' },
    });
    expect(serialized.routers).toEqual([{
      id: 'expert_router',
      candidates: ['execute_python', 'planner'],
      strategy: 'round_robin',
      scorer: 'planner',
      meta: { purpose: 'expert-choice' },
      extension: { keep: true },
    }]);
    expect(serialized.edges?.[0]).toMatchObject({
      from: 'planner',
      to: 'expert_router',
      kind: 'route',
      meta: { channel: 'decision' },
    });
    expect(serialized.sampling).toEqual(sampling);
    expect(executableInfo(serialized)).toEqual({ ok: true, reason: 'graph_compiled' });
  });

  it('uses route edges for routers and tool-call edges for tool agents', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'planner',
      hub: { role: 'orchestrator', skills: ['react_loop'] },
      tools: [],
      agents: [
        { id: 'planner', kind: 'planner' },
        { id: 'python_agent', kind: 'tool' },
      ],
      routers: [{ id: 'router', candidates: ['python_agent'] }],
      edges: [],
    };
    const graph = workflowToFlow(workflow);
    const rules = createEdgeRules(graph.nodes, graph.edges, {
      edge_kinds: ['message', 'tool_call', 'feedback', 'route', 'sample_barrier'],
    });

    expect(rules.options({
      source: 'planner', target: 'router', sourceHandle: null, targetHandle: null,
    })).toEqual([{ kind: 'route', reason: '' }]);
    expect(rules.error({
      source: 'planner', target: 'python_agent', sourceHandle: null, targetHandle: null,
    }, 'tool_call')).toBe('');
    expect(rules.error({
      source: 'planner', target: 'python_agent', sourceHandle: null, targetHandle: null,
    }, 'message')).toContain('Tool');
  });

  it('reports missing router candidates without changing sampling', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'hub',
      hub: { role: 'orchestrator', skills: ['react_loop'] },
      tools: [],
      agents: [{ id: 'hub', kind: 'hub' }],
      routers: [{ id: 'router', candidates: ['missing'], strategy: 'llm_choice' }],
      edges: [{ from: 'hub', to: 'router', kind: 'route' }],
      sampling,
    };

    expect(executableInfo(workflow)).toEqual({
      ok: false,
      reason: 'Router router 的候选 missing 不存在。',
      nodeId: 'router',
    });
    expect(workflow.sampling).toEqual(sampling);
  });
});

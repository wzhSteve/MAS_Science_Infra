import { describe, expect, it } from 'vitest';
import type { SamplingSpec, WorkflowSpec } from '../../../shared/api/types';
import { DEFAULT_PLANNER_PROMPT, DEFAULT_VERIFIER_PROMPT } from './defaultPrompts';
import { createEdgeRules } from './edgeRules';
import { executableInfo, flowToWorkflow, workflowToFlow } from './workflowGraph';

const sampling: SamplingSpec = {
  mode: 'arpo',
  group_n: 4,
  sites: [{
    id: 'after-planner',
    anchor: { kind: 'after_agent_turn', agent_id: 'planner' },
    gate: { type: 'entropy_delta', params: { threshold: 0.2 } },
  }],
};

describe('workflowGraph schema 0.3', () => {
  it('defaults an empty canvas to planner, not hub', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'centralized',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: ['web_search'],
      agents: [],
      edges: [],
      sampling,
    };
    const graph = workflowToFlow(workflow);
    expect(graph.nodes.some((node) => node.id === 'hub')).toBe(false);
    expect(graph.nodes.some((node) => node.id === 'executor')).toBe(false);
    expect(graph.nodes.find((node) => node.data.kind === 'planner')?.id).toBe('planner');
    expect(graph.nodes.some((node) => node.id === 'web_search')).toBe(false);
  });

  it('keeps sampling when the graph is serialized', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'centralized',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: ['web_search'],
      agents: [{
        id: 'planner',
        kind: 'planner',
        role: 'planner',
        skills: [],
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
    expect(serialized.entry_agent).toBe('planner');
    expect(serialized.tools).toContain('web_search');
    expect(serialized.sampling).toEqual(sampling);
    expect(serialized.extension).toEqual({ owner: 'research' });
    expect(serialized.agents?.find((agent) => agent.id === 'planner')?.extension).toEqual({ keep: true });
  });

  it('round-trips complete agent and router fields', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: ['python_coder'],
      agents: [
        {
          id: 'planner',
          kind: 'planner',
          role: 'research_planner',
          skills: ['planning'],
          tools: ['python_coder'],
          memory_scope: 'shared',
          system_prompt: 'Plan carefully',
          model: 'inherit',
          trainable: true,
          profile: { temperature: 0.2 },
          meta: { owner: 'team-a' },
        },
        {
          id: 'python_coder',
          kind: 'tool',
          role: 'tool',
          trainable: false,
          profile: { backend: 'llm' },
          meta: { capability: 'python' },
        },
      ],
      routers: [{
        id: 'expert_router',
        candidates: ['python_coder', 'planner'],
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
    const pool = graph.nodes.find((node) => node.type === 'pool');
    expect(router?.data.candidates).toEqual(['python_coder', 'planner']);
    expect(pool?.id).toBe('pool_expert_router');
    expect(pool?.data.members).toEqual(['python_coder', 'planner']);
    expect(graph.nodes.some((node) => node.id === 'python_coder')).toBe(false);
    expect(graph.edges.some((edge) => edge.source === 'expert_router' && edge.target === 'pool_expert_router' && edge.data?.kind === 'route')).toBe(true);

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
    expect(serialized.agents?.find((agent) => agent.id === 'python_coder')).toMatchObject({
      kind: 'tool',
      trainable: false,
      profile: { backend: 'llm' },
      meta: { capability: 'python' },
    });
    expect(serialized.routers).toEqual([{
      id: 'expert_router',
      candidates: ['python_coder', 'planner'],
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

  it('draws a pool under each router and keeps tool-agents inside it', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: [],
      agents: [
        { id: 'planner', kind: 'planner' },
        { id: 'python_agent', kind: 'tool', trainable: true },
        { id: 'verifier', kind: 'verifier' },
        { id: 'hub', kind: 'planner', role: 'hub', system_prompt: 'legacy' },
        { id: 'executor', role: 'executor', tools: ['python_agent'] },
      ],
      routers: [{ id: 'router', candidates: ['python_agent'] }],
      edges: [
        { from: 'planner', to: 'router', kind: 'route' },
        { from: 'router', to: 'verifier', kind: 'message' },
      ],
    };
    const graph = workflowToFlow(workflow);
    const rules = createEdgeRules(graph.nodes, graph.edges, {
      edge_kinds: ['message', 'tool_call', 'feedback', 'route', 'sample_barrier'],
    });

    expect(graph.nodes.some((node) => node.id === 'hub' || node.id === 'executor' || node.id === 'python_agent')).toBe(false);
    expect(graph.nodes.find((node) => node.type === 'pool')?.data.members).toEqual(expect.arrayContaining(['python_agent']));
    expect(graph.edges.map((edge) => [edge.source, edge.target, edge.data?.kind])).toEqual(expect.arrayContaining([
      ['planner', 'router', 'route'],
      ['router', 'pool_router', 'route'],
      ['pool_router', 'verifier', 'message'],
    ]));
    expect(rules.options({
      source: 'planner', target: 'router', sourceHandle: null, targetHandle: null,
    })[0]?.kind).toBe('route');
    expect(rules.error({
      source: 'router', target: 'verifier', sourceHandle: null, targetHandle: null,
    }, 'route')).toContain('tool-agent pool');
    const serialized = flowToWorkflow(graph.nodes, graph.edges, workflow);
    expect(serialized.agents?.some((agent) => agent.id === 'hub' || agent.id === 'executor')).toBe(false);
    expect(serialized.agents?.find((agent) => agent.id === 'python_agent')?.trainable).toBe(false);
    expect(serialized.edges?.some((edge) => edge.from === 'router' && edge.to === 'verifier' && edge.kind === 'message')).toBe(true);
    expect(serialized.edges?.some((edge) => edge.from.startsWith('pool_') || edge.to.startsWith('pool_'))).toBe(false);
  });

  it('folds two fan-out pools back onto their routers', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'centralized',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: ['python_coder', 'think'],
      agents: [
        { id: 'planner', kind: 'planner', trainable: true },
        { id: 'python_coder', kind: 'tool', trainable: true },
        { id: 'think', kind: 'tool', trainable: true },
        { id: 'verifier', kind: 'verifier', trainable: true },
      ],
      routers: [
        { id: 'route_a', candidates: ['python_coder'], strategy: 'from_plan' },
        { id: 'route_b', candidates: ['think'], strategy: 'from_plan' },
      ],
      edges: [
        { from: 'planner', to: 'route_a', kind: 'route' },
        { from: 'planner', to: 'route_b', kind: 'route' },
        { from: 'route_a', to: 'verifier', kind: 'message' },
        { from: 'route_b', to: 'verifier', kind: 'message' },
        { from: 'verifier', to: 'planner', kind: 'feedback' },
      ],
    };
    const graph = workflowToFlow(workflow);
    expect(graph.nodes.find((node) => node.id === 'pool_route_a')?.data.members).toEqual(['python_coder']);
    expect(graph.nodes.find((node) => node.id === 'pool_route_b')?.data.members).toEqual(['think']);
    expect(graph.nodes.some((node) => node.id === 'python_coder' || node.id === 'think')).toBe(false);
    const serialized = flowToWorkflow(graph.nodes, graph.edges, workflow);
    expect(serialized.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ from: 'route_a', to: 'verifier', kind: 'message' }),
      expect.objectContaining({ from: 'route_b', to: 'verifier', kind: 'message' }),
    ]));
    expect(serialized.edges?.some((edge) => edge.from.startsWith('pool_') || edge.to.startsWith('pool_'))).toBe(false);
    expect(serialized.agents?.find((agent) => agent.id === 'python_coder')?.trainable).toBe(false);
    expect(serialized.agents?.find((agent) => agent.id === 'think')?.trainable).toBe(false);
  });

  it('reports missing router candidates without changing sampling', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'graph',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: [],
      agents: [{ id: 'planner', kind: 'planner' }],
      routers: [{ id: 'router', candidates: ['missing'], strategy: 'from_plan' }],
      edges: [{ from: 'planner', to: 'router', kind: 'route' }],
      sampling,
    };

    expect(executableInfo(workflow)).toEqual({
      ok: false,
      reason: 'Router router 的候选 missing 不存在。',
      nodeId: 'router',
    });
    expect(workflow.sampling).toEqual(sampling);
  });

  it('stores a pool member tier on the tool agent profile', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'centralized',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [] },
      tools: ['wikipedia_search'],
      agents: [
        { id: 'planner', kind: 'planner', tools: ['wikipedia_search'] },
        { id: 'wikipedia_search', kind: 'tool', trainable: false, profile: { backend: 'llm' } },
      ],
      routers: [{ id: 'route_exec', candidates: ['wikipedia_search'], strategy: 'from_plan' }],
      edges: [
        { from: 'planner', to: 'route_exec', kind: 'route' },
        { from: 'route_exec', to: 'verifier', kind: 'message' },
      ],
    };
    const graph = workflowToFlow(workflow);
    const pool = graph.nodes.find((node) => node.id === 'pool_route_exec');
    expect(pool?.data.memberTiers?.wikipedia_search).toBeUndefined();
    const edited = graph.nodes.map((node) => node.id === pool?.id
      ? { ...node, data: { ...node.data, memberTiers: { wikipedia_search: 'pro' as const } } }
      : node);
    const serialized = flowToWorkflow(edited, graph.edges, workflow);
    expect(serialized.agents?.find((agent) => agent.id === 'wikipedia_search')?.profile).toMatchObject({
      backend: 'llm',
      tier: 'pro',
    });
  });

  it('shows the default prompt when a saved planner or verifier prompt is empty', () => {
    const workflow: WorkflowSpec = {
      schema_version: '0.3',
      topology: 'centralized',
      entry_agent: 'planner',
      hub: { role: 'planner', skills: [], system_prompt: '' },
      tools: [],
      agents: [
        { id: 'planner', kind: 'planner', system_prompt: '' },
        { id: 'verifier', kind: 'verifier', system_prompt: '  ' },
        { id: 'base_generator', kind: 'tool', system_prompt: 'keep tool' },
      ],
      edges: [],
    };
    const graph = workflowToFlow(workflow);
    expect(graph.nodes.find((node) => node.data.kind === 'planner')?.data.system_prompt).toBe(DEFAULT_PLANNER_PROMPT);
    expect(graph.nodes.find((node) => node.data.kind === 'verifier')?.data.system_prompt).toBe(DEFAULT_VERIFIER_PROMPT);
  });
});

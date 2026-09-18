import { describe, expect, it } from 'vitest';
import { buildTrajectoryGraph, orderAgents, candidateForTrajNode } from './trajectoryGraph';
import type { WorkflowSpec } from '../graph/workflowGraph';

const hubWf: WorkflowSpec = {
  topology: 'hub_react',
  entry_agent: 'hub',
  hub: { role: 'orchestrator', skills: ['react_loop'] },
  tools: ['execute_python', 'web_search'],
  agents: [
    {
      id: 'hub',
      role: 'orchestrator',
      tools: ['execute_python', 'web_search'],
      trainable: true,
    },
  ],
  edges: [],
};

const pevWf: WorkflowSpec = {
  topology: 'graph',
  entry_agent: 'planner',
  hub: { role: 'orchestrator', verify: 'verifier' },
  tools: ['execute_python'],
  agents: [
    { id: 'planner', role: 'planner', tools: [], trainable: true },
    { id: 'executor', role: 'executor', tools: ['execute_python'], trainable: true },
    { id: 'verifier', role: 'verifier', tools: [], trainable: false },
  ],
  edges: [
    { from: 'planner', to: 'executor', kind: 'route' },
    { from: 'executor', to: 'verifier', kind: 'message' },
    { from: 'verifier', to: 'planner', kind: 'feedback' },
  ],
};

describe('orderAgents', () => {
  it('orders PEV by route/message from entry', () => {
    const ids = orderAgents(pevWf).map((a) => a.id);
    expect(ids[0]).toBe('planner');
    expect(ids.indexOf('executor')).toBeGreaterThan(ids.indexOf('planner'));
    expect(ids).toContain('verifier');
  });

  it('falls back to hub for hub_react', () => {
    expect(orderAgents(hubWf).map((a) => a.id)).toEqual(['hub']);
  });
});

describe('buildTrajectoryGraph', () => {
  it('hub path: start → hub → end with tool children', () => {
    const g = buildTrajectoryGraph(hubWf);
    const kinds = g.nodes.map((n) => n.kind);
    expect(kinds[0]).toBe('start');
    expect(kinds[kinds.length - 1]).toBe('end');
    expect(g.nodes.some((n) => n.kind === 'agent' && n.agentId === 'hub')).toBe(true);
    const tools = g.nodes.filter((n) => n.kind === 'tool');
    expect(tools.map((t) => t.toolId).sort()).toEqual(['execute_python', 'web_search']);
  });

  it('maps agent to after_agent_turn candidate', () => {
    const g = buildTrajectoryGraph(hubWf);
    const agent = g.nodes.find((n) => n.kind === 'agent')!;
    expect(agent.candidate?.anchorKind).toBe('after_agent_turn');
    expect(agent.candidate?.agentId).toBe('hub');
    expect(candidateForTrajNode(agent)?.recommendedGate).toBe('entropy_delta');
  });

  it('maps tool to after_tool candidate', () => {
    const g = buildTrajectoryGraph(hubWf);
    const tool = g.nodes.find((n) => n.toolId === 'execute_python')!;
    expect(tool.candidate?.anchorKind).toBe('after_tool');
    expect(tool.candidate?.toolId).toBe('execute_python');
  });

  it('PEV includes verify with verifier_fail gate', () => {
    const g = buildTrajectoryGraph(pevWf);
    const labels = g.nodes.filter((n) => n.kind === 'agent' || n.kind === 'verify').map((n) => n.label);
    expect(labels).toContain('planner');
    expect(labels).toContain('executor');
    const ver = g.nodes.find((n) => n.kind === 'verify');
    expect(ver).toBeTruthy();
    expect(ver!.candidate?.anchorKind).toBe('after_verifier');
    expect(ver!.candidate?.recommendedGate).toBe('verifier_fail');
  });
});

describe('router nodes (agent-framework A5)', () => {
  const routerWf: WorkflowSpec = {
    ...hubWf,
    topology: 'graph',
    entry_agent: 'hub',
    routers: [
      { id: 'route_main', candidates: ['execute_python', 'expert_phys'], strategy: 'llm_choice' },
    ],
  };

  it('renders router as branchable main-path node', () => {
    const g = buildTrajectoryGraph(routerWf);
    const r = g.nodes.find((n) => n.kind === 'router');
    expect(r).toBeTruthy();
    expect(r!.agentId).toBe('route_main');
    expect(r!.branchable).toBe(true);
    expect(g.mainPath).toContain(r!.id);
  });

  it('router candidate → after_agent_turn anchored on router id', () => {
    const g = buildTrajectoryGraph(routerWf);
    const r = g.nodes.find((n) => n.kind === 'router')!;
    expect(r.candidate?.anchorKind).toBe('after_agent_turn');
    expect(r.candidate?.agentId).toBe('route_main');
  });

  it('router tool candidates appear as tool children', () => {
    const g = buildTrajectoryGraph(routerWf);
    const tools = g.nodes.filter((n) => n.kind === 'tool' && n.parentAgentId === 'route_main');
    expect(tools.map((t) => t.toolId)).toContain('execute_python');
  });
});

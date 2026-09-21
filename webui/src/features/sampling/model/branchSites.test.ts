import { describe, expect, it } from 'vitest';
import type { WorkflowSpec } from '../../../shared/api/types';
import {
  deriveBranchCandidates,
  effectiveSites,
  findCandidateSite,
  updateCandidateSite,
} from './branchSites';

const workflow: WorkflowSpec = {
  schema_version: '0.3',
  topology: 'graph',
  entry_agent: 'planner',
  hub: { role: 'orchestrator', skills: ['react_loop'] },
  tools: ['web_search'],
  agents: [
    { id: 'planner', kind: 'planner', tools: ['web_search'] },
    { id: 'python_coder', kind: 'tool' },
    { id: 'verifier', kind: 'verifier' },
  ],
  routers: [{ id: 'router', candidates: ['python_coder'] }],
  edges: [],
};

describe('branch site model', () => {
  it('derives user-facing observation points from workflow entities', () => {
    const candidates = deriveBranchCandidates(workflow);
    expect(candidates.map((candidate) => candidate.label)).toEqual(expect.arrayContaining([
      'After planner turn',
      'After web_search',
      'After python_coder',
      'After verifier verification',
      'After router routing',
    ]));
  });

  it('materializes and updates one observation point without replacing policy fields', () => {
    const candidates = deriveBranchCandidates(workflow);
    const candidate = candidates.find((item) => item.nodeId === 'planner')!;
    const sampling = {
      mode: 'arpo',
      group_n: 4,
      beam_size: 2,
      initial_rollouts: 2,
      barriers: [],
      sites: [],
      extra: { keep: true },
    };
    const updated = updateCandidateSite(sampling, candidates, candidate, {
      gate: { type: 'always', params: { keep: true } },
    });
    const site = findCandidateSite(effectiveSites(updated, candidates), candidate);

    expect(updated.extra).toEqual({ keep: true });
    expect(site?.gate).toEqual({ type: 'always', params: { keep: true } });
    expect(site?.fork).toMatchObject({ beam_size: 2, resume_mode: 'messages' });
  });
});

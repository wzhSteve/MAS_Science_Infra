import type {
  AgentSpec,
  BranchAnchorKind,
  BranchSiteSpec,
  SamplingSpec,
  WorkflowSpec,
} from '../../../shared/api/types';

export type BranchCandidateKind = 'agent' | 'tool' | 'verifier' | 'router' | 'edge';

export interface BranchCandidate {
  id: string;
  kind: BranchCandidateKind;
  label: string;
  nodeId: string;
  anchor: BranchSiteSpec['anchor'];
  recommendedGate: string;
}

function agentsOf(workflow: WorkflowSpec): AgentSpec[] {
  return workflow.agents?.length
    ? workflow.agents
    : [{
        id: 'hub',
        kind: 'hub',
        role: workflow.hub?.role || 'orchestrator',
        tools: workflow.tools || [],
      }];
}

function candidateId(anchor: BranchSiteSpec['anchor']): string {
  return [
    anchor.kind,
    anchor.agent_id || '',
    anchor.tool_id || '',
    anchor.edge_id || '',
  ].join(':');
}

export function deriveBranchCandidates(workflow: WorkflowSpec): BranchCandidate[] {
  const candidates: BranchCandidate[] = [];
  const seen = new Set<string>();
  const add = (candidate: Omit<BranchCandidate, 'id'>) => {
    const id = candidateId(candidate.anchor);
    if (!seen.has(id)) {
      seen.add(id);
      candidates.push({ ...candidate, id });
    }
  };

  for (const agent of agentsOf(workflow)) {
    if (agent.kind === 'tool') {
      add({
        kind: 'tool',
        label: `After ${agent.id}`,
        nodeId: agent.id,
        anchor: { kind: 'after_tool', agent_id: agent.id, tool_id: agent.id },
        recommendedGate: 'entropy_delta',
      });
      continue;
    }
    if (agent.kind === 'verifier' || agent.id === 'verifier' || agent.role === 'verifier') {
      add({
        kind: 'verifier',
        label: `After ${agent.id} verification`,
        nodeId: agent.id,
        anchor: { kind: 'after_verifier', agent_id: agent.id },
        recommendedGate: 'verifier_fail',
      });
    } else {
      add({
        kind: 'agent',
        label: `After ${agent.id} turn`,
        nodeId: agent.id,
        anchor: { kind: 'after_agent_turn', agent_id: agent.id },
        recommendedGate: 'entropy_delta',
      });
    }
    for (const tool of agent.tools || (agent.id === 'hub' ? workflow.tools : [])) {
      add({
        kind: 'tool',
        label: `After ${tool}`,
        nodeId: tool,
        anchor: { kind: 'after_tool', agent_id: agent.id, tool_id: tool },
        recommendedGate: 'entropy_delta',
      });
    }
  }

  for (const router of workflow.routers || []) {
    add({
      kind: 'router',
      label: `After ${router.id} routing`,
      nodeId: router.id,
      anchor: { kind: 'after_agent_turn', agent_id: router.id },
      recommendedGate: 'entropy_delta',
    });
  }

  (workflow.edges || []).forEach((edge, index) => {
    if (edge.kind !== 'sample_barrier') return;
    const edgeId = `e-${edge.from}-${edge.to}-${index}`;
    add({
      kind: 'edge',
      label: `On ${edge.from} → ${edge.to}`,
      nodeId: edge.to,
      anchor: { kind: 'on_edge', agent_id: edge.from, edge_id: edgeId },
      recommendedGate: 'always',
    });
  });
  return candidates;
}

function sameAnchor(site: BranchSiteSpec, candidate: BranchCandidate): boolean {
  const anchor = site.anchor;
  return anchor.kind === candidate.anchor.kind
    && (anchor.agent_id || null) === (candidate.anchor.agent_id || null)
    && (anchor.tool_id || null) === (candidate.anchor.tool_id || null)
    && (anchor.edge_id || null) === (candidate.anchor.edge_id || null);
}

export function defaultSite(candidate: BranchCandidate): BranchSiteSpec {
  return {
    id: `site_${candidate.id.replace(/[^A-Za-z0-9_-]/g, '_')}`,
    enabled: true,
    anchor: { ...candidate.anchor },
    when: 'first',
    nth: 1,
    gate: { type: candidate.recommendedGate, params: {} },
    fork: { beam_size: 2, share_observation: true, resume_mode: 'messages', probe_max_tokens: 128 },
    reward: { scheme: 'scalar_grpo', p_plus: 0.8, k_min: 2, epsilon_f: 0.001, dead_end_backprop: 1 },
    priority: 10,
  };
}

export function effectiveSites(sampling: SamplingSpec, _candidates?: BranchCandidate[]): BranchSiteSpec[] {
  return sampling.sites || [];
}

export function legacySiteCount(sampling: SamplingSpec): number {
  return sampling.sites?.length ? 0 : (sampling.barriers ?? ['after_tool']).length;
}

export function findCandidateSite(sites: BranchSiteSpec[], candidate: BranchCandidate): BranchSiteSpec | undefined {
  return sites.find((site) => sameAnchor(site, candidate));
}

export function updateCandidateSite(
  sampling: SamplingSpec,
  candidates: BranchCandidate[],
  candidate: BranchCandidate,
  patch: Partial<BranchSiteSpec>,
): SamplingSpec {
  const sites = effectiveSites(sampling);
  const existing = findCandidateSite(sites, candidate);
  const defaults = defaultSite(candidate);
  const base = existing || defaults;
  const next = {
    ...base,
    ...patch,
    anchor: { ...base.anchor, ...(patch.anchor || {}) },
    gate: { ...base.gate, ...(patch.gate || {}) },
    fork: { ...base.fork, ...(patch.fork || {}) },
    reward: { ...base.reward, ...(patch.reward || {}) },
  };
  return {
    ...sampling,
    barriers: [],
    sites: [...sites.filter((site) => !sameAnchor(site, candidate)), next],
  };
}

export function candidateTypeLabel(kind: BranchCandidateKind): string {
  return {
    agent: 'Agent',
    tool: 'Tool',
    verifier: 'Verifier',
    router: 'Router',
    edge: 'Edge',
  }[kind];
}

export const BRANCH_GATE_OPTIONS = [
  { value: 'entropy_delta', label: 'Entropy ΔH', help: '根据 ΔH = H_current - H_root 判断是否 Branch。' },
  { value: 'dual_entropy', label: 'Dual Entropy', help: '根据 U = H_π × H_B 判断是否 Branch。' },
  { value: 'always', label: 'Always', help: '到达 Branch Site 时直接 Branch。' },
  { value: 'tool_ok', label: 'Tool OK', help: 'Tool 正常返回时 Branch。' },
  { value: 'tool_error', label: 'Tool Error', help: 'Tool 调用失败时 Branch。' },
  { value: 'verifier_pass', label: 'Verifier Pass', help: 'Verifier 判定通过时 Branch。' },
  { value: 'verifier_fail', label: 'Verifier Fail', help: 'Verifier 判定失败时 Branch。' },
  { value: 'contradiction', label: 'Contradiction', help: '检测到结果矛盾时 Branch。' },
  { value: 'failure_trigger', label: 'Failure Trigger', help: '当前执行失败时创建补救 Branch。' },
] as const;

export const SAMPLING_MODE_OPTIONS = [
  { value: 'grpo_n', label: 'GRPO', help: '每道题生成固定数量的独立 Rollout。' },
  { value: 'arpo', label: 'ARPO', help: '根据 Entropy ΔH 在 Branch Site 扩展后续路径。' },
  { value: 'aepo', label: 'AEPO', help: '使用 Entropy 与 Information Gain 分配 Branch Budget。' },
  { value: 'appo', label: 'APPO', help: '自适应路径采样；当前训练实现映射到 ARPO。' },
  { value: 'rae', label: 'RAE', help: '结合 Verifier Verdict 与 Dead-end Backprop。' },
] as const;

export function gateOptionsFor(mode: string | undefined, kind: BranchCandidateKind): string[] {
  const options = mode === 'rae'
    ? ['dual_entropy', 'verifier_fail', 'contradiction', 'failure_trigger', 'always']
    : mode === 'aepo' || mode === 'appo'
      ? ['dual_entropy', 'entropy_delta', 'always']
      : ['entropy_delta', 'always'];
  if (kind === 'tool') options.push('tool_ok', 'tool_error');
  if (kind === 'verifier') options.push('verifier_pass', 'verifier_fail', 'contradiction');
  return Array.from(new Set(options));
}

export function isBranchingMode(mode?: string): boolean {
  return !['grpo', 'grpo_n'].includes(mode || 'grpo_n');
}

export function anchorKind(site: BranchSiteSpec): BranchAnchorKind {
  return site.anchor.kind;
}

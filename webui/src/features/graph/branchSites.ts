/** Derive branch-site candidates from a WorkflowSpec for MAS UI overlay. */

import type { WorkflowSpec } from './workflowGraph';

export type BranchCandidate = {
  id: string;
  label: string;
  anchorKind: string;
  agentId?: string;
  toolId?: string;
  edgeId?: string;
  recommendedGate: string;
  nodeId: string; // graph node to badge
};

export type BranchSiteConfig = {
  id: string;
  enabled: boolean;
  anchor: {
    kind: string;
    agent_id?: string | null;
    tool_id?: string | null;
    edge_id?: string | null;
  };
  when?: string;
  nth?: number;
  gate: { type: string; params?: Record<string, unknown> };
  fork?: {
    beam_size?: number | null;
    share_observation?: boolean;
    resume_mode?: string;
    probe_max_tokens?: number;
  };
  reward?: {
    scheme?: string;
    p_plus?: number;
    k_min?: number;
    epsilon_f?: number;
  };
  priority?: number;
};

export const GATE_TYPES = [
  'entropy_delta',
  'dual_entropy',
  'always',
  'tool_ok',
  'tool_error',
  'verifier_pass',
  'verifier_fail',
  'contradiction',
  'failure_trigger',
] as const;

export function deriveBranchCandidates(wf: WorkflowSpec): BranchCandidate[] {
  const out: BranchCandidate[] = [];
  const agents = wf.agents?.length
    ? wf.agents
    : [
        {
          id: 'hub',
          tools: wf.tools || [],
          role: wf.hub?.role || 'orchestrator',
        },
      ];

  for (const a of agents) {
    const tools = a.tools?.length ? a.tools : a.id === 'hub' ? wf.tools || [] : [];
    // W3: kind=tool agents are tool-call targets too — derive after_tool
    // candidates from the agent kind, not the legacy node type. Blank router
    // candidates ride as blank:<id> (see workflowGraph.flowToWorkflow).
    const isToolAgent = a.kind === 'tool';
    for (const t of tools) {
      out.push({
        id: `cand_after_tool_${a.id}_${t}`,
        label: `after ${t}`,
        anchorKind: 'after_tool',
        agentId: a.id,
        toolId: t,
        recommendedGate: 'entropy_delta',
        nodeId: t,
      });
    }
    // W3: kind=tool agents are tool-call targets too — derive after_tool
    // candidates from the agent kind, not the legacy node type.
    if (isToolAgent) {
      out.push({
        id: `cand_after_tool_${a.id}`,
        label: `after ${a.id} (tool-agent)`,
        anchorKind: 'after_tool',
        agentId: a.id,
        toolId: a.id,
        recommendedGate: 'entropy_delta',
        nodeId: a.id,
      });
    }
    out.push({
      id: `cand_after_turn_${a.id}`,
      label: `after ${a.id} turn`,
      anchorKind: 'after_agent_turn',
      agentId: a.id,
      recommendedGate: 'entropy_delta',
      nodeId: a.id,
    });
  }

  const hasVerifier =
    !!wf.hub?.verify || (wf.agents || []).some((a) => a.id === 'verifier' || a.role === 'verifier');
  if (hasVerifier) {
    out.push({
      id: 'cand_after_verifier',
      label: 'after verifier',
      anchorKind: 'after_verifier',
      agentId: 'verifier',
      recommendedGate: 'verifier_fail',
      nodeId: 'verifier',
    });
  }

  (wf.edges || []).forEach((e, i) => {
    if ((e.kind || '') !== 'sample_barrier') return;
    const edgeId = `e-${e.from}-${e.to}-${i}`;
    out.push({
      id: `cand_edge_${edgeId}`,
      label: `barrier ${e.from}→${e.to}`,
      anchorKind: 'on_edge',
      edgeId,
      agentId: e.from,
      recommendedGate: 'always',
      nodeId: e.to,
    });
  });

  // Advanced: on_token candidates (P2) — one per trainable agent
  for (const a of agents) {
    if (a.trainable === false) continue;
    out.push({
      id: `cand_on_token_${a.id}`,
      label: `on_token @${a.id}`,
      anchorKind: 'on_token',
      agentId: a.id,
      recommendedGate: 'entropy_delta',
      nodeId: a.id,
    });
  }

  return out;
}

export function siteKey(site: BranchSiteConfig): string {
  const a = site.anchor || { kind: 'after_tool' };
  return [a.kind, a.agent_id || '', a.tool_id || '', a.edge_id || ''].join('|');
}

export function findSiteForCandidate(
  sites: BranchSiteConfig[] | undefined,
  cand: BranchCandidate,
): BranchSiteConfig | undefined {
  const list = sites || [];
  return list.find((s) => {
    const a = s.anchor || { kind: '' };
    if (a.kind !== cand.anchorKind) return false;
    if (cand.toolId && a.tool_id !== cand.toolId) return false;
    if (cand.agentId && a.agent_id && a.agent_id !== cand.agentId) return false;
    if (cand.edgeId && a.edge_id !== cand.edgeId) return false;
    return true;
  });
}

export function upsertSiteFromCandidate(
  sites: BranchSiteConfig[],
  cand: BranchCandidate,
  patch: Partial<BranchSiteConfig>,
): BranchSiteConfig[] {
  const existing = findSiteForCandidate(sites, cand);
  const base: BranchSiteConfig = existing || {
    id: cand.id.replace(/^cand_/, 'site_'),
    enabled: true,
    anchor: {
      kind: cand.anchorKind,
      agent_id: cand.agentId || null,
      tool_id: cand.toolId || null,
      edge_id: cand.edgeId || null,
    },
    when: 'first',
    gate: { type: cand.recommendedGate, params: {} },
    fork: { beam_size: 2, share_observation: true, resume_mode: 'messages' },
    reward: { scheme: 'scalar_grpo' },
    priority: 10,
  };
  const next = { ...base, ...patch, anchor: { ...base.anchor, ...(patch.anchor || {}) } };
  const others = sites.filter((s) => s.id !== next.id && siteKey(s) !== siteKey(next));
  return [...others, next];
}

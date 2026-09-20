/** Derive an ordered episode trajectory from WorkflowSpec for Rollout Sampling panel. */

import type { WorkflowSpec, AgentSpec } from '../graph/workflowGraph';
import type { BranchCandidate } from '../graph/branchSites';

export type TrajNodeKind = 'start' | 'agent' | 'tool' | 'verify' | 'router' | 'end';

export type TrajNode = {
  id: string;
  kind: TrajNodeKind;
  label: string;
  agentId?: string;
  toolId?: string;
  /** Can host a branch socket */
  branchable: boolean;
  /** Linked BranchCandidate when branchable */
  candidate?: BranchCandidate;
  /** Parent agent id for tool nodes */
  parentAgentId?: string;
};

export type TrajEdge = {
  id: string;
  from: string;
  to: string;
  dashed?: boolean;
};

export type TrajectoryGraph = {
  nodes: TrajNode[];
  edges: TrajEdge[];
  /** Main path agent/verify ids in order (no start/end/tools) */
  mainPath: string[];
};

function agentsOf(wf: WorkflowSpec): AgentSpec[] {
  if (wf.agents && wf.agents.length > 0) return wf.agents;
  return [
    {
      id: 'hub',
      role: wf.hub?.role || 'orchestrator',
      skills: wf.hub?.skills || ['react_loop'],
      tools: wf.tools || [],
      trainable: true,
    },
  ];
}

function toolsFor(wf: WorkflowSpec, a: AgentSpec): string[] {
  if (a.tools && a.tools.length) return a.tools;
  if (a.id === 'hub') return wf.tools || [];
  return [];
}

/** BFS along route/message edges from entry; fallback to agents array order. */
export function orderAgents(wf: WorkflowSpec): AgentSpec[] {
  const agents = agentsOf(wf);
  const byId = new Map(agents.map((a) => [a.id, a]));
  const entry = wf.entry_agent || agents[0]?.id || 'hub';

  const adj = new Map<string, string[]>();
  for (const e of wf.edges || []) {
    const kind = String(e.kind || 'message');
    if (kind !== 'route' && kind !== 'message') continue;
    if (!byId.has(e.from) && e.from !== entry) continue;
    const list = adj.get(e.from) || [];
    if (!list.includes(e.to)) list.push(e.to);
    adj.set(e.from, list);
  }

  const ordered: AgentSpec[] = [];
  const seen = new Set<string>();
  const q: string[] = [entry];
  while (q.length) {
    const id = q.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const a = byId.get(id);
    if (a) ordered.push(a);
    for (const nxt of adj.get(id) || []) {
      if (!seen.has(nxt) && byId.has(nxt)) q.push(nxt);
    }
  }

  // Append any agents not reached (keep declaration order)
  for (const a of agents) {
    if (!seen.has(a.id)) ordered.push(a);
  }

  // Ensure verifier appears if hub.verify points to it but missing from agents walk
  const verifyId = wf.hub?.verify || null;
  if (verifyId && !ordered.some((a) => a.id === verifyId)) {
    const v =
      byId.get(verifyId) ||
      ({ id: verifyId, role: 'verifier', skills: ['verifier'], trainable: false } as AgentSpec);
    ordered.push(v);
  }

  return ordered;
}

export function candidateForTrajNode(n: TrajNode): BranchCandidate | undefined {
  if (!n.branchable) return undefined;
  if (n.kind === 'agent' && n.agentId) {
    return {
      id: `cand_after_turn_${n.agentId}`,
      label: `after ${n.agentId} turn`,
      anchorKind: 'after_agent_turn',
      agentId: n.agentId,
      recommendedGate: 'entropy_delta',
      nodeId: n.agentId,
    };
  }
  if (n.kind === 'tool' && n.agentId && n.toolId) {
    return {
      id: `cand_after_tool_${n.agentId}_${n.toolId}`,
      label: `after ${n.toolId}`,
      anchorKind: 'after_tool',
      agentId: n.agentId,
      toolId: n.toolId,
      recommendedGate: 'entropy_delta',
      nodeId: n.toolId,
    };
  }
  if (n.kind === 'verify') {
    const vid = n.agentId || 'verifier';
    return {
      id: 'cand_after_verifier',
      label: `after ${vid}`,
      anchorKind: 'after_verifier',
      agentId: vid,
      recommendedGate: 'verifier_fail',
      nodeId: vid,
    };
  }
  if (n.kind === 'router' && n.agentId) {
    // agent-framework A5: router boundary — anchor on the router node id.
    return {
      id: `cand_after_turn_${n.agentId}`,
      label: `after ${n.agentId} (router)`,
      anchorKind: 'after_agent_turn',
      agentId: n.agentId,
      recommendedGate: 'entropy_delta',
      nodeId: n.agentId,
    };
  }
  return undefined;
}

/**
 * Build simplified episode trajectory: start → agents (±tools) → verify → end.
 * Tools are included as branchable children (UI may fold them).
 */
export function buildTrajectoryGraph(wf: WorkflowSpec): TrajectoryGraph {
  const ordered = orderAgents(wf);
  const nodes: TrajNode[] = [];
  const edges: TrajEdge[] = [];
  const mainPath: string[] = [];

  nodes.push({ id: 'traj_start', kind: 'start', label: 'start', branchable: false });

  let prev = 'traj_start';
  for (const a of ordered) {
    const isVerifier = a.id === 'verifier' || a.role === 'verifier' || a.id === wf.hub?.verify;
    const agentNodeId = `traj_agent_${a.id}`;

    if (isVerifier && a.id !== ordered[0]?.id) {
      // treat as verify step on main path
      const vn: TrajNode = {
        id: `traj_verify_${a.id}`,
        kind: 'verify',
        label: a.id,
        agentId: a.id,
        branchable: true,
      };
      vn.candidate = candidateForTrajNode(vn);
      nodes.push(vn);
      edges.push({ id: `e_${prev}_${vn.id}`, from: prev, to: vn.id });
      mainPath.push(vn.id);
      prev = vn.id;
      continue;
    }

    const an: TrajNode = {
      id: agentNodeId,
      kind: 'agent',
      label: a.id,
      agentId: a.id,
      branchable: true,
    };
    an.candidate = candidateForTrajNode(an);
    nodes.push(an);
    edges.push({ id: `e_${prev}_${an.id}`, from: prev, to: an.id });
    mainPath.push(an.id);
    prev = an.id;

    for (const t of toolsFor(wf, a)) {
      const tn: TrajNode = {
        id: `traj_tool_${a.id}_${t}`,
        kind: 'tool',
        label: t,
        agentId: a.id,
        toolId: t,
        parentAgentId: a.id,
        branchable: true,
      };
      tn.candidate = candidateForTrajNode(tn);
      nodes.push(tn);
      edges.push({ id: `e_${an.id}_${tn.id}`, from: an.id, to: tn.id, dashed: true });
    }
  }

  // agent-framework A5: RouterSpec boundaries are branchable candidates on
  // the main path (routing boundary can host a gate, design §2.2).
  const routerNodes = wf.routers || [];
  for (const r of routerNodes) {
    const rn: TrajNode = {
      id: `traj_router_${r.id}`,
      kind: 'router',
      label: r.id,
      agentId: r.id,
      branchable: true,
    };
    rn.candidate = candidateForTrajNode(rn);
    nodes.push(rn);
    edges.push({ id: `e_${prev}_${rn.id}`, from: prev, to: rn.id });
    mainPath.push(rn.id);
    prev = rn.id;

    // tool-agent and blank-agent candidates hang off the router like tool
    // children (W5: blank:<id> labels strip the prefix for display).
    for (const c of r.candidates || []) {
      const isBlank = c.startsWith('blank:');
      const tn: TrajNode = {
        id: `traj_tool_${r.id}_${c}`,
        kind: 'tool',
        label: isBlank ? c.slice('blank:'.length) : c,
        agentId: r.id,
        toolId: c,
        parentAgentId: r.id,
        branchable: true,
      };
      tn.candidate = candidateForTrajNode(tn);
      nodes.push(tn);
      edges.push({ id: `e_${rn.id}_${tn.id}`, from: rn.id, to: tn.id, dashed: true });
    }
  }

  // If hub.verify set but verifier not already a node
  const verifyName = wf.hub?.verify;
  if (verifyName && !nodes.some((n) => n.kind === 'verify')) {
    const vn: TrajNode = {
      id: `traj_verify_${verifyName}`,
      kind: 'verify',
      label: verifyName,
      agentId: verifyName,
      branchable: true,
    };
    vn.candidate = candidateForTrajNode(vn);
    nodes.push(vn);
    edges.push({ id: `e_${prev}_${vn.id}`, from: prev, to: vn.id });
    mainPath.push(vn.id);
    prev = vn.id;
  }

  nodes.push({ id: 'traj_end', kind: 'end', label: 'end', branchable: false });
  edges.push({ id: `e_${prev}_traj_end`, from: prev, to: 'traj_end' });

  return { nodes, edges, mainPath };
}

/** Branchable nodes only (for sockets). */
export function branchableNodes(
  g: TrajectoryGraph,
  opts?: { includeTools?: boolean },
): TrajNode[] {
  const includeTools = !!opts?.includeTools;
  return g.nodes.filter((n) => {
    if (!n.branchable || !n.candidate) return false;
    if (n.kind === 'tool' && !includeTools) return false;
    return true;
  });
}

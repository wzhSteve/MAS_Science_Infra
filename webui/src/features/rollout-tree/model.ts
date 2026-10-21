import type { Outcome, TreeDetail, TreeEdge, TreeNode } from './types';

export const MAX_PAGES = 5;
export const terminalRun = (state?: string) => ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(state || '');
const LIVE_TREE_ID = /^[a-f0-9]{32}$/;

/** Move the open graph onto the tree named by the latest live event. */
export function followedTreeId(
  event: { tree_id?: unknown; run_id?: unknown } | undefined,
  selection: { runId: string; treeId?: string },
  pinned: boolean,
  runState?: string,
): string | null {
  if (pinned || terminalRun(runState)) return null;
  const treeId = typeof event?.tree_id === 'string' ? event.tree_id : '';
  const runId = typeof event?.run_id === 'string' ? event.run_id : '';
  if (!LIVE_TREE_ID.test(treeId) || treeId === selection.treeId) return null;
  if (runId && runId !== selection.runId) return null;
  return treeId;
}
export const answerName = (id: string) => `回答 ${id.length > 14 ? `${id.slice(0, 6)}…${id.slice(-6)}` : id}`;
export const rewardText = (value?: number | null) => value == null ? '—' : String(Number(value.toFixed(4)));
export const resultId = (node: TreeNode) => node.result_node_id || node.node_id;
export function selectableId(node: TreeNode, nodes: TreeNode[] = []) {
  const resultExists = nodes.some(item => item.kind === 'outcome' && item.node_id === node.result_node_id);
  return node.result_node_id && (resultExists || ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(node.store_status || ''))
    ? node.result_node_id : node.node_id;
}
export const executionName = (node: TreeNode) => `${node.agent_id || (node.agent_kind === 'tool' ? '工具' : 'Agent')}${node.turn != null ? ` · 第${node.turn}次执行` : ''}`;
export const edgeId = (edge: TreeEdge) => JSON.stringify([edge.source_node_id, edge.target_node_id, edge.kind, edge.window_id, edge.snapshot_ref, edge.site_id]);
export function candidateNodes(tree: TreeDetail['tree']) {
  return tree.schema_version === 3 ? tree.rollouts || []
    : [...tree.nodes, ...tree.pending_nodes].filter(node => node.kind === 'rollout');
}
export function answerState(node: TreeNode, outcome?: Outcome) {
  if (node.terminal_unconfirmed) return { label: '结果待确认', tone: 'warning' };
  if (node.execution_error === 'timeout') return { label: '执行超时', tone: 'danger' };
  if (node.status === 'failed') return { label: '执行失败', tone: 'danger' };
  if (node.status === 'cancelled') return { label: '已取消', tone: 'neutral' };
  if (node.status === 'running') return { label: '执行中', tone: 'info' };
  if (node.status === 'enqueued') return { label: '等待执行', tone: 'neutral' };
  if (node.status === 'succeeded') {
    if (node.training_status === 'rejected') return { label: '执行已完成 · 训练样本未接纳', tone: 'warning' };
    return { label: outcome?.answer == null ? '执行已结束 · 答案未记录' : '执行已完成', tone: 'neutral' };
  }
  return { label: '状态未记录', tone: 'neutral' };
}
export function answerSource(node: TreeNode) {
  if (node.origin === 'initial') return '独立生成';
  if (node.origin === 'independent_fill') return '独立补采样';
  if (node.origin === 'branch' && node.parent_id) return `从${answerName(node.parent_id)}的中间位置重新生成`;
  return '生成来源未记录';
}
export function mergeDetail(first: TreeDetail, next: TreeDetail): TreeDetail {
  if (first.tree.tree_id !== next.tree.tree_id || first.tree.revision !== next.tree.revision
    || (first.tree.schema_version || 2) !== (next.tree.schema_version || 2)) {
    throw new Error('不能合并不同题目或不同版本的记录。');
  }
  const merge = <T,>(a: T[], b: T[], key: (item: T) => string) => [...new Map([...a, ...b].map(item => [key(item), item])).values()];
  return {
    ...next,
    tree: { ...next.tree,
      nodes: merge(first.tree.nodes, next.tree.nodes, n => n.node_id),
      pending_nodes: merge(first.tree.pending_nodes, next.tree.pending_nodes, n => n.node_id),
      edges: merge(first.tree.edges || [], next.tree.edges || [], edgeId),
      rollouts: merge(first.tree.rollouts || [], next.tree.rollouts || [], n => n.node_id),
      plans: merge(first.tree.plans, next.tree.plans, p => p.plan_id),
      outcomes: { ...first.tree.outcomes, ...next.tree.outcomes },
      issues: [...new Set([...first.tree.issues, ...next.tree.issues])],
    },
  };
}

// Parent-before-child preorder gives compact, deterministic forest layout.
export function forestPositions(nodes: Pick<TreeNode, 'node_id' | 'parent_id' | 'kind'>[]) {
  const actual = nodes.filter(node => node.kind !== 'query');
  const ids = new Set(actual.map(node => node.node_id));
  const children = new Map<string, typeof actual>();
  for (const node of actual) {
    const parent = node.parent_id && ids.has(node.parent_id) ? node.parent_id : '';
    const siblings = children.get(parent) || [];
    siblings.push(node); children.set(parent, siblings);
  }
  const positions = new Map<string, { x: number; y: number }>();
  const stack = [...(children.get('') || [])].reverse().map(node => ({ node, depth: 0 }));
  while (stack.length) {
    const entry = stack.pop()!;
    if (positions.has(entry.node.node_id)) continue;
    positions.set(entry.node.node_id, { x: entry.depth * 310, y: positions.size * 115 });
    for (const child of [...(children.get(entry.node.node_id) || [])].reverse()) stack.push({ node: child, depth: entry.depth + 1 });
  }
  return positions;
}

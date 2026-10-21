import { describe, expect, it, vi, afterEach } from 'vitest';
import { parseRoute, routeHash, workspaceRoute } from '../../app/navigation';
import { createTreeClient } from './api';
import { answerState, candidateNodes, followedTreeId, forestPositions, mergeDetail, rewardText, selectableId } from './model';
import { ancestorPath, executionPositions } from './executionLayout';
import type { TreeDetail, TreeNode } from './types';

export const node = (id: string, extra: Partial<TreeNode> = {}): TreeNode => ({
  node_id: id, parent_id: 'query', kind: 'rollout', origin: 'initial',
  status: 'succeeded', store_status: 'succeeded', training_status: null,
  record_issues: [], attempt_id: 'attempt', attempt_sequence: 1, site_id: null,
  window_id: null, event_id: null, boundary_snapshot_ref: null,
  reward: null, verdict: null, execution_error: null, judgments: [], previous_attempts: [],
  decision: {}, metrics: {}, ...extra,
});
export const detail = (revision = 1, nodes = [node('a')]): TreeDetail => ({
  experiment_id: 'exp', run_id: 'abcdef123456', run_state: 'running',
  tree: { tree_id: 'a'.repeat(32), group_id: 'group', mode: 'train', revision,
    updated_at: null, query: 'Question', query_truncated: false, nodes, pending_nodes: [],
    outcomes: { a: { answer: 'Answer', reward: 0 } }, plans: [], issues: [] },
  page: { total: nodes.length, next_cursor: null, truncated: false, plan_total: 0, next_plan_offset: null },
});

afterEach(() => vi.unstubAllGlobals());
describe('sample result model', () => {
  it('follows the live tree when the open one is already finished', () => {
    const live = 'b'.repeat(32);
    const other = 'a'.repeat(32);
    const event = { tree_id: live, run_id: 'abcdef123456' };
    expect(followedTreeId(event, { runId: 'abcdef123456', treeId: other }, false, 'running')).toBe(live);
    expect(followedTreeId(event, { runId: 'abcdef123456', treeId: other }, true, 'running')).toBeNull();
    expect(followedTreeId(event, { runId: 'abcdef123456', treeId: live }, false, 'running')).toBeNull();
    expect(followedTreeId({ tree_id: 'global' }, { runId: 'abcdef123456', treeId: other }, false, 'running')).toBeNull();
    expect(followedTreeId(event, { runId: 'abcdef123456', treeId: other }, false, 'succeeded')).toBeNull();
  });
  it('round trips fixed run, selected answer and explicit view', () => {
    const route = { ...workspaceRoute('exp', 'records'), samples: {
      runId: 'abcdef123456', treeId: 'a'.repeat(32), nodeId: 'parent:child/1', view: 'branches' as const,
    } };
    expect(parseRoute(routeHash(route))).toEqual(route);
    expect(parseRoute('#/experiments/exp/runs/abcdef123456/samples')).toMatchObject({ samples: { view: 'auto' } });
    expect(parseRoute('#/experiments/exp/runs/invalid/samples').kind).toBe('not-found');
    expect(parseRoute('#/experiments/exp/runs/abcdef123456/samples?answer=x').kind).toBe('not-found');
    expect(parseRoute('#/experiments/exp/runs/abcdef123456/samples?view=unknown').kind).toBe('not-found');
  });
  it('keeps existing console routes unchanged', () => {
    const route = { ...workspaceRoute('exp'), console: 'training' as const, runId: 'abcdef123456' };
    expect(parseRoute(routeHash(route))).toEqual(route);
  });
  it('does not confuse zero, missing reward, failed execution or incomplete records', () => {
    expect(rewardText(0)).toBe('0');
    expect(rewardText(null)).toBe('—');
    expect(answerState(node('a'), { answer: 'A' }).label).toBe('执行已完成');
    expect(answerState(node('a', { terminal_unconfirmed: true })).label).toBe('结果待确认');
    expect(answerState(node('a', { status: 'failed' })).tone).toBe('danger');
    expect(answerState(node('a', { training_status: 'rejected' })).label).toContain('未接纳');
    expect(answerState(node('a')).label).toContain('答案未记录');
  });
  it('merges pages only within the same tree revision and keeps parent outcome', () => {
    const first = detail();
    const second = detail(1, [node('a'), node('child', { parent_id: 'a', origin: 'branch' })]);
    second.tree.outcomes = { child: { reward: 1 } };
    const merged = mergeDetail(first, second);
    expect(merged.tree.nodes).toHaveLength(2);
    expect(merged.tree.outcomes.a.reward).toBe(0);
    expect(merged.tree.outcomes.child.reward).toBe(1);
    expect(() => mergeDetail(first, detail(2))).toThrow();
  });
  it('lays out a forest without introducing a query execution node', () => {
    const nodes = [node('query', { kind: 'query', parent_id: null }), node('a'),
      node('b'), node('child', { parent_id: 'a', origin: 'branch' })];
    const positions = forestPositions(nodes);
    expect(positions.size).toBe(3);
    expect(positions.get('child')!.x).toBeGreaterThan(positions.get('a')!.x);
    expect(positions.get('b')!.x).toBe(positions.get('a')!.x);
    expect(forestPositions(nodes.map(n => ({ ...n, reward: 5 })))).toEqual(positions);
  });
  it('lays out shared prefixes horizontally and highlights only inherited ancestors', () => {
    const nodes = ['q', 'a1', 'tool', 'a2', 'ra', 'b1', 'rb', 'c1', 'rc'].map(id =>
      node(id, { kind: id === 'q' ? 'query' : id.startsWith('r') ? 'outcome' : 'execution' }));
    const edges = [
      ['q', 'a1'], ['a1', 'tool'], ['tool', 'a2'], ['a2', 'ra'],
      ['tool', 'b1'], ['b1', 'rb'], ['q', 'c1'], ['c1', 'rc'],
    ].map(([source_node_id, target_node_id]) => ({
      source_node_id, target_node_id, kind: target_node_id === 'b1' ? 'branch' as const : 'sequence' as const,
    }));
    const positions = executionPositions(nodes, edges);
    expect(positions.size).toBe(nodes.length);
    for (const edge of edges) expect(positions.get(edge.target_node_id)!.x).toBeGreaterThan(positions.get(edge.source_node_id)!.x);
    expect(positions.get('a2')!.y).toBe(positions.get('a1')!.y);
    expect(positions.get('b1')!.y).not.toBe(positions.get('a2')!.y);
    expect(positions.get('c1')!.y).not.toBe(positions.get('a1')!.y);
    expect(ancestorPath('rb', edges).nodes).toEqual(new Set(['rb', 'b1', 'tool', 'a1', 'q']));
    expect(executionPositions(nodes.map(n => ({ ...n, status: 'running', reward: 10 })), edges, positions)).toEqual(positions);
    const appended = executionPositions([...nodes, node('new', { kind: 'execution' })], [
      ...edges, { source_node_id: 'tool', target_node_id: 'new', kind: 'branch' },
    ], positions);
    for (const [id, position] of positions) expect(appended.get(id)).toEqual(position);
  });
  it('uses all parallel edges for topological depth and rejoins the main lane', () => {
    const nodes = ['q', 'a', 'tool1', 'tool2', 'tool2b', 'join'].map(id => node(id, {
      kind: id === 'q' ? 'query' : 'execution', rollout_id: 'r',
    }));
    const edges = [['q', 'a'], ['a', 'tool1'], ['a', 'tool2'], ['tool2', 'tool2b'], ['tool1', 'join'], ['tool2b', 'join']]
      .map(([source_node_id, target_node_id]) => ({ source_node_id, target_node_id, kind: 'sequence' as const }));
    const positions = executionPositions(nodes, edges);
    expect(positions.get('tool1')!.x).toBe(positions.get('tool2')!.x);
    expect(positions.get('tool1')!.y).not.toBe(positions.get('tool2')!.y);
    expect(positions.get('join')!.x).toBe(positions.get('tool2b')!.x + 300);
    expect(positions.get('join')!.y).toBe(positions.get('a')!.y);
    expect(ancestorPath('join', edges).nodes.size).toBe(6);
  });
  it('merges v3 edges and candidates without showing executions as answers', () => {
    const first = detail();
    first.tree.schema_version = 3;
    first.tree.nodes = [node('execution', { kind: 'execution' })];
    first.tree.rollouts = [node('rollout', { result_node_id: 'result' })];
    first.tree.edges = [{ source_node_id: 'execution', target_node_id: 'result', kind: 'sequence' }];
    const second = { ...first, tree: { ...first.tree,
      nodes: [node('result', { kind: 'outcome', rollout_id: 'rollout' })],
      outcomes: { result: { reward: 0 } },
    } };
    const merged = mergeDetail(first, second);
    expect(merged.tree.edges).toHaveLength(1);
    expect(merged.tree.nodes).toHaveLength(2);
    expect(candidateNodes(merged.tree)).toEqual(first.tree.rollouts);
    expect(merged.tree.outcomes.result.reward).toBe(0);
  });
  it('keeps pending candidates selectable without targeting a not-yet-created outcome', () => {
    const pending = node('rollout', { status: 'running', store_status: 'running', result_node_id: 'result' });
    expect(selectableId(pending)).toBe('rollout');
    expect(selectableId({ ...pending, store_status: 'succeeded' })).toBe('result');
    expect(selectableId(pending, [node('result', { kind: 'outcome' })])).toBe('result');
  });
});

describe('conditional tree reads', () => {
  it('reuses 304 responses and sends ETag', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(detail()), { headers: { etag: '"v1"' } }))
      .mockResolvedValueOnce(new Response(null, { status: 304 }));
    vi.stubGlobal('fetch', fetch);
    const client = createTreeClient('exp', 'abcdef123456');
    const signal = new AbortController().signal;
    const first = await client.detail('a'.repeat(32), signal);
    expect(await client.detail('a'.repeat(32), signal)).toBe(first);
    expect(fetch.mock.calls[1][1].headers).toEqual({ 'If-None-Match': '"v1"' });
  });
  it('rejects cross-run and stale-cursor responses explicitly', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ ...detail(), run_id: 'other' })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: '树已更新' }), { status: 409 })));
    const client = createTreeClient('exp', 'abcdef123456');
    const signal = new AbortController().signal;
    await expect(client.detail('a'.repeat(32), signal)).rejects.toThrow('不属于当前训练');
    await expect(client.detail('a'.repeat(32), signal, '1:100')).rejects.toMatchObject({ status: 409 });
  });
  it('reads an exact scoped node and refuses a different identity', async () => {
    const payload = { experiment_id: 'exp', run_id: 'abcdef123456', tree_id: 'tree',
      node: node('tool:1'), detail: { output: 'actual output' }, outcome: null };
    const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(payload)))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...payload, node: node('other') })));
    vi.stubGlobal('fetch', fetch);
    const client = createTreeClient('exp', 'abcdef123456');
    const signal = new AbortController().signal;
    expect((await client.node('tree', 'tool:1', signal)).detail?.output).toBe('actual output');
    expect(fetch.mock.calls[0][0]).toContain('/tree/nodes/tool%3A1?experiment_id=exp');
    await expect(client.node('tree', 'tool:1', signal)).rejects.toThrow('不属于当前执行节点');
  });
});

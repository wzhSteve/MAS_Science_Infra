import { useState } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { SampleSelection } from '../../app/navigation';
import type { TreeDetail, TreeNode } from './types';

const { mockState } = vi.hoisted(() => ({ mockState: vi.fn() }));
vi.mock('./useSampleResults', () => ({ useSampleResults: mockState }));
vi.mock('../../app/providers/RuntimeProvider', () => ({
  useRuntimeCommands: () => ({ viewTraining: vi.fn() }),
  useRuntimeEvents: () => ({ latest: '', seq: 0, error: null, connected: false }),
}));
vi.mock('../training/components/TrainingConsole', () => ({
  Snapshot: () => <button>本次配置</button>, TRAIN_STATE: { running: '运行中', succeeded: '已完成' },
}));
vi.mock('./BranchGraph', () => ({ default: ({ onSelect }: { onSelect: (id: string) => void }) =>
  <button onClick={() => onSelect('child')}>树中回答 child</button> }));
vi.mock('./ExecutionGraph', () => ({ default: ({ onSelect }: { onSelect: (id: string) => void }) =>
  <button onClick={() => onSelect('exec')}>hub 第2次执行</button> }));
import { SampleResults } from './SampleResults';

const parent: TreeNode = {
  node_id: 'parent', parent_id: 'query', kind: 'rollout', origin: 'initial',
  status: 'succeeded', store_status: 'succeeded', training_status: null, record_issues: [],
  attempt_id: 'attempt', attempt_sequence: 1, site_id: null, window_id: null,
  event_id: null, boundary_snapshot_ref: null, reward: null, verdict: null,
  execution_error: null, judgments: [], previous_attempts: [], metrics: {}, decision: {},
};
const data: TreeDetail = {
  experiment_id: 'exp', run_id: 'abcdef123456', run_state: 'running',
  tree: { tree_id: 'a'.repeat(32), query: '真实题目', query_truncated: false,
    group_id: 'group', mode: 'train', revision: 1, updated_at: null,
    nodes: [parent, { ...parent, node_id: 'child', parent_id: 'parent', origin: 'branch' }],
    pending_nodes: [], plans: [], issues: [], outcomes: {
      parent: { answer: '父回答内容', reward: 0 }, child: { answer: '子回答内容', reward: 0.5 },
    } },
  page: { total: 2, next_cursor: null, truncated: false, plan_total: 0, next_plan_offset: null },
};
function Harness({ view = 'answers' }: { view?: SampleSelection['view'] }) {
  const [selection, setSelection] = useState<SampleSelection>({
    runId: data.run_id, treeId: data.tree.tree_id, view,
  });
  return <SampleResults experimentId="exp" selection={selection} active onNavigate={setSelection} onHistory={vi.fn()} />;
}
afterEach(cleanup);

it('starts with answers and opens the same detail from either view without automatic switching', async () => {
  const resource = (value: unknown) => ({ data: value, loading: false, error: null, refresh: vi.fn() });
  mockState.mockReturnValue({
    run: resource({ state: 'running' }), list: resource({ items: [], total: 0 }), detail: resource(data),
    offset: 0, mode: '', refresh: vi.fn(), filter: vi.fn(), setOffset: vi.fn(), resetWindow: vi.fn(),
    pages: 1, planPages: 1,
  });
  render(<Harness />);
  expect(screen.queryByRole('button', { name: /返回|←/ })).toBeNull();
  expect(screen.getByRole('button', { name: '训练记录' })).toBeTruthy();
  expect(screen.getByRole('button', { name: '回答列表' }).getAttribute('aria-pressed')).toBe('true');
  expect(screen.queryByLabelText('回答 child详情')).toBeNull();
  expect(screen.queryByText('树中回答 child')).toBeNull();
  expect(screen.getByText('父回答内容')).toBeTruthy();
  fireEvent.click(screen.getAllByRole('button', { name: '查看详情' })[1]);
  expect(screen.getByLabelText('回答 child详情')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '分支关系' }));
  await waitFor(() => expect(screen.getByText('树中回答 child')).toBeTruthy());
  expect(screen.getByLabelText('回答 child详情')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '关闭回答详情' }));
  expect(screen.queryByLabelText('回答 child详情')).toBeNull();
  fireEvent.click(screen.getByText('树中回答 child'));
  expect(screen.getByLabelText('回答 child详情')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '查看来源回答' }));
  expect(screen.getByLabelText('回答 parent详情')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '回答列表' }));
  expect(screen.getByLabelText('回答 parent详情')).toBeTruthy();
  expect(screen.queryByText('Agent 执行过程')).toBeNull();
});

it('shows an illustrated empty state while keeping errors distinct from missing data', () => {
  const resource = (value: unknown, error: string | null = null) => ({ data: value, loading: false, error, refresh: vi.fn() });
  mockState.mockReturnValue({
    run: resource({ state: 'failed' }), list: resource({ items: [], total: 0, availability: 'missing' }),
    detail: resource(null, '读取失败'), offset: 0, mode: '', refresh: vi.fn(), filter: vi.fn(),
    setOffset: vi.fn(), resetWindow: vi.fn(), pages: 1, planPages: 1,
  });
  const { container } = render(<Harness />);
  expect(screen.getByText('暂无题目记录')).toBeTruthy();
  expect(screen.getByText('暂时无法读取回答')).toBeTruthy();
  expect(container.querySelectorAll('.sample-placeholder svg[aria-hidden="true"]')).toHaveLength(2);
});

it('keeps failed answers and actual negative reward visible below a failure summary', () => {
  const failed = { ...data, tree: { ...data.tree,
    nodes: [{ ...parent, status: 'failed', execution_error: 'execution_error' }],
    outcomes: { parent: { reward: -1, error_details: {
      stage: 'graph_build', error_type: 'ValueError', message: 'Invalid node name',
    } } } }, page: { ...data.page, total: 1 } };
  const resource = (value: unknown) => ({ data: value, loading: false, error: null, refresh: vi.fn() });
  mockState.mockReturnValue({
    run: resource({ state: 'failed' }), list: resource({ items: [{ tree_id: data.tree.tree_id,
      group_id: 'group', mode: 'train', created_at: '2026-09-28T05:00:00Z',
      rollout_count: 1, failed_count: 1, completed_count: 1, missing_result_count: 1, branch_count: 0,
    }], total: 1 }), detail: resource(failed),
    offset: 0, mode: '', refresh: vi.fn(), filter: vi.fn(), setOffset: vi.fn(), resetWindow: vi.fn(),
    pages: 1, planPages: 1,
  });

  render(<Harness />);
  expect(screen.getByText('本组尝试均执行失败')).toBeTruthy();
  expect(screen.getByText('-1')).toBeTruthy();
  expect(screen.getByRole('button', { name: '查看详情' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '查看详情' }));
  expect(screen.getByLabelText('执行失败原因')).toBeTruthy();
  expect(screen.getByText('Invalid node name')).toBeTruthy();
});

it('defaults v3 to execution graph, reads true details, and respects an explicit answers choice', async () => {
    const execution: TreeNode = { ...parent, node_id: 'exec', kind: 'execution', agent_id: 'hub', turn: 2 };
    const v3: TreeDetail = { ...data, tree: { ...data.tree, schema_version: 3,
      nodes: [execution, { ...parent, node_id: 'result', kind: 'outcome', rollout_id: 'parent' }],
      rollouts: [{ ...parent, result_node_id: 'result' }], edges: [], outcomes: { result: { answer: '真实最终答案', reward: 0 } },
      issues: ['prefix_unconfirmed', 'trace_unavailable'],
    } };
    const resource = (value: unknown) => ({ data: value, loading: false, error: null, refresh: vi.fn() });
    mockState.mockReturnValue({
      run: resource({ state: 'running' }), list: resource({ items: [], total: 0 }), detail: resource(v3),
      nodeDetail: resource({ node: execution, detail: { input: [{ role: 'user', content: '真实输入' }],
        output: '<script>不执行</script>', reasoning: null, tool_calls: [{ function: { name: 'Python', arguments: '{"code":"print(1)"}' } }],
        observation: '1', content_truncated: true, elapsed_seconds: 0 } }),
      offset: 0, mode: '', refresh: vi.fn(), pages: 1, planPages: 1,
    });
    const { container, rerender } = render(<Harness view="auto" />);
    expect(screen.getByRole('button', { name: '执行树' }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByText('prefix_unconfirmed')).toBeTruthy();
    expect(screen.getByText('trace_unavailable')).toBeTruthy();
    await waitFor(() => expect(screen.getByText('hub 第2次执行')).toBeTruthy());
    fireEvent.click(screen.getByText('hub 第2次执行'));
    expect(screen.getByLabelText('hub · 第2次执行详情')).toBeTruthy();
    expect(screen.getByText('<script>不执行</script>')).toBeTruthy();
    expect(container.querySelector('script')).toBeNull();
    expect(screen.getByText('真实输入')).toBeTruthy();
    expect(screen.getByText('{"code":"print(1)"}')).toBeTruthy();
    expect(screen.getByText('未记录独立 reasoning，不补造推理内容。')).toBeTruthy();
    expect(screen.getByText('保存内容已截断，以下不是完整执行正文。')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '回答列表' }));
    expect(screen.getAllByRole('button', { name: '查看详情' })).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: '查看详情' }));
    expect(screen.getByLabelText('回答 parent详情')).toBeTruthy();
    expect(screen.getAllByText('0').length).toBeGreaterThan(0);
    expect(screen.queryByText('最终奖励未记录，不代表奖励为 0。')).toBeNull();
    expect(screen.getByRole('button', { name: '回答列表' }).getAttribute('aria-pressed')).toBe('true');
    rerender(<Harness view="auto" />);
    expect(screen.getByRole('button', { name: '回答列表' }).getAttribute('aria-pressed')).toBe('true');
  });

  it('defaults v2 automatic views to answers and distinguishes missing final reward', () => {
    const resource = (value: unknown) => ({ data: value, loading: false, error: null, refresh: vi.fn() });
    mockState.mockReturnValue({ run: resource({ state: 'running' }), list: resource({ items: [], total: 0 }),
      detail: resource({ ...data, tree: { ...data.tree, outcomes: {} } }),
      offset: 0, mode: '', refresh: vi.fn(), pages: 1, planPages: 1,
    });
    render(<Harness view="auto" />);
    expect(screen.getByRole('button', { name: '回答列表' }).getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(screen.getAllByRole('button', { name: '查看详情' })[0]);
    expect(screen.getByText('最终奖励未记录，不代表奖励为 0。')).toBeTruthy();
    expect(screen.queryByRole('button', { name: '执行树' })).toBeNull();
  });

import { lazy, memo, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { GitBranch, List } from 'lucide-react';
import type { SampleSelection } from '../../app/navigation';
import { useRuntimeCommands, useRuntimeEvents } from '../../app/providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { Select } from '../../shared/ui/select';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { SampleHeader } from './SampleHeader';
import { SampleState } from './SampleState';
import { AnswerDetails } from './AnswerDetails';
import { AnswerList } from './AnswerList';
import { ExecutionDetails } from './ExecutionDetails';
import { candidateNodes, followedTreeId, MAX_PAGES, resultId, selectableId, terminalRun } from './model';
import { useSampleResults } from './useSampleResults';

const BranchGraph = lazy(() => import('./BranchGraph'));
const ExecutionGraph = lazy(() => import('./ExecutionGraph'));
const NO_EDGES: NonNullable<import('./types').TreeDetail['tree']['edges']> = [];

export const SampleResults = memo(function SampleResults({ experimentId, selection, active, onNavigate, onHistory }: {
  experimentId: string; selection: SampleSelection; active: boolean;
  onNavigate: (selection: SampleSelection) => void; onHistory: () => void;
}) {
  const state = useSampleResults(experimentId, selection.runId, selection.treeId, selection.nodeId, active);
  const events = useRuntimeEvents();
  const [pinned, setPinned] = useState(false);
  const pinnedRef = useRef(false);
  pinnedRef.current = pinned;
  const seenTreeEvent = useRef(events.seq);
  const treeRefresh = useRef<number | null>(null);
  const runState = useRef(state.run.data?.state);
  runState.current = state.run.data?.state;
  const navigation = useRef({ selection, onNavigate });
  useLayoutEffect(() => { navigation.current = { selection, onNavigate }; }, [selection, onNavigate]);
  useEffect(() => { setPinned(false); }, [selection.runId]);
  useEffect(() => {
    if (events.latest !== 'rollout_tree' || events.seq === seenTreeEvent.current) return;
    seenTreeEvent.current = events.seq;
    const current = navigation.current.selection;
    const nextTree = followedTreeId(events.data, current, pinnedRef.current, runState.current);
    if (nextTree) navigation.current.onNavigate({ ...current, treeId: nextTree, nodeId: undefined });
    if (treeRefresh.current !== null) return;
    treeRefresh.current = window.setTimeout(() => {
      treeRefresh.current = null;
      void state.refresh();
    }, 400);
  }, [events.latest, events.seq, events.data, state.refresh]);
  useEffect(() => () => { if (treeRefresh.current !== null) window.clearTimeout(treeRefresh.current); }, []);
  useEffect(() => {
    const page = state.detail.data?.page;
    if (!page?.next_cursor || state.pages >= MAX_PAGES || terminalRun(state.run.data?.state)) return;
    state.loadMore();
  }, [state.detail.data?.page, state.pages, state.run.data?.state, state.loadMore]);
  const { viewTraining } = useRuntimeCommands();
  const showLog = useCallback(() => viewTraining(selection.runId), [viewTraining, selection.runId]);
  const selectTree = useCallback((treeId: string, pin = false) => {
    if (pin) setPinned(true);
    navigation.current.onNavigate({
      ...navigation.current.selection, treeId, nodeId: undefined,
    });
  }, []);
  const selectAnswer = useCallback((nodeId: string) => navigation.current.onNavigate({
    ...navigation.current.selection, nodeId,
  }), []);
  const trigger = useRef<HTMLElement | null>(null);
  const closeDetails = useCallback(() => {
    navigation.current.onNavigate({ ...navigation.current.selection, nodeId: undefined });
    requestAnimationFrame(() => { if (trigger.current?.isConnected) trigger.current.focus({ preventScroll: true }); });
  }, []);
  const openAnswer = useCallback((id: string) => {
    trigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    selectAnswer(id);
  }, [selectAnswer]);
  useEffect(() => {
    if (active && !selection.treeId && state.list.data?.items[0]) selectTree(state.list.data.items[0].tree_id);
  }, [active, selection.treeId, state.list.data, selectTree]);
  const data = state.detail.data;
  const isExecutionTree = data?.tree.schema_version === 3;
  const nodes = useMemo(() => data ? candidateNodes(data.tree) : [], [data?.tree]);
  const candidate = nodes.find(node => node.node_id === selection.nodeId || resultId(node) === selection.nodeId);
  const selectedId = candidate ? selectableId(candidate, data?.tree.nodes) : selection.nodeId;
  const exact = state.nodeDetail?.data;
  const chosen = exact && exact.node.node_id === selectedId ? exact.node
    : data?.tree.nodes.find(node => node.node_id === selectedId) || candidate;
  const chosenOutcome = exact && exact.node.node_id === selectedId ? exact.outcome || undefined : data?.tree.outcomes[selectedId || ''];
  const automaticView = useRef<{ run: string; view: 'answers' | 'branches' }>();
  const runKey = `${experimentId}:${selection.runId}`;
  const defaultView = automaticView.current?.run === runKey ? automaticView.current.view : isExecutionTree ? 'branches' : 'answers';
  useEffect(() => {
    if (data && automaticView.current?.run !== runKey) automaticView.current = { run: runKey, view: isExecutionTree ? 'branches' : 'answers' };
  }, [data, runKey, isExecutionTree]);
  const view = selection.view === 'auto' ? defaultView : selection.view;
  const summary = state.list.data?.items.find(item => item.tree_id === selection.treeId);
  const [graphTree, setGraphTree] = useState<string>();
  useEffect(() => {
    if (view === 'branches') setGraphTree(selection.treeId);
  }, [view, selection.treeId]);
  const displayGraph = view === 'branches' || graphTree === selection.treeId;
  const run = state.run.data;
  const issues = state.run.error || state.list.error || state.detail.error;
  const allFailed = summary && summary.rollout_count > 0 && summary.failed_count === summary.rollout_count;
  return <section className="sample-results" aria-label="采样结果">
    <SampleHeader experimentId={experimentId} runId={selection.runId} run={run}
      runState={state.list.data?.run_state} updatedAt={data?.tree.updated_at}
      onHistory={onHistory} onLog={showLog} onRefresh={state.refresh} />
    {issues && <InlineNotice tone="warning">读取未完成，已加载内容可能陈旧：{issues}<Button size="sm" onClick={state.refresh}>重试</Button>
      {state.cursor && <Button size="sm" onClick={state.resetWindow}>从首批重新读取</Button>}
    </InlineNotice>}
    {state.list.data?.recording_status === 'degraded' && <InlineNotice tone="warning">本次运行的记录存在缺口，不代表所有回答执行失败。
      <details><summary>记录问题</summary>{state.list.data.recording_errors?.map(error => <p key={error}>{error}</p>)}</details>
    </InlineNotice>}
    <div className={`sample-body${chosen ? ' has-details' : ''}`}>
      <aside className="sample-topics" aria-label="题目记录">
        <div className="sample-topics-heading"><strong>题目记录</strong>
          <Select aria-label="筛选训练或验证" value={state.mode} onChange={event => {
            state.filter(event.target.value);
            onNavigate({ ...selection, treeId: undefined, nodeId: undefined });
          }}>
            <option value="">训练与验证</option><option value="train">训练</option><option value="val">验证</option>
          </Select></div>
        <div className="sample-topic-list">
          {state.list.data?.items.map(item => <button type="button" key={item.tree_id} className={`sample-topic${item.tree_id === selection.treeId ? ' is-selected' : ''}`}
            aria-pressed={item.tree_id === selection.treeId} onClick={() => selectTree(item.tree_id, true)}>
            <strong>{item.query || (item.availability === 'unreadable' ? '题目记录读取失败' : '题目内容未记录')}</strong>
            <span>{item.rollout_count} 条轨迹 · 已完成 {item.completed_count}/{item.rollout_count} · {item.branch_count ? `${item.branch_count} 次来自分支` : '无分支'}</span>
            <small>{item.mode === 'train' ? '训练' : '验证'} · {item.group_id.slice(0, 8)} · {new Date(item.created_at).toLocaleTimeString()}</small>
            {(item.failed_count > 0 || item.missing_result_count > 0 || item.error) && <small className="sample-warning">
              {item.error || `执行失败 ${item.failed_count} · 结果有缺失 ${item.missing_result_count}`}</small>}
          </button>)}
          {!state.list.data?.items.length && <SampleState
            kind={state.list.error ? 'error' : state.list.loading ? 'loading' : 'empty'}
            title={state.list.error ? '记录读取失败' : state.list.loading ? '正在读取题目' : '暂无题目记录'}
            description={state.list.error ? '请重试读取，不会影响训练。' : state.list.loading ? '题目记录准备就绪后显示在这里。' :
              state.list.data?.availability === 'missing' ? '此运行尚未生成记录，或创建于旧版本。' : '当前筛选范围内没有记录。'} />}
        </div>
        <footer><Button size="sm" disabled={state.offset === 0} onClick={() => state.setOffset(Math.max(0, state.offset - 20))}>上一页</Button>
          <small>{state.list.data?.total ?? '—'} 条记录</small>
          <Button size="sm" disabled={state.list.data?.next_offset == null} onClick={() => state.setOffset(state.list.data!.next_offset!)}>下一页</Button></footer>
      </aside>
      <div className="sample-main" role="region" aria-label="题目回答">
        {!data ? <SampleState kind={state.detail.error ? 'error' : selection.treeId ? 'loading' : 'select'}
          title={state.detail.error ? '暂时无法读取回答' : selection.treeId ? '正在读取回答' : '从一道题目开始'}
          description={state.detail.error ? '记录仍保留在服务器，请重试读取。' : selection.treeId ? '正在加载这次采样的结果与来源。' : '选择左侧题目，查看回答与分支关系。'}>
          {state.detail.error && <Button size="sm" onClick={state.refresh}>重试读取</Button>}
        </SampleState> : <>
          <div className="sample-question"><h2>{data.tree.query || '题目内容未记录'}</h2>
            {data.tree.query_truncated && <small>题目为截断摘要</small>}
            <p>{data.tree.mode === 'train' ? '训练' : '验证'} · 采样记录 {data.tree.group_id.slice(0, 12)}
              {summary && (allFailed ? ` · ${summary.rollout_count} 次尝试均执行失败`
                : ` · 已结束 ${summary.completed_count}/${summary.rollout_count}${summary.failed_count ? ` · 执行失败 ${summary.failed_count}` : ''}`)}
            </p>
            <div className="sample-view-switch" role="group" aria-label="结果查看方式">
              <Button size="sm" variant={view === 'answers' ? 'primary' : 'ghost'} aria-pressed={view === 'answers'}
                onClick={() => onNavigate({ ...selection, view: 'answers' })}><List size={14} />回答列表</Button>
              <Button size="sm" variant={view === 'branches' ? 'primary' : 'ghost'} aria-pressed={view === 'branches'}
                onClick={() => onNavigate({ ...selection, view: 'branches' })}><GitBranch size={14} />{isExecutionTree ? '执行树' : '分支关系'}</Button>
              {pinned && !terminalRun(state.run.data?.state) && <Button size="sm" variant="ghost" onClick={() => setPinned(false)}>跟随正在采样的题目</Button>}
            </div>
            {allFailed && <SampleState kind="failed" compact title="本组尝试均执行失败"
              description="失败记录与实际奖励仍保留在下方。请查看训练日志定位原因。">
              <Button size="sm" variant="ghost" onClick={showLog}>查看失败日志</Button>
            </SampleState>}
          </div>
          {!isExecutionTree && <p className="field-hint">旧版记录仅保存回答及来源，未记录逐节点执行过程。</p>}
          {isExecutionTree && !data.tree.nodes.some(node => node.kind === 'execution') &&
            <p className="field-hint">尚无实际执行节点，正在等待执行记录；候选采样状态可在回答列表查看。</p>}
          {data.tree.issues.length > 0 && <InlineNotice tone="warning">
            <strong>执行记录存在缺口，当前图不代表完整执行过程。</strong>
            {data.tree.issues.map(issue => <p key={issue}>{issue}</p>)}
          </InlineNotice>}
          {isExecutionTree && state.nodeDetail?.error && chosen?.kind !== 'execution' && chosen?.kind !== 'query' &&
            <InlineNotice tone="warning">节点内容读取失败：{state.nodeDetail.error}
              <Button size="sm" onClick={() => void state.nodeDetail.refresh()}>重试节点读取</Button></InlineNotice>}
          {selection.nodeId && !chosen && <InlineNotice tone="warning">
            {isExecutionTree ? (state.nodeDetail?.loading ? '正在定位所选节点…' : '所选节点无法读取，未用其他记录代替。') : data.lookup_node_id === selection.nodeId
              ? '所选回答尚未加载或来源无法确认。已进行有界查找，不会用其他记录代替。'
              : '正在定位所选回答…'}
            <Button size="sm" onClick={closeDetails}>关闭选择</Button></InlineNotice>}
          {(state.cursor || state.detail.error) && <Button size="sm" variant="ghost" onClick={state.resetWindow}>从首批重新读取回答</Button>}
          <div key={selection.treeId} className="sample-views">
            <div className="sample-answer-scroll" hidden={view !== 'answers'}>
              <AnswerList nodes={nodes} outcomes={data.tree.outcomes} mode={data.tree.mode} selectedId={selectedId} onSelect={openAnswer} />
            </div>
            {displayGraph && <div className="sample-graph-host" hidden={view !== 'branches'}>
              <Suspense fallback={<SampleState kind="loading" title="正在加载分支关系" />}>
                {isExecutionTree
                  ? <ExecutionGraph key={selection.treeId} nodes={data.tree.nodes} edges={data.tree.edges || NO_EDGES} outcomes={data.tree.outcomes} mode={data.tree.mode} selectedId={selectedId} onSelect={openAnswer} />
                  : <BranchGraph nodes={data.tree.nodes} outcomes={data.tree.outcomes} selectedId={selectedId} onSelect={openAnswer} />}
              </Suspense>
            </div>}
          </div>
          {data.tree.pending_nodes.length > 0 && <p className="sample-warning">有 {data.tree.pending_nodes.length} 条记录的来源待关联，未伪装为树根。</p>}
          <div className="sample-load-more"><small>{isExecutionTree
            ? `当前窗口显示 ${data.tree.nodes.length} 个节点（含问题与必要来源），过程记录总数 ${data.page.total}。`
            : `当前窗口已加载 ${nodes.length}/${data.page.total} 条回答（含必要来源）。`}奖励与完成状态不等于答案正确。
            {isExecutionTree && ` 候选回答 ${nodes.length} 条，仅显示已加载范围。`}</small>
            {data.page.next_cursor && <Button size="sm" onClick={() => {
              if (state.pages >= MAX_PAGES) { closeDetails(); state.nextWindow(); }
              else state.loadMore();
            }}>{state.pages >= MAX_PAGES ? '查看下一批记录' : '加载更多记录'}</Button>}</div>
          {data.page.plan_total > 0 && <details className="sample-plans"><summary>分支计划 · 已加载 {data.tree.plans.length}/{data.page.plan_total}</summary>
            {data.tree.plans.filter(plan => plan.status !== 'enqueued').map(plan => <p key={plan.plan_id}>
              {plan.site_id || '位置未记录'} · {plan.status === 'skipped' ? '未执行' : '等待入队'}
              {' · '}{({ budget_exhausted: '预算不足', missing_resume_messages: '缺少恢复前缀' })[plan.reason || ''] || plan.reason || '原因未记录'}
            </p>)}
            {!data.tree.plans.some(plan => plan.status !== 'enqueued') && <p>已加载的计划均已入队。</p>}
            {data.page.next_plan_offset !== null && <Button size="sm" disabled={state.planPages >= MAX_PAGES} onClick={() => state.loadMore(true)}>
              {state.planPages >= MAX_PAGES ? '已达计划加载上限' : '加载更多计划'}</Button>}
          </details>}
        </>}
      </div>
      {chosen && data && (chosen.kind === 'execution' || chosen.kind === 'query'
        ? <ExecutionDetails key={`${selection.treeId}:${chosen.node_id}`} node={chosen} query={data.tree.query}
          detail={exact?.node.node_id === chosen.node_id ? exact.detail : undefined}
          loading={state.nodeDetail?.loading ?? false} error={state.nodeDetail?.error}
          onRetry={() => void state.nodeDetail.refresh()} onClose={closeDetails} />
        : <AnswerDetails key={`${selection.treeId}:${chosen.node_id}`} node={chosen} outcome={chosenOutcome} mode={data.tree.mode}
          unresolved={data.tree.pending_nodes.some(n => n.node_id === chosen.node_id)} onSelect={selectAnswer} onClose={closeDetails} />)}
    </div>
  </section>;
});

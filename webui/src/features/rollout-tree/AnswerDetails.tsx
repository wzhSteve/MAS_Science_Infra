import { memo, useEffect, useRef, useState } from 'react';
import { ArrowLeft, Maximize2, Minimize2, X } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { answerName, answerSource, answerState, rewardText } from './model';
import type { Outcome, TreeNode } from './types';
import { SampleState } from './SampleState';

export const AnswerDetails = memo(function AnswerDetails({ node, outcome, mode, unresolved, onSelect, onClose }: {
  node: TreeNode; outcome?: Outcome; mode?: string; unresolved: boolean; onSelect: (id: string) => void; onClose: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const root = useRef<HTMLElement>(null);
  useEffect(() => { root.current?.focus({ preventScroll: true }); }, []);
  const state = answerState(node, outcome);
  const title = answerName(node.rollout_id || node.node_id);
  return <aside ref={root} tabIndex={-1} className={`sample-details${expanded ? ' is-expanded' : ''}`}
    aria-label={`${title}详情`} onKeyDown={event => {
      if (event.key === 'Escape') { event.stopPropagation(); onClose(); }
    }}>
    <header><h3>{title}</h3>
      <Button size="sm" variant="ghost" aria-label={expanded ? '恢复详情宽度' : '展开阅读'} onClick={() => setExpanded(value => !value)}>
        {expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</Button>
      <Button size="sm" variant="ghost" aria-label="关闭回答详情" onClick={onClose}><X size={16} /></Button></header>
    <div className="sample-details-scroll">
      <div className="sample-result-heading"><span className={`sample-state is-${state.tone}`}>{state.label}</span>
        <span>最终奖励 <strong>{rewardText(outcome?.reward)}</strong>{mode === 'val' ? ' · 验证集' : mode === 'train' ? ' · 训练集' : ''}</span></div>
      {node.record_issues.length > 0 && <p className="sample-warning">记录存在冲突或缺口，以下内容不能视为完整结果。</p>}
      {outcome?.error_details && <section className="sample-execution-error" aria-label="执行失败原因">
        <h4>执行失败原因</h4>
        <p>{({ graph_build: '工作流图构建', graph_execution: '工作流执行', result_processing: '结果处理' })[outcome.error_details.stage] || outcome.error_details.stage}
          {' · '}{outcome.error_details.error_type}</p>
        <p className="sample-full-answer">{outcome.error_details.message}</p>
      </section>}
      <section><h4>回答内容</h4>{outcome?.answer != null ? <p className="sample-full-answer">{outcome.answer}</p>
        : <SampleState compact kind={node.status === 'failed' ? 'failed' : 'empty'} title="未记录最终答案" />}
        {outcome?.answer_truncated && <p className="field-hint">这里只保存了答案摘要，内容已截断；没有可用的完整答案下载。</p>}
        {outcome?.reward == null && <p className="field-hint">最终奖励未记录，不代表奖励为 0。</p>}
      </section>
      <section><h4>生成来源</h4><p>{unresolved ? '父记录尚未关联，来源待确认。' : answerSource(node)}</p>
        {node.origin === 'branch' && <>
          <p className="field-hint">从来源回答执行过程的一个中间位置重新生成，不是把其最终答案作为输入。</p>
          <p>分支位置：{node.site_id || '未记录'}{node.window_id ? ` · ${node.window_id}` : ''}</p>
          {node.parent_id && <Button size="sm" onClick={() => onSelect(node.parent_id!)}><ArrowLeft size={13} />查看来源回答</Button>}
        </>}
      </section>
      <details><summary>训练判定</summary>
        <dl><dt>节点 credit</dt><dd>{node.reward == null ? '未计算 / 未记录' : rewardText(node.reward)}</dd>
          <dt>判定</dt><dd>{node.verdict || (node.judgments.length ? '查看多项判定明细' : '未记录')}</dd>
          <dt>训练样本</dt><dd>{({ adapted: '已适配，接纳状态待确认', empty: '未提取到训练样本', accepted: '已进入训练批次', rejected: '适配拒绝' })[node.training_status || ''] || '未记录'}</dd></dl>
        {node.judgments.length > 0 && <pre>{JSON.stringify(node.judgments, null, 2)}</pre>}
        <p className="field-hint">最终奖励与节点 credit 含义不同，奖励高不等于答案正确或该窗口贡献更大。此页面未自动运行诊断。</p>
      </details>
      <details><summary>技术详情</summary>
        <dl>{Object.entries({
          'Rollout ID': node.rollout_id || node.node_id, 'Parent ID': node.parent_id,
          'Attempt': node.attempt_id, '尝试序号': node.attempt_sequence || '未记录',
          'Store 状态': node.store_status, '执行状态': node.status,
          '窗口': node.window_id, '事件': node.event_id, '快照引用': node.boundary_snapshot_ref,
        }).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? '未记录'}</dd></div>)}</dl>
        {node.record_issues.map(issue => <p key={issue} className="sample-warning">{issue}</p>)}
        <h4>Gate 与指标</h4><pre>{JSON.stringify({ decision: node.decision, metrics: node.metrics }, null, 2)}</pre>
        {node.previous_attempts.length > 0 && <><h4>历史尝试记录</h4><pre>{JSON.stringify(node.previous_attempts, null, 2)}</pre></>}
      </details>
    </div>
  </aside>;
});

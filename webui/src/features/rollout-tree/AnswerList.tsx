import { memo } from 'react';
import { Button } from '../../shared/ui/button';
import { answerName, answerSource, answerState, resultId, rewardText, selectableId } from './model';
import type { Outcome, TreeNode } from './types';
import { SampleState } from './SampleState';

const AnswerCard = memo(function AnswerCard({ node, outcome, mode, selected, onSelect }: {
  node: TreeNode; outcome?: Outcome; mode?: string; selected: boolean; onSelect: (id: string) => void;
}) {
  const state = answerState(node, outcome);
  return <article className={`sample-answer${selected ? ' is-selected' : ''}`} aria-label={answerName(node.node_id)}>
    <header><strong title={node.node_id}>{answerName(node.node_id)}</strong>
      <span className={`sample-state is-${state.tone}`}>{state.label}</span>
      <span className="sample-reward">最终奖励 <b>{rewardText(outcome?.reward)}</b>{mode === 'val' ? ' · 验证集' : mode === 'train' ? ' · 训练集' : ''}</span></header>
    <p className="sample-source">{answerSource(node)}</p>
    {outcome?.answer != null ? <p className="sample-answer-preview">{outcome.answer}</p>
      : <SampleState compact kind={node.status === 'failed' ? 'failed' : node.status === 'running' || node.status === 'enqueued' ? 'loading' : 'empty'}
        title={node.status === 'running' || node.status === 'enqueued' ? '等待回答生成' : '未记录最终答案'} />}
    <footer>{outcome?.answer_truncated && <small>已记录答案为截断摘要</small>}
      <Button size="sm" variant="ghost" aria-pressed={selected} onClick={() => onSelect(selectableId(node))}>查看详情</Button></footer>
  </article>;
});

export const AnswerList = memo(function AnswerList({ nodes, outcomes, mode, selectedId, onSelect }: {
  nodes: TreeNode[]; outcomes: Record<string, Outcome>; mode?: string; selectedId?: string; onSelect: (id: string) => void;
}) {
  return <div className="sample-answer-list">
    {nodes.map(node => <AnswerCard key={node.node_id} node={node} outcome={outcomes[resultId(node)]} mode={mode}
      selected={selectedId === node.node_id || selectedId === resultId(node)} onSelect={onSelect} />)}
    {!nodes.length && <SampleState title="暂无生成记录" description="实际入队的回答会显示在这里，未执行的分支计划单独保留。" />}
  </div>;
});

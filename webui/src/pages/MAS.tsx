import { useState } from 'react';
import type { Bundle, RlConfig } from '../shared/api/types';
import { MasGraphEditor, WorkflowActions, ParquetCollect, useMasDraft, type RlSettingsSlot } from '../features/mas';
import { PageHeader } from '../shared/components/PageHeader';
import { Section } from '../shared/components/Section';
import { StatusBadge } from '../shared/components/StatusBadge';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';

export type MASPanelProps = {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  normalizeRl: (rl: RlConfig) => RlConfig;
  renderRlSettings?: RlSettingsSlot;
};

export function MASPanel(props: MASPanelProps) {
  const draft = useMasDraft(props);
  const [algo, setAlgo] = useState('grpo');
  const { workflow } = draft;
  if (!workflow) return <LoadingState label="加载 workflow…" />;

  return <div className="page-stack">
    <PageHeader title="MAS · Workflow" eyebrow="MULTI-AGENT SYSTEM"
      description={<>编辑 Agent / Tool 与连线，保存为 WorkflowSpec。topology · <code>{workflow.topology}</code></>}
      actions={<>
        <StatusBadge tone={draft.workflowDirty ? 'warning' : 'neutral'}>{draft.workflowDirty ? '图未保存' : '图已同步'}</StatusBadge>
        {draft.rlDirty && <StatusBadge tone="warning">训练简参未保存</StatusBadge>}
      </>} />
    <Section className="mas-workspace-section">
      <WorkflowActions pending={draft.pending} executable={Boolean(props.bundle?.executable?.ok)}
        onSave={draft.save} onCollect={draft.collect} algo={algo} onAlgoChange={setAlgo} />
      {draft.notice && <InlineNotice tone={draft.notice.tone}>{draft.notice.message}</InlineNotice>}
      {draft.rlNotice && <InlineNotice tone={draft.rlNotice.tone}>{draft.rlNotice.message}</InlineNotice>}
      {!props.bundle?.executable?.ok && <InlineNotice tone="danger">
        {props.bundle?.executable?.reason || 'Workflow 不可执行'}
      </InlineNotice>}
      <MasGraphEditor workflow={workflow} palette={draft.palette} onChange={draft.onWorkflowChange}
        rl={draft.rl} onRlPatch={draft.onRlPatch} onRlSave={draft.saveRl} savingRl={draft.savingRl}
        renderRlSettings={props.renderRlSettings} />
    </Section>
    <ParquetCollect pending={draft.pending} executable={Boolean(props.bundle?.executable?.ok)} algo={algo}
      onCollect={draft.collect} onPreview={draft.preview} rows={draft.rows} />
  </div>;
}

import type { Bundle, MetaResponse } from '../../../shared/api/types';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { LoadingState } from '../../../shared/components/LoadingState';
import { LogPanel } from '../../../shared/components/LogPanel';
import { PageHeader } from '../../../shared/components/PageHeader';
import { useAction } from '../../../shared/hooks/useAction';
import { Button } from '../../../shared/ui/button';
import { useRlDraft } from '../model/useRlDraft';
import { RlSettingsForm } from './RlSettingsForm';
import { RlTrainingActions } from './RlTrainingActions';

type Props = {
  expId: string;
  bundle: Bundle | null;
  meta: MetaResponse | null;
  onReload: () => void;
  trainRunId: string | null;
  trainRunning: boolean;
  trainLog: string;
  aglOnline: boolean;
  onStartTrain: () => Promise<void>;
  onStopTrain: () => Promise<void>;
  onRefreshLog: () => Promise<void>;
};

export function RLPanel(props: Props) {
  if (!props.bundle || props.bundle.id !== props.expId) return <LoadingState label="加载 rl.yaml…" />;
  return <RlWorkspace key={props.expId} {...props} bundle={props.bundle} />;
}

function RlWorkspace({ expId, bundle, meta, onReload, trainRunId, trainRunning, trainLog, aglOnline, onStartTrain, onStopTrain, onRefreshLog }: Props & { bundle: Bundle }) {
  const draft = useRlDraft(expId, bundle.rl, onReload, onStartTrain);
  const refresh = useAction();
  return <div className="page-stack settings-page">
    <PageHeader title="RL" eyebrow="强化学习" description="管理训练超参与运行状态。参数草稿与 MAS 训练简参独立保存。" />
    <RlSettingsForm rl={draft.rl} meta={meta} dirty={draft.dirty} pending={draft.pending} onPatch={draft.patch} onSave={draft.save} onRecommend={draft.recommend} onPrefill={draft.prefill} />
    {draft.notice && <InlineNotice tone={draft.notice.tone}>{draft.notice.message}</InlineNotice>}
    <RlTrainingActions trainRunId={trainRunId} trainRunning={trainRunning} aglOnline={aglOnline} pending={draft.pending} onStart={draft.start} onStop={onStopTrain} />
    <LogPanel title="训练日志" log={trainLog} empty="训练 stdout 将在此显示；后台刷新不会覆盖上方配置草稿。" actions={trainRunId ?
      <Button size="sm" loading={refresh.pending !== null} onClick={() => { void refresh.run('refresh', onRefreshLog); }}>刷新日志</Button> : undefined} />
    {refresh.notice && <InlineNotice tone={refresh.notice.tone}>{refresh.notice.message}</InlineNotice>}
  </div>;
}

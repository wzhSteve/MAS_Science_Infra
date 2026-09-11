import { Play, ScanLine, FlaskConical } from 'lucide-react';
import { useCommandStatus, useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { InlineNotice } from '../../shared/components/InlineNotice';
import type { Bundle } from '../../shared/api/types';

export function GlobalActionBar({ executable }: { executable: Bundle['executable'] }) {
  const { collect, diagnose, startTrain } = useRuntimeCommands();
  const { pending, notice } = useCommandStatus();
  const { data } = useTraining();
  return <div className="global-action-area">
    <div className="global-actions" aria-label="快捷运行">
      <Button size="sm" disabled={!!pending || executable.ok === false} loading={pending === 'collect'}
        title={executable.ok === false ? executable.reason : '使用已保存的 workflow，mock=true，n=1'} onClick={() => void collect()}>
        <FlaskConical size={14} aria-hidden="true" />Collect <span className="button-annotation">mock</span>
      </Button>
      <Button size="sm" disabled={!!pending} loading={pending === 'diagnose'} title="使用已保存的 Harness 配置" onClick={() => void diagnose()}>
        <ScanLine size={14} aria-hidden="true" />Diagnose
      </Button>
      <Button size="sm" variant="primary" disabled={!!pending || data?.running} loading={pending === 'train'} onClick={() => void startTrain()}>
        <Play size={13} aria-hidden="true" />Train
      </Button>
    </div>
    {executable.ok === false && <p className="field-error action-reason">Collect 不可用：{executable.reason}</p>}
    {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
  </div>;
}

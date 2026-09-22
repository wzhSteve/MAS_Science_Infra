import { Play, ScanLine } from 'lucide-react';
import { useCommandStatus, useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { InlineNotice } from '../../shared/components/InlineNotice';

export function GlobalActionBar() {
  const { diagnose, startTrain } = useRuntimeCommands();
  const { pending, notice } = useCommandStatus();
  const { data } = useTraining();
  return <div className="global-action-area">
    <div className="global-actions" aria-label="快捷运行">
      <Button size="sm" disabled={!!pending} loading={pending === 'diagnose'} title="使用已保存的 Harness 配置" onClick={() => void diagnose()}>
        <ScanLine size={14} aria-hidden="true" />Diagnose
      </Button>
      <Button size="sm" variant="primary" disabled={!!pending || data?.running} loading={pending === 'train'} onClick={() => void startTrain()}>
        <Play size={13} aria-hidden="true" />Train
      </Button>
    </div>
    {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
  </div>;
}

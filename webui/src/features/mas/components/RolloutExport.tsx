import { useEffect, useRef, useState } from 'react';
import { Download } from 'lucide-react';
import { masApi } from '../api';
import { errorMessage } from '../../../shared/api/http';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';

export function RolloutExport({ experimentId, runId }: { experimentId: string; runId: string }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const download = async () => {
    const controller = new AbortController();
    request.current = controller;
    setLoading(true); setError(null);
    try {
      const response = await fetch(masApi.rolloutExportUrl(experimentId, runId), { signal: controller.signal });
      if (!response.ok) throw new Error(`导出请求失败（HTTP ${response.status}），请检查记录是否存在或损坏。`);
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url; link.download = `rollout-${runId}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  };
  return <div className="mas-rollout-export">
    <Button size="sm" variant="ghost" loading={loading} disabled={loading} onClick={() => void download()}>
      <Download size={14} />下载完整 JSON
    </Button>
    <span className="field-hint">包含本次任务、Prompt 和工具返回，不含传输凭据。</span>
    {error && <InlineNotice tone="danger">{error}</InlineNotice>}
  </div>;
}

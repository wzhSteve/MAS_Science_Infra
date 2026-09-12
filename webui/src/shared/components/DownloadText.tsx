import { memo, useState } from 'react';
import { Download } from 'lucide-react';
import { Button } from '../ui/button';
import { InlineNotice } from './InlineNotice';

export const DownloadText = memo(function DownloadText({ text, filename, label = '下载详细日志' }: {
  text: string; filename: string; label?: string;
}) {
  const [error, setError] = useState(false);
  const download = () => {
    setError(false);
    let url: string | undefined;
    const link = document.createElement('a');
    try {
      url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
      link.href = url; link.download = filename; link.hidden = true;
      document.body.appendChild(link); link.click();
    } catch { setError(true); }
    finally {
      link.remove();
      if (url) { const value = url; setTimeout(() => URL.revokeObjectURL(value), 1000); }
    }
  };
  return <>
    <Button size="sm" variant="ghost" onClick={download}><Download size={13} />{label}</Button>
    {error && <InlineNotice tone="warning">下载失败，请重试。</InlineNotice>}
  </>;
});

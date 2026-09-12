import { memo, useState } from 'react';
import { Button } from '../ui/button';
import { InlineNotice } from './InlineNotice';

const PAGE_SIZE = 16_000;

function boundary(text: string, offset: number) {
  const index = Math.min(offset, text.length);
  const current = text.charCodeAt(index);
  const previous = text.charCodeAt(index - 1);
  // Keep a surrogate pair on one page without scanning the loaded text.
  return current >= 0xdc00 && current <= 0xdfff && previous >= 0xd800 && previous <= 0xdbff ? index - 1 : index;
}

export const TextContent = memo(function TextContent({ text, label = '文本内容', className }: {
  text: string; label?: string; className?: string;
}) {
  const [page, setPage] = useState(0);
  const [downloadError, setDownloadError] = useState(false);
  const pageCount = Math.max(1, Math.ceil(text.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  if (page !== currentPage) setPage(currentPage);
  const start = boundary(text, currentPage * PAGE_SIZE);
  const end = boundary(text, (currentPage + 1) * PAGE_SIZE);
  const download = () => {
    setDownloadError(false);
    let url: string | undefined;
    const link = document.createElement('a');
    try {
      url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
      link.href = url;
      link.download = 'loaded-text.txt';
      link.hidden = true;
      document.body.appendChild(link);
      link.click();
    } catch {
      setDownloadError(true);
    } finally {
      link.remove();
      if (url) {
        const objectUrl = url;
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1_000);
      }
    }
  };
  return <>
    <pre className={className} tabIndex={0} aria-label={label}>{text.slice(start, end)}</pre>
    <div className="field-hint text-content-controls" role="group" aria-label={`${label}分页与下载`}>
      <span role="status">已加载文本：{text.length ? start + 1 : 0}–{end} / {text.length} 个 UTF-16 码元；第 {currentPage + 1} / {pageCount} 页。</span>
      {pageCount > 1 && <>
        <Button size="sm" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>上一页</Button>
        <Button size="sm" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>下一页</Button>
      </>}
      <Button size="sm" onClick={download}>下载全部已加载文本</Button>
    </div>
    {downloadError && <InlineNotice tone="warning">无法下载已加载文本，请重试下载。</InlineNotice>}
  </>;
});

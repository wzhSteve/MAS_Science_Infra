import { memo, type ReactNode } from 'react';
import { EmptyState } from './EmptyState';
import { Section } from './Section';
import { TextContent } from './TextContent';

export const LogPanel = memo(function LogPanel({ log, actions, empty, title = '训练日志' }: {
  log: string; actions?: ReactNode; empty?: ReactNode; title?: string;
}) {
  return <Section title={title} actions={actions} className="log-section">
    {log ? <>
      <TextContent text={log} label={title} className="log-pre" />
      <p className="field-hint">分页与下载仅包含当前已加载的日志文本，不代表后端完整日志。</p>
    </> : <EmptyState title="尚无训练 stdout">{empty}</EmptyState>}
  </Section>;
});

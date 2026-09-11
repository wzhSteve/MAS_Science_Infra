import { memo, type ReactNode } from 'react';
import { EmptyState } from './EmptyState';
import { Section } from './Section';

export const LogPanel = memo(function LogPanel({ log, actions, empty, title = '训练日志' }: {
  log: string; actions?: ReactNode; empty?: ReactNode; title?: string;
}) {
  return <Section title={title} actions={actions} className="log-section">
    {log ? <pre className="log-pre" tabIndex={0} aria-label={title}>{log}</pre> : <EmptyState title="尚无训练 stdout">{empty}</EmptyState>}
  </Section>;
});

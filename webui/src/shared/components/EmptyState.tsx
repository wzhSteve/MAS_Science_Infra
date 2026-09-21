import type { ReactNode } from 'react';
export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty-state"><p>{title}</p>{children && <div className="field-hint">{children}</div>}</div>;
}

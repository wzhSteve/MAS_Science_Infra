import { useState, type ReactNode } from 'react';

export function LazyDetails({ summary, children, className, open = false }: {
  summary: ReactNode; children: ReactNode; className?: string; open?: boolean;
}) {
  const [state, setState] = useState({ prop: open, expanded: open });
  const expanded = state.prop === open ? state.expanded : open;
  if (state.prop !== open) setState({ prop: open, expanded: open });
  return <details className={className} open={expanded || undefined} onToggle={event => {
    if (event.target !== event.currentTarget) return;
    const next = event.currentTarget.open;
    setState(previous => previous.prop === open && previous.expanded === next ? previous : { prop: open, expanded: next });
  }}>
    <summary>{summary}</summary>
    {expanded && children}
  </details>;
}

import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

export type Tone = 'neutral' | 'success' | 'warning' | 'danger' | 'info';
export function StatusBadge({ children, tone = 'neutral', className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return <span className={cn('status-badge', `status-badge--${tone}`, className)}>{children}</span>;
}

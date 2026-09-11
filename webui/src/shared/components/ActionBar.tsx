import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

export function ActionBar({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('action-bar', className)}>{children}</div>;
}

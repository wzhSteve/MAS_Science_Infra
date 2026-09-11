import type { ReactNode } from 'react';
import { CircleAlert, CircleCheck, Info } from 'lucide-react';
import { cn } from '../lib/cn';
import type { Tone } from './StatusBadge';

export type Notice = { message: string; tone: Tone };
export function InlineNotice({ children, tone = 'info', className }: { children: ReactNode; tone?: Tone; className?: string }) {
  const Icon = tone === 'danger' || tone === 'warning' ? CircleAlert : tone === 'success' ? CircleCheck : Info;
  return <div className={cn('inline-notice', `inline-notice--${tone}`, className)} role={tone === 'danger' ? 'alert' : 'status'}>
    <Icon size={15} aria-hidden="true" /><div>{children}</div>
  </div>;
}

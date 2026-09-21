import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

export function Section({ title, description, actions, children, className, id }: {
  title?: string; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; id?: string;
}) {
  return <section id={id} className={cn('ui-section', className)}>
    {(title || actions) && <div className="section-header"><div>{title && <h2>{title}</h2>}{description && <p className="section-description">{description}</p>}</div>{actions}</div>}
    {children}
  </section>;
}

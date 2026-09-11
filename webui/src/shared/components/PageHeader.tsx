import type { ReactNode } from 'react';

export function PageHeader({ title, description, actions, eyebrow }: {
  title: string; description?: ReactNode; actions?: ReactNode; eyebrow?: string;
}) {
  return <header className="page-header">
    <div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1>{description && <p className="page-description">{description}</p>}</div>
    {actions && <div className="page-header-actions">{actions}</div>}
  </header>;
}

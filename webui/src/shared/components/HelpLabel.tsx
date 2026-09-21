import type { ReactNode } from 'react';
import { CircleHelp } from 'lucide-react';
import { Tooltip } from '../ui/tooltip';

export function HelpLabel({ children, help }: { children: ReactNode; help: ReactNode }) {
  return <span className="help-label">
    <span>{children}</span>
    <Tooltip content={help}>
      <span className="help-label__trigger" tabIndex={0} aria-label="查看说明">
        <CircleHelp size={14} strokeWidth={1.8} aria-hidden="true" />
      </span>
    </Tooltip>
  </span>;
}

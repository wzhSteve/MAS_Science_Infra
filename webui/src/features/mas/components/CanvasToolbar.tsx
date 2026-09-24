import { memo, type ReactNode } from 'react';
import { Button, type ButtonProps } from '../../../shared/ui/button';
import { Tooltip } from '../../../shared/ui/tooltip';
import { cn } from '../../../shared/lib/cn';

export const CanvasToolbar = memo(function CanvasToolbar({ children, className }: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cn('canvas-toolbar', className)}>{children}</div>;
});

export function CanvasToolbarGroup({ children }: { children: ReactNode }) {
  return <div className="canvas-toolbar-group">{children}</div>;
}

export function CanvasToolbarDivider() {
  return <span className="canvas-toolbar-divider" aria-hidden="true" />;
}

export const CanvasToolButton = memo(function CanvasToolButton({
  label, children, className, ...props
}: ButtonProps & { label: string; children: ReactNode }) {
  return <Tooltip content={label}>
    <Button {...props} size="sm" variant="ghost" className={cn('canvas-tool', className)}
      aria-label={props['aria-label'] || label}>
      {children}
    </Button>
  </Tooltip>;
});

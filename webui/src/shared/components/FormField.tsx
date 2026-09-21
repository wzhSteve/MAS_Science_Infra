import { cloneElement, useId, type ReactElement, type ReactNode } from 'react';
import { cn } from '../lib/cn';

interface ControlProps { id?: string; 'aria-describedby'?: string; 'aria-invalid'?: boolean }

export function FormField({ label, hint, error, children, className }: {
  label: ReactNode; hint?: ReactNode; error?: string; children: ReactElement<ControlProps>; className?: string;
}) {
  const generatedId = useId();
  const id = children.props.id || generatedId;
  const descriptionId = `${id}-description`;
  return <div className={cn('form-field', className)}>
    <label htmlFor={id}>{label}</label>
    {cloneElement(children, { id, 'aria-describedby': cn(children.props['aria-describedby'], (hint || error) && descriptionId) || undefined, 'aria-invalid': error ? true : undefined })}
    {(hint || error) && <div id={descriptionId} className={error ? 'field-error' : 'field-hint'}>{error || hint}</div>}
  </div>;
}

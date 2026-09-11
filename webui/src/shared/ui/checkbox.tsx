import { forwardRef, type InputHTMLAttributes } from 'react';
import { cn } from '../lib/cn';

export const Checkbox = forwardRef<HTMLInputElement, Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>>(function Checkbox({ className, ...props }, ref) {
  return <input ref={ref} type="checkbox" className={cn('ui-checkbox', className)} {...props} />;
});

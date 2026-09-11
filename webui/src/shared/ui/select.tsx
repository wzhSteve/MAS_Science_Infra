import { forwardRef, type SelectHTMLAttributes } from 'react';
import { cn } from '../lib/cn';

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, ...props }, ref) {
  return <select ref={ref} className={cn('ui-input ui-select', className)} {...props} />;
});

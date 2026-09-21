import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { LoaderCircle } from 'lucide-react';
import { cn } from '../lib/cn';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', loading, disabled, className, children, type = 'button', ...props },
  ref,
) {
  return (
    <button {...props} ref={ref} type={type} className={cn('ui-button', `ui-button--${variant}`, `ui-button--${size}`, className)}
      disabled={disabled || loading} aria-busy={loading || undefined}>
      {loading ? <LoaderCircle size={14} className="animate-spin" aria-hidden="true" /> : null}
      {children}
    </button>
  );
});

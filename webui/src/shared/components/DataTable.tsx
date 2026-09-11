import type { TableHTMLAttributes } from 'react';
import { cn } from '../lib/cn';

export function DataTable({ className, children, ...props }: TableHTMLAttributes<HTMLTableElement>) {
  return <div className="table-scroll"><table className={cn('data-table', className)} {...props}>{children}</table></div>;
}

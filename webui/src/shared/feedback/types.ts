import type { ReactNode } from 'react';

export type NotificationTone = 'success' | 'info' | 'warning' | 'danger';

export interface NotificationAction {
  label: string;
  run: () => void;
}

export interface NotificationInput {
  title?: string;
  message: string;
  tone: NotificationTone;
  duration?: number | null;
  dedupeKey?: string;
  action?: NotificationAction;
}

export interface Notification extends NotificationInput {
  id: string;
  count: number;
  createdAt: number;
}

export interface ConfirmOptions {
  title: string;
  description: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: 'default' | 'danger';
}

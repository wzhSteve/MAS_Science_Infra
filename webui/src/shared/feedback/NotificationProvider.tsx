import { createContext, useMemo, useState, type ReactNode } from 'react';
import { Toast } from 'radix-ui';
import { createNotificationStore } from './notificationStore';
import type { NotificationInput } from './types';
import { NotificationViewport } from './NotificationViewport';

export interface NotifyApi {
  show: (input: NotificationInput) => string;
  success: (message: string, options?: Omit<NotificationInput, 'message' | 'tone'>) => string;
  info: (message: string, options?: Omit<NotificationInput, 'message' | 'tone'>) => string;
  warning: (message: string, options?: Omit<NotificationInput, 'message' | 'tone'>) => string;
  error: (message: string, options?: Omit<NotificationInput, 'message' | 'tone'>) => string;
  dismiss: (id: string) => void;
  dismissKey: (key: string) => void;
}

export const NotificationContext = createContext<NotifyApi | null>(null);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [store] = useState(createNotificationStore);
  const api = useMemo<NotifyApi>(() => ({
    show: input => store.push(input),
    success: (message, options) => store.push({ ...options, message, tone: 'success' }),
    info: (message, options) => store.push({ ...options, message, tone: 'info' }),
    warning: (message, options) => store.push({ ...options, message, tone: 'warning' }),
    error: (message, options) => store.push({ ...options, message, tone: 'danger' }),
    dismiss: store.dismiss,
    dismissKey: store.dismissKey,
  }), [store]);
  return <NotificationContext.Provider value={api}>
    <Toast.Provider swipeDirection="right">
      {children}
      <NotificationViewport store={store} />
    </Toast.Provider>
  </NotificationContext.Provider>;
}

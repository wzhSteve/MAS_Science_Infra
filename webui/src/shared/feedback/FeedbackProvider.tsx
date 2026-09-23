import type { ReactNode } from 'react';
import { ConfirmProvider } from './ConfirmProvider';
import { NotificationProvider } from './NotificationProvider';

export function FeedbackProvider({ children }: { children: ReactNode }) {
  return <NotificationProvider><ConfirmProvider>{children}</ConfirmProvider></NotificationProvider>;
}

import { useContext } from 'react';
import { NotificationContext } from './NotificationProvider';

export function useNotify() {
  const notify = useContext(NotificationContext);
  if (!notify) throw new Error('NotificationProvider is required');
  return notify;
}

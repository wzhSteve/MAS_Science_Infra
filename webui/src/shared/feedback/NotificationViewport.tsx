import { useEffect, useSyncExternalStore } from 'react';
import { Toast } from 'radix-ui';
import type { NotificationStore } from './notificationStore';
import { NotificationItem } from './NotificationItem';

export function NotificationViewport({ store }: { store: NotificationStore }) {
  const notifications = useSyncExternalStore(store.subscribe, store.getSnapshot);
  useEffect(() => {
    if (!notifications.length) return;
    const closeCurrent = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) store.dismiss(notifications[0].id);
    };
    window.addEventListener('keydown', closeCurrent);
    return () => window.removeEventListener('keydown', closeCurrent);
  }, [notifications, store]);
  return <Toast.Viewport className="feedback-viewport">
    {notifications.slice(0, 3).map(item => <NotificationItem key={item.id} item={item} store={store} />)}
  </Toast.Viewport>;
}

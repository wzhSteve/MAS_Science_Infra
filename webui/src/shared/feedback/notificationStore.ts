import type { Notification, NotificationInput } from './types';

const MAX_NOTIFICATIONS = 20;

export function createNotificationStore() {
  let sequence = 0;
  let snapshot: readonly Notification[] = [];
  const listeners = new Set<() => void>();
  const publish = (next: readonly Notification[]) => {
    snapshot = next;
    listeners.forEach(listener => listener());
  };
  return {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    getSnapshot: () => snapshot,
    push(input: NotificationInput) {
      if (input.dedupeKey) {
        const index = snapshot.findIndex(item => item.dedupeKey === input.dedupeKey);
        if (index >= 0) {
          const previous = snapshot[index];
          const updated = {
            ...previous, ...input, count: previous.count + 1, createdAt: Date.now(),
          };
          publish([updated, ...snapshot.slice(0, index), ...snapshot.slice(index + 1)]);
          return previous.id;
        }
      }
      const notification: Notification = {
        ...input,
        id: `notice-${Date.now()}-${++sequence}`,
        count: 1,
        createdAt: Date.now(),
      };
      publish([notification, ...snapshot].slice(0, MAX_NOTIFICATIONS));
      return notification.id;
    },
    dismiss(id: string) {
      const next = snapshot.filter(item => item.id !== id);
      if (next.length !== snapshot.length) publish(next);
    },
    dismissKey(dedupeKey: string) {
      const next = snapshot.filter(item => item.dedupeKey !== dedupeKey);
      if (next.length !== snapshot.length) publish(next);
    },
    clear() {
      if (snapshot.length) publish([]);
    },
  };
}

export type NotificationStore = ReturnType<typeof createNotificationStore>;

import { memo } from 'react';
import { Toast } from 'radix-ui';
import { CircleAlert, CircleCheck, Info, X } from 'lucide-react';
import type { Notification } from './types';
import type { NotificationStore } from './notificationStore';

const durations = { success: 3000, info: 4000, warning: 6000, danger: Infinity };

export const NotificationItem = memo(function NotificationItem({ item, store }: {
  item: Notification;
  store: NotificationStore;
}) {
  const Icon = item.tone === 'success' ? CircleCheck : item.tone === 'info' ? Info : CircleAlert;
  const duration = item.duration === null ? Infinity : item.duration
    ?? (item.tone === 'warning' && item.action ? Infinity : durations[item.tone]);
  return <Toast.Root className={`feedback-toast feedback-toast--${item.tone}`} duration={duration}
    type={item.tone === 'danger' ? 'foreground' : 'background'}
    onOpenChange={open => { if (!open) store.dismiss(item.id); }}
    onClick={event => {
      if (!(event.target instanceof Element) || !event.target.closest('button')) store.dismiss(item.id);
    }}>
    <Icon className="feedback-toast-icon" size={18} aria-hidden="true" />
    <div className="feedback-toast-content">
      {item.title && <Toast.Title className="feedback-toast-title">{item.title}</Toast.Title>}
      <Toast.Description className="feedback-toast-description">{item.message}</Toast.Description>
      {item.count > 1 && <span className="feedback-toast-count">重复 {item.count} 次</span>}
    </div>
    {item.action && <button type="button" className="feedback-toast-action" onClick={event => {
      event.stopPropagation();
      item.action?.run();
    }}>{item.action.label}</button>}
    <Toast.Close className="feedback-toast-close" aria-label="关闭通知" onClick={event => event.stopPropagation()}>
      <X size={15} />
    </Toast.Close>
  </Toast.Root>;
});

import { createContext, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { AlertDialog } from 'radix-ui';
import { Button } from '../ui/button';
import type { ConfirmOptions } from './types';

export type Confirm = (options: ConfirmOptions) => Promise<boolean>;
export const ConfirmContext = createContext<Confirm | null>(null);

interface Request {
  options: ConfirmOptions;
  resolve: (accepted: boolean) => void;
  trigger: HTMLElement | null;
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<Request | null>(null);
  const current = useRef<Request | null>(null);
  const confirm = useCallback<Confirm>(options => {
    if (current.current) return Promise.resolve(false);
    return new Promise(resolve => {
      const next = {
        options, resolve,
        trigger: document.activeElement instanceof HTMLElement ? document.activeElement : null,
      };
      current.current = next;
      setRequest(next);
    });
  }, []);
  const close = useCallback((accepted: boolean) => {
    current.current?.resolve(accepted);
    current.current = null;
    setRequest(null);
  }, []);
  const value = useMemo(() => confirm, [confirm]);
  useEffect(() => () => {
    current.current?.resolve(false);
    current.current = null;
  }, []);
  return <ConfirmContext.Provider value={value}>
    {children}
    {request && <AlertDialog.Root open onOpenChange={open => { if (!open) close(false); }}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="dialog-overlay" />
        <AlertDialog.Content className="dialog-content feedback-confirm" onCloseAutoFocus={event => {
          event.preventDefault();
          request.trigger?.focus();
        }}>
          <AlertDialog.Title className="dialog-title">{request.options.title}</AlertDialog.Title>
          <AlertDialog.Description className="dialog-description feedback-confirm-description">
            {request.options.description}
          </AlertDialog.Description>
          <div className="dialog-actions">
            <AlertDialog.Cancel asChild><Button onClick={() => close(false)}>
              {request.options.cancelLabel || '取消'}
            </Button></AlertDialog.Cancel>
            <AlertDialog.Action asChild><Button variant={request.options.tone === 'danger' ? 'danger' : 'primary'} onClick={() => close(true)}>
              {request.options.confirmLabel || '确认'}
            </Button></AlertDialog.Action>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>}
  </ConfirmContext.Provider>;
}

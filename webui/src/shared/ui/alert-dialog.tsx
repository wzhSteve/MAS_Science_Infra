import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertDialog } from 'radix-ui';
import { Button } from './button';

export function useTrainConfirmation() {
  const [open, setOpen] = useState(false);
  const resolveRef = useRef<((confirmed: boolean) => void) | null>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const confirm = useCallback(() => new Promise<boolean>(resolve => {
    if (resolveRef.current) { resolve(false); return; }
    triggerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    resolveRef.current = resolve;
    setOpen(true);
  }), []);
  const close = useCallback((accepted: boolean) => {
    resolveRef.current?.(accepted);
    resolveRef.current = null;
    setOpen(false);
  }, []);
  useEffect(() => () => { resolveRef.current?.(false); resolveRef.current = null; }, []);

  const dialog = <AlertDialog.Root open={open} onOpenChange={value => { if (!value) close(false); }}>
    <AlertDialog.Portal>
      <AlertDialog.Overlay className="dialog-overlay" />
      <AlertDialog.Content className="dialog-content" onCloseAutoFocus={event => {
        event.preventDefault();
        triggerRef.current?.focus();
      }}>
        <div className="eyebrow">TRAINING</div>
        <AlertDialog.Title className="dialog-title">启动训练</AlertDialog.Title>
        <AlertDialog.Description className="dialog-description">
          训练将停止正在运行的本地 LLM，并使用已保存的实验配置。确认启动？
        </AlertDialog.Description>
        <div className="dialog-actions">
          <AlertDialog.Cancel asChild><Button onClick={() => close(false)}>取消</Button></AlertDialog.Cancel>
          <AlertDialog.Action asChild><Button variant="primary" onClick={() => close(true)}>确认启动</Button></AlertDialog.Action>
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  </AlertDialog.Root>;
  return { confirm, dialog };
}

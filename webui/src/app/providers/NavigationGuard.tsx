import { createContext, useCallback, useContext, useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { AlertDialog } from 'radix-ui';
import { Button } from '../../shared/ui/button';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { errorMessage } from '../../shared/api/http';
import { createDraftRegistry, DraftRegistryContext, type DraftRegistry } from '../../shared/hooks/useUnsavedChanges';

interface LeaveRequest {
  resolve: (allow: boolean) => void;
  trigger: HTMLElement | null;
}

const NavigationGuardContext = createContext<(() => Promise<boolean>) | null>(null);

function LeaveDialog({ registry, request, onClose }: {
  registry: DraftRegistry; request: LeaveRequest; onClose: (allow: boolean) => void;
}) {
  const drafts = useSyncExternalStore(registry.subscribe, registry.getSnapshot);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = drafts.some(draft => draft.busy);
  const dirty = drafts.filter(draft => draft.dirty);
  const conflicts = dirty.filter((draft, index) => dirty.findIndex(other => other.resource === draft.resource) !== index);
  const canSave = dirty.some(draft => draft.canSave) && conflicts.length === 0;
  const hasTemporaryInput = dirty.some(draft => !draft.canSave);
  const save = async () => {
    if (saving || busy || !canSave) return;
    setSaving(true);
    setError(null);
    try {
      for (const draft of dirty.filter(item => item.canSave)) {
        if (!(await registry.save(draft.id))) {
          setError(`“${draft.label}”未能保存，请取消切换并检查对应面板；已经保存的内容会保留。`);
          return;
        }
      }
      onClose(true);
    } catch (reason) {
      setError(`保存失败，仍保留当前实验：${errorMessage(reason)}`);
    } finally {
      setSaving(false);
    }
  };

  return <AlertDialog.Root open onOpenChange={open => { if (!open && !saving) onClose(false); }}>
    <AlertDialog.Portal>
      <AlertDialog.Overlay className="dialog-overlay" />
      <AlertDialog.Content className="dialog-content" onCloseAutoFocus={event => {
        event.preventDefault();
        request.trigger?.focus();
      }}>
        <AlertDialog.Title className="dialog-title">{busy ? '当前实验还有操作未完成' : '切换实验前保存修改？'}</AlertDialog.Title>
        <AlertDialog.Description className="dialog-description">
          {busy ? '请等待保存或执行请求返回后再切换。切换面板、返回首页不会取消任务。'
            : '切换实验会关闭当前编辑会话。已保存的运行记录不会删除；未提交的调试输入和界面状态不会带到另一个实验。'}
        </AlertDialog.Description>
        {drafts.length > 0 && <ul className="navigation-draft-list">
          {drafts.map(draft => <li key={draft.id}>{draft.label} · {draft.busy ? '操作中' : '未保存'}</li>)}
        </ul>}
        {conflicts.length > 0 && <InlineNotice tone="warning">
          同一份配置在多个面板中都有未保存修改。请取消切换，先在对应面板确认要保留的版本，避免相互覆盖。
        </InlineNotice>}
        {hasTemporaryInput && <p className="field-hint">
          调试输入等临时内容没有独立保存接口。“保存配置后切换”只保存可持久化的配置，临时输入仍会被放弃；取消切换可保留全部内容。
        </p>}
        {error && <InlineNotice tone="danger">{error}</InlineNotice>}
        <div className="dialog-actions">
          <Button disabled={saving} onClick={() => onClose(false)}>取消切换</Button>
          <Button disabled={busy || saving} onClick={() => onClose(true)}>{dirty.length ? '放弃修改并切换' : '继续切换'}</Button>
          {canSave && <Button variant="primary" loading={saving} disabled={busy || saving} onClick={() => void save()}>{hasTemporaryInput ? '保存配置后切换' : '保存后切换'}</Button>}
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  </AlertDialog.Root>;
}

export function NavigationGuardProvider({ children }: { children: ReactNode }) {
  const [registry] = useState(createDraftRegistry);
  const [request, setRequest] = useState<LeaveRequest | null>(null);
  const pending = useRef<LeaveRequest | null>(null);
  const confirmLeave = useCallback(() => {
    if (pending.current) return Promise.resolve(false);
    if (registry.getSnapshot().length === 0) return Promise.resolve(true);
    return new Promise<boolean>(resolve => {
      const next = { resolve, trigger: document.activeElement instanceof HTMLElement ? document.activeElement : null };
      pending.current = next;
      setRequest(next);
    });
  }, [registry]);
  const close = useCallback((allow: boolean) => {
    pending.current?.resolve(allow);
    pending.current = null;
    setRequest(null);
  }, []);
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (registry.getSnapshot().length === 0) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      window.removeEventListener('beforeunload', beforeUnload);
      pending.current?.resolve(false);
    };
  }, [registry]);
  return <DraftRegistryContext.Provider value={registry}>
    <NavigationGuardContext.Provider value={confirmLeave}>
      {children}
      {request && <LeaveDialog registry={registry} request={request} onClose={close} />}
    </NavigationGuardContext.Provider>
  </DraftRegistryContext.Provider>;
}

export function useConfirmExperimentLeave() {
  const confirm = useContext(NavigationGuardContext);
  if (!confirm) throw new Error('NavigationGuardProvider is required');
  return confirm;
}

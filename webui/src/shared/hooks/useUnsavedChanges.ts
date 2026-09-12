import { createContext, useContext, useLayoutEffect, useRef } from 'react';

export interface DraftRegistration {
  label: string;
  resource: string;
  dirty: boolean;
  busy?: boolean;
  save?: () => Promise<boolean>;
}

export interface DraftStatus {
  id: string;
  label: string;
  resource: string;
  dirty: boolean;
  busy: boolean;
  canSave: boolean;
}

export function createDraftRegistry() {
  const entries = new Map<string, () => DraftRegistration>();
  const listeners = new Set<() => void>();
  let snapshot: readonly DraftStatus[] = [];

  const publish = () => {
    const next: DraftStatus[] = [];
    for (const [id, read] of entries) {
      const value = read();
      if (value.dirty || value.busy) next.push({
        id, label: value.label, resource: value.resource, dirty: value.dirty,
        busy: Boolean(value.busy), canSave: Boolean(value.save),
      });
    }
    if (next.length === snapshot.length && next.every((item, index) => {
      const previous = snapshot[index];
      return item.id === previous.id && item.label === previous.label && item.resource === previous.resource
        && item.dirty === previous.dirty && item.busy === previous.busy && item.canSave === previous.canSave;
    })) return;
    snapshot = next;
    listeners.forEach(listener => listener());
  };

  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    register: (id: string, read: () => DraftRegistration) => {
      entries.set(id, read);
      publish();
      return () => {
        if (entries.get(id) === read) { entries.delete(id); publish(); }
      };
    },
    publish,
    save: async (id: string) => {
      const entry = entries.get(id)?.();
      if (!entry || !entry.dirty) return true;
      if (entry.busy || !entry.save) return false;
      return entry.save();
    },
  };
}

export type DraftRegistry = ReturnType<typeof createDraftRegistry>;
export const DraftRegistryContext = createContext<DraftRegistry | null>(null);

export function useUnsavedChanges(id: string, registration: DraftRegistration) {
  const registry = useContext(DraftRegistryContext);
  const current = useRef(registration);
  useLayoutEffect(() => {
    current.current = registration;
    registry?.publish();
  });
  useLayoutEffect(() => registry?.register(id, () => current.current), [registry, id]);
}

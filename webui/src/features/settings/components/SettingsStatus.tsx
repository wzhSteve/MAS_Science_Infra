import { memo, useContext, useSyncExternalStore } from 'react';
import { DraftRegistryContext, type DraftStatus } from '../../../shared/hooks/useUnsavedChanges';

const EMPTY: readonly DraftStatus[] = [];
const emptySnapshot = () => EMPTY;
const emptySubscribe = () => () => {};
const RESOURCES = new Set(['llm', 'rl', 'harness']);

export function useSettingsStatus() {
  const registry = useContext(DraftRegistryContext);
  return useSyncExternalStore(registry?.subscribe ?? emptySubscribe, registry?.getSnapshot ?? emptySnapshot);
}

export const SettingsStatus = memo(function SettingsStatus({ resource }: { resource?: string }) {
  const entries = useSettingsStatus().filter(entry => resource ? entry.resource === resource : RESOURCES.has(entry.resource));
  const pending = entries.some(entry => entry.busy);
  const changed = new Set(entries.filter(entry => entry.dirty).map(entry => entry.resource));
  if (!pending && changed.size === 0) return null;
  return <span className="experiment-settings-status" role="status">
    {pending ? '设置操作中' : resource ? '未保存' : `${changed.size} 项设置未保存`}
  </span>;
});

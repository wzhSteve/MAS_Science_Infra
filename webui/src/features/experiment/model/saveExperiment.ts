import type { DraftRegistry, DraftStatus } from '../../../shared/hooks/useUnsavedChanges';

const RESOURCES = new Set(['meta', 'workflow', 'llm', 'rl', 'harness', 'model-binding-inference', 'model-binding-training']);
export const isExperimentDraft = (draft: DraftStatus) => RESOURCES.has(draft.resource);
const saveOrder = (resource: string) => resource === 'workflow' ? 0 : resource === 'rl' ? 1 : resource.startsWith('model-binding') ? 3 : 2;

export async function saveExperimentDrafts(registry: DraftRegistry, reload: () => Promise<void>) {
  const drafts = registry.getSnapshot().filter(isExperimentDraft);
  if (drafts.some(item => item.busy)) throw new Error('配置操作尚未完成，请稍后保存。');
  const dirty = drafts.filter(item => item.dirty).sort((a, b) => saveOrder(a.resource) - saveOrder(b.resource));
  if (new Set(dirty.map(item => item.resource)).size !== dirty.length) {
    throw new Error('同一份配置存在多个编辑来源，请先确认要保留的版本。');
  }
  const saved: string[] = [];
  for (const item of dirty) {
    if (!item.canSave || !(await registry.save(item.id))) {
      throw new Error(`“${item.label}”未能保存。${saved.length ? `已保存：${saved.join('、')}；这些修改不会撤销。` : ''}请检查对应字段。`);
    }
    saved.push(item.label);
    // Publish committed configuration before saving dependent model bindings.
    await reload();
  }
  if (registry.getSnapshot().filter(isExperimentDraft).some(item => item.dirty || item.busy)) {
    throw new Error('保存期间产生了新修改，请再次保存。');
  }
}

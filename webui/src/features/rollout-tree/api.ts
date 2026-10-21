import { ApiError, experimentQuery } from '../../shared/api/http';
import type { NodeDetail, TreeDetail, TreeList } from './types';

// Scoped to one page session; no cross-experiment response cache.
export function createTreeClient(experimentId: string, runId: string) {
  const base = `/api/rl/runs/${encodeURIComponent(runId)}/rollout-trees`;
  const cache = new Map<string, { etag: string; value: unknown }>();
  async function read<T extends { experiment_id: string; run_id: string }>(path: string, signal: AbortSignal): Promise<T> {
    const saved = cache.get(path);
    const response = await fetch(path, { signal, headers: saved ? { 'If-None-Match': saved.etag } : {} });
    if (response.status === 304 && saved) return saved.value as T;
    if (!response.ok) {
      const text = await response.text();
      let message = text || response.statusText;
      try {
        const body: unknown = JSON.parse(text);
        if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') message = body.detail;
      } catch { /* Preserve non-JSON server error text. */ }
      throw new ApiError(message, response.status);
    }
    const value: T = await response.json();
    if (value.experiment_id !== experimentId || value.run_id !== runId) throw new Error('响应不属于当前训练运行。');
    const etag = response.headers.get('etag');
    if (etag) {
      cache.delete(path);
      cache.set(path, { etag, value });
      if (cache.size > 16) cache.delete(cache.keys().next().value!);
    }
    return value;
  }
  return {
    node: async (treeId: string, nodeId: string, signal: AbortSignal) => {
      const result = await read<NodeDetail>(`${base}/${encodeURIComponent(treeId)}/nodes/${encodeURIComponent(nodeId)}?${experimentQuery(experimentId)}`, signal);
      if (result.tree_id !== treeId || result.node.node_id !== nodeId) throw new Error('响应不属于当前执行节点。');
      return result;
    },
    list: (offset: number, mode: string, signal: AbortSignal) =>
      read<TreeList>(`${base}?${experimentQuery(experimentId)}&offset=${offset}&limit=20${mode ? `&mode=${mode}` : ''}`, signal),
    detail: async (treeId: string, signal: AbortSignal, cursor?: string, planOffset = 0, nodeId?: string) => {
      const params = new URLSearchParams({ experiment_id: experimentId, limit: '500', plan_offset: String(planOffset) });
      if (cursor) params.set('cursor', cursor);
      if (nodeId) params.set('node_id', nodeId);
      const result = await read<TreeDetail>(`${base}/${encodeURIComponent(treeId)}?${params}`, signal);
      if (result.tree.tree_id !== treeId) throw new Error('响应不属于当前题目记录。');
      return result;
    },
  };
}

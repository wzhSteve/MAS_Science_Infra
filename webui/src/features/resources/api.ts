import { experimentPath, request } from '../../shared/api/http';
import type { HealthResponse } from '../../shared/api/types';

export type ModelResourceType = 'inference' | 'training';
export interface ModelResourceConfig {
  kind?: 'api' | 'local';
  model?: string;
  base_url?: string;
  model_path?: string;
  port?: number;
  gpu_memory_utilization?: number;
}
export interface ModelResource {
  id: string;
  name: string;
  type: ModelResourceType;
  revision: number;
  config: ModelResourceConfig;
  credential_mode: 'saved' | 'service' | 'none';
  api_key_set: boolean;
  created_at: string;
  updated_at: string;
}
export interface ResourceWrite {
  name: string;
  type: ModelResourceType;
  config: ModelResourceConfig;
  credential_mode: ModelResource['credential_mode'];
  api_key?: string;
  clear_api_key?: boolean;
}
export interface BindingSelection {
  resource_id: string | null;
  resource: ModelResource | null;
  error?: string;
}
export interface ModelBindings {
  revision: number;
  inference: BindingSelection;
  training: BindingSelection;
}
export interface ResourceList {
  items: ModelResource[];
  total: number;
  offset: number;
  limit: number;
}
export interface ResourceDetail extends ModelResource {
  references: Array<{ experiment_id: string; purpose: ModelResourceType }>;
}
export interface LocalModelCandidate {
  path: string;
  name: string;
  architectures: string[];
  model_type?: string | null;
  torch_dtype?: string | null;
  registered_resource_id?: string | null;
}

export interface DatasetResource {
  id: string;
  name: string;
  path: string;
}
export interface DatasetCatalog {
  revision: number;
  items: DatasetResource[];
}
export const datasetResourcesApi = {
  list: (signal?: AbortSignal) => request<DatasetCatalog>('/api/dataset-resources', { signal }),
  save: (body: { name: string; path: string; revision: number }, id?: string) =>
    request<DatasetCatalog>(`/api/dataset-resources${id ? `/${encodeURIComponent(id)}` : ''}`, {
      method: id ? 'PUT' : 'POST', body: JSON.stringify(body),
    }),
};

const path = (id: string) => `/api/model-resources/${encodeURIComponent(id)}`;
const bindingPath = (id: string) => `${experimentPath(id)}/model-bindings`;
export const modelResourcesApi = {
  list: (options: { type?: ModelResourceType; offset?: number; limit?: number; q?: string } = {}, signal?: AbortSignal) => {
    const query = new URLSearchParams({ offset: String(options.offset || 0), limit: String(options.limit || 20) });
    if (options.type) query.set('type', options.type);
    if (options.q) query.set('q', options.q);
    return request<ResourceList>(`/api/model-resources?${query}`, { signal });
  },
  get: (id: string, signal?: AbortSignal) => request<ResourceDetail>(path(id), { signal }),
  create: (body: ResourceWrite) => request<ModelResource>('/api/model-resources', { method: 'POST', body: JSON.stringify(body) }),
  update: (id: string, revision: number, body: ResourceWrite) =>
    request<ModelResource>(path(id), { method: 'PUT', body: JSON.stringify({ ...body, revision }) }),
  remove: (id: string, revision: number) => request<{ ok: boolean }>(`${path(id)}?revision=${revision}`, { method: 'DELETE' }),
  probe: (id: string, revision: number) => request<HealthResponse>(`${path(id)}/probe`, { method: 'POST', body: JSON.stringify({ revision }) }),
  bindings: (id: string, signal?: AbortSignal) => request<ModelBindings>(bindingPath(id), { signal }),
  bind: (id: string, revision: number, purpose: ModelResourceType, resource_id: string | null) =>
    request<ModelBindings>(bindingPath(id), { method: 'PUT', body: JSON.stringify({ revision, purpose, resource_id }) }),
  registerLocal: (id: string, body: { revision: number; name: string; model_path: string }) =>
    request<{ resource: ModelResource; bindings: ModelBindings }>(`${bindingPath(id)}/register-local`, {
      method: 'POST', body: JSON.stringify(body),
    }),
  migrate: (id: string, body: { revision: number; purpose: ModelResourceType; name: string; credential_mode: 'copy' | 'service' }) =>
    request<{ resource: ModelResource; bindings: ModelBindings }>(`${bindingPath(id)}/migrate`, { method: 'POST', body: JSON.stringify(body) }),
  discoverLocal: (signal?: AbortSignal) =>
    request<{ items: LocalModelCandidate[] }>('/api/model-resources/discover-local', { signal }),
};

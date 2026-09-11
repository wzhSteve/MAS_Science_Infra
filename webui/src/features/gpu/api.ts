import { request } from '../../shared/api/http';
import type { GpuResponse } from '../../shared/api/types';

export const gpuApi = {
  gpus: (signal?: AbortSignal) => request<GpuResponse>('/api/gpus', { signal }),
};

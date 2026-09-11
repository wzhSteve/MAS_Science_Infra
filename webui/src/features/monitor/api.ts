import { request } from '../../shared/api/http';
import type { MonitorResponse } from '../../shared/api/types';

export const monitorApi = {
  monitor: (id: string, signal?: AbortSignal) =>
    request<MonitorResponse>(`/api/monitor/${encodeURIComponent(id)}`, { signal }),
};

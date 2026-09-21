import { experimentQuery, request } from '../../shared/api/http';
import type { Hypothesis } from '../../shared/api/types';

export const harnessApi = {
  diagnose: (id: string) =>
    request<{ hypotheses?: Hypothesis[]; n: number; path?: string }>(
      `/api/harness/diagnose?${experimentQuery(id)}`, { method: 'POST', body: '{}' }),
};

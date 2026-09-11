export class ApiError extends Error {
  constructor(message: string, public readonly status?: number, public readonly detail?: unknown) {
    super(message);
    this.name = 'ApiError';
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    let detail: unknown = text || response.statusText;
    try {
      const body: unknown = JSON.parse(text);
      if (body && typeof body === 'object' && 'detail' in body) detail = body.detail;
    } catch {
      // Non-JSON error pages still carry useful server diagnostics.
    }
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), response.status, detail);
  }
  return response.json() as Promise<T>;
}

export const experimentQuery = (id: string) => `experiment_id=${encodeURIComponent(id)}`;
export const experimentPath = (id: string) => `/api/experiments/${encodeURIComponent(id)}`;

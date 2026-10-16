import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { UserProjectRuntimePanel } from './GraphPalette';
import { ExecutorToolPool } from './NodeInspector';

const { userProjectRuntime, putUserProjectRuntime } = vi.hoisted(() => ({
  userProjectRuntime: vi.fn(),
  putUserProjectRuntime: vi.fn(),
}));

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>();
  return {
    ...actual,
    masApi: {
      ...actual.masApi,
      userProjectRuntime,
      putUserProjectRuntime,
    },
  };
});

afterEach(() => {
  cleanup();
  userProjectRuntime.mockReset();
  putUserProjectRuntime.mockReset();
});

describe('Executor tool pool checkboxes', () => {
  it('UserProjectRuntimePanel renders API allowed_tools and never Google', async () => {
    userProjectRuntime.mockResolvedValue({
      enabled_tools: ['Bing_Search_Tool', 'Web_Fetch_Tool'],
      allowed_tools: [
        'Base_Generator_Tool',
        'Python_Coder_Tool',
        'Wikipedia_Search_Tool',
        'Bing_Search_Tool',
        'Web_Fetch_Tool',
        'Google_Search_Tool',
      ],
    });
    render(<UserProjectRuntimePanel projectId="agentflow" family="agentflow" wraps="AgentFlow" title="AgentFlow" onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText('Web_Fetch_Tool')).toBeTruthy());
    expect(screen.getByText('Bing_Search_Tool')).toBeTruthy();
    expect(screen.queryByText('Google_Search_Tool')).toBeNull();
  });

  it('falls back to the five canonical tools when allowed_tools is missing', async () => {
    userProjectRuntime.mockResolvedValue({ enabled_tools: ['Bing_Search_Tool'] });
    render(<ExecutorToolPool projectId="epc-aw-main" />);
    await waitFor(() => expect(screen.getByText('Web_Fetch_Tool')).toBeTruthy());
    expect(screen.getByText('Base_Generator_Tool')).toBeTruthy();
    expect(screen.queryByText('Google_Search_Tool')).toBeNull();
  });
});

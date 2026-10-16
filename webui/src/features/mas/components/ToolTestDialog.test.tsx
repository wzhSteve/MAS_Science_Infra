import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Palette } from '../../../shared/api/types';
import { defaultToolTestAgentId, toolTestNeedsUrl, ToolTestDialog } from './ToolTestDialog';

const { toolAgentTest } = vi.hoisted(() => ({ toolAgentTest: vi.fn() }));

vi.mock('../api', () => ({
  masApi: { toolAgentTest },
}));

afterEach(() => {
  cleanup();
  toolAgentTest.mockReset();
});

const palette: Palette = {
  tool_agents: [
    { id: 'google_search', backend: 'llm', llm_required: true },
    { id: 'wikipedia_search', backend: 'llm', llm_required: true },
    { id: 'bing_search', backend: 'llm', llm_required: true },
    { id: 'web_fetch', backend: 'llm', llm_required: true },
    { id: 'web_search', backend: 'llm', llm_required: true },
  ],
};

describe('defaultToolTestAgentId', () => {
  it('prefers bing_search and never defaults to google_search', () => {
    expect(defaultToolTestAgentId(palette.tool_agents!)).toBe('bing_search');
    expect(defaultToolTestAgentId([
      { id: 'google_search' },
      { id: 'wikipedia_search' },
    ])).toBe('wikipedia_search');
    expect(defaultToolTestAgentId([])).toBe('');
  });
});

describe('toolTestNeedsUrl', () => {
  it('is true for web_fetch and web_search', () => {
    expect(toolTestNeedsUrl('web_fetch')).toBe(true);
    expect(toolTestNeedsUrl('web_search')).toBe(true);
    expect(toolTestNeedsUrl('bing_search')).toBe(false);
  });
});

describe('ToolTestDialog', () => {
  it('defaults to bing_search and sends { url } for web_fetch', async () => {
    const user = userEvent.setup();
    toolAgentTest.mockResolvedValue({
      requested_tier: 'lite',
      resolved_tier: 'lite',
      output: 'ok',
      elapsed_ms: 1,
      ok: true,
    });
    render(<ToolTestDialog open onOpenChange={() => {}} palette={palette} />);
    const select = screen.getByLabelText('选择工具代理') as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('bing_search'));
    expect(select.value).not.toBe('google_search');

    await user.selectOptions(select, 'web_fetch');
    expect(screen.getByLabelText('测试 URL')).toBeTruthy();
    expect(screen.queryByLabelText('测试查询 query')).toBeNull();

    await user.type(screen.getByLabelText('测试 URL'), 'https://example.com/article');
    await user.click(screen.getByRole('button', { name: '运行' }));
    await waitFor(() => expect(toolAgentTest).toHaveBeenCalled());
    expect(toolAgentTest).toHaveBeenCalledWith({
      agent_id: 'web_fetch',
      tier: 'lite',
      args: { url: 'https://example.com/article' },
    });
  });
});

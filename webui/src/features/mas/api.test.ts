import { describe, expect, it } from 'vitest';
import { CANONICAL_ALLOWED_TOOLS, resolveAllowedTools, resolveEnabledTools } from './api';

describe('resolveAllowedTools', () => {
  it('falls back to the five canonical tools and never includes Google', () => {
    expect(resolveAllowedTools()).toEqual([...CANONICAL_ALLOWED_TOOLS]);
    expect(resolveAllowedTools([])).toEqual([...CANONICAL_ALLOWED_TOOLS]);
    expect(resolveAllowedTools(['Google_Search_Tool', 'Ground_Google_Search_Tool'])).toEqual([...CANONICAL_ALLOWED_TOOLS]);
    expect(resolveAllowedTools()).toContain('Web_Fetch_Tool');
    expect(resolveAllowedTools()).not.toContain('Google_Search_Tool');
  });

  it('uses the API catalog after dropping Google', () => {
    expect(resolveAllowedTools(['Bing_Search_Tool', 'Google_Search_Tool', 'Web_Fetch_Tool'])).toEqual([
      'Bing_Search_Tool',
      'Web_Fetch_Tool',
    ]);
  });
});

describe('resolveEnabledTools', () => {
  it('drops Google and falls back to the catalog when empty', () => {
    expect(resolveEnabledTools(['Google_Search_Tool'], ['Web_Fetch_Tool'])).toEqual(['Web_Fetch_Tool']);
    expect(resolveEnabledTools(['Bing_Search_Tool', 'Google_Search_Tool'])).toEqual(['Bing_Search_Tool']);
  });
});

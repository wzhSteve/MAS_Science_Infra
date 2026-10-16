import { memo, useEffect, useId, useMemo, useState } from 'react';
import { AlertDialog } from 'radix-ui';
import { masApi, type ToolAgentTestRequest, type ToolAgentTestResponse } from '../api';
import type { Palette } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { InlineNotice } from '../../../shared/components/InlineNotice';

type TierValue = 'lite' | 'pro';

export function defaultToolTestAgentId(agents: Array<{ id: string }>): string {
  if (!agents.length) return '';
  const bing = agents.find((agent) => agent.id === 'bing_search');
  if (bing) return bing.id;
  const firstNonGoogle = agents.find((agent) => agent.id !== 'google_search');
  return firstNonGoogle?.id || agents[0].id;
}

export function toolTestNeedsUrl(agentId: string): boolean {
  return agentId === 'web_fetch' || agentId === 'web_search';
}

export const ToolTestDialog = memo(function ToolTestDialog({ open, onOpenChange, palette }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  palette: Palette;
}) {
  const radioName = useId();
  const agents = useMemo(() => palette.tool_agents || [], [palette.tool_agents]);
  const [agentId, setAgentId] = useState('');
  const [tier, setTier] = useState<TierValue>('lite');
  const [fetchTopN, setFetchTopN] = useState(3);
  const [query, setQuery] = useState('');
  const [url, setUrl] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ToolAgentTestResponse | null>(null);

  const selectedAgent = agents.find((agent) => agent.id === agentId);
  const hasTiers = Boolean(selectedAgent?.has_tiers && (selectedAgent?.tiers || []).length);
  const needsUrl = toolTestNeedsUrl(agentId);

  // 每次打开重置表单：默认 bing_search（否则第一个非 google 项）；tier 重置为 lite。
  useEffect(() => {
    if (!open) return;
    setError('');
    setResult(null);
    setQuery('');
    setUrl('');
    setTier('lite');
    setAgentId(defaultToolTestAgentId(agents));
    // 仅在打开瞬间依据当时的节点库重置，palette 后续变化交给下方校正逻辑。
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // 节点库晚于对话框加载（或被刷新）时校正选择：保留仍有效的当前项，否则回退默认。
  useEffect(() => {
    if (!agents.length) return;
    setAgentId((current) => agents.some((agent) => agent.id === current)
      ? current
      : defaultToolTestAgentId(agents));
  }, [agents]);

  // 切换工具或 tier 变化时同步默认抓取条数。
  useEffect(() => {
    if (!hasTiers || tier !== 'pro') return;
    setFetchTopN(Math.max(1, Math.min(Math.round(selectedAgent?.default_fetch_n ?? 3) || 3, 10)));
  }, [agentId, tier, hasTiers, selectedAgent]);

  const run = async () => {
    setPending(true);
    setError('');
    setResult(null);
    try {
      const args: Record<string, unknown> = needsUrl ? { url } : { query };
      const body: ToolAgentTestRequest = { agent_id: agentId, tier: hasTiers ? tier : 'lite', args };
      if (hasTiers && tier === 'pro') body.fetch_top_n = Math.max(1, Math.min(Math.round(fetchTopN) || 3, 10));
      const response = await masApi.toolAgentTest(body);
      setResult(response);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPending(false);
    }
  };

  const tierTone = result?.resolved_tier === 'pro' ? 'warning' : 'info';

  return <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
    <AlertDialog.Portal>
      <AlertDialog.Overlay className="dialog-overlay" />
      <AlertDialog.Content className="dialog-content tool-test-dialog">
        <AlertDialog.Title className="dialog-title">测试 Tool Agent</AlertDialog.Title>
        <AlertDialog.Description className="dialog-description">
          选择一个工具代理，配置权重与查询，直接调用后端验证输出。仅用于调试，不会改动工作流。
        </AlertDialog.Description>
        <label className="tool-test-field">工具代理
          <Select aria-label="选择工具代理" value={agentId} disabled={!agents.length}
            onChange={(event) => setAgentId(event.target.value)}>
            {!agents.length && <option value="">节点库为空</option>}
            {agents.map((agent) => <option key={agent.id} value={agent.id}>
              {agent.id}{agent.description ? ` · ${agent.description}` : ''}
            </option>)}
          </Select>
          {hasTiers && <span className="field-hint">支持 lite / pro 两档权重</span>}
        </label>
        <fieldset className="tool-test-field tool-test-tier" disabled={!hasTiers}>
          <legend>Tier</legend>
          {(['lite', 'pro'] as const).map((option) => (
            <label key={option} className={`tool-test-tier-option${tier === option ? ' is-selected' : ''}${!hasTiers && option === 'pro' ? ' is-disabled' : ''}`}>
              <input type="radio" name={radioName} value={option}
                checked={hasTiers ? tier === option : option === 'lite'}
                disabled={!hasTiers}
                onChange={() => setTier(option)} />
              <span>{option}</span>
            </label>
          ))}
          <span className="field-hint">{hasTiers ? 'lite 仅返回标题与链接；pro 会抓取前 N 条网页做 RAG 摘要。' : '当前工具不支持分档，固定使用 lite。'}</span>
        </fieldset>
        {hasTiers && tier === 'pro' && <label className="tool-test-field">抓取条数（fetch_top_n）
          <Input type="number" min={1} max={10} value={String(fetchTopN)}
            aria-label="pro 抓取条数 fetch_top_n"
            onChange={(event) => setFetchTopN(Number(event.target.value) || 3)} />
          <span className="field-hint">对前 N 条结果抓取网页并生成 RAG 摘要，范围 1–10。</span>
        </label>}
        {needsUrl ? <label className="tool-test-field">URL
          <Input value={url} placeholder="例如：https://example.com/article" aria-label="测试 URL"
            onChange={(event) => setUrl(event.target.value)} />
        </label> : <label className="tool-test-field">查询（query）
          <Input value={query} placeholder="例如：量子纠错码最新进展" aria-label="测试查询 query"
            onChange={(event) => setQuery(event.target.value)} />
        </label>}
        {error && <InlineNotice tone="danger">{error}</InlineNotice>}
        {result && <div className="tool-test-result">
          <div className="tool-test-result-meta">
            <StatusBadge tone={tierTone}>tier: {result.resolved_tier}</StatusBadge>
            <StatusBadge tone="neutral">耗时 {result.elapsed_ms} ms</StatusBadge>
          </div>
          <pre className="tool-test-output mono">{result.output}</pre>
        </div>}
        <div className="dialog-actions">
          <AlertDialog.Cancel asChild><Button disabled={pending}>关闭</Button></AlertDialog.Cancel>
          <Button variant="primary" loading={pending}
            disabled={pending || !agentId || !(needsUrl ? url.trim() : query.trim())}
            onClick={() => void run()}>运行</Button>
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  </AlertDialog.Root>;
});

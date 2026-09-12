import { memo, useState } from 'react';
import type { ModelReadinessState } from '../model/useModelReadiness';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { DownloadText } from '../../../shared/components/DownloadText';

const kinds: Record<string, string> = { api: '远程 API', local: '本地模型服务', rl_endpoint: '训练注入' };
const credentials = { experiment: '实验密钥', service: '服务默认密钥', none: '未提供密钥' };

export const ModelReadiness = memo(function ModelReadiness({ readiness, onConfigureModel, includeData = false }: {
  readiness: ModelReadinessState; onConfigureModel: () => void; includeData?: boolean;
}) {
  const { data, error, loading, refresh } = readiness;
  const [expanded, setExpanded] = useState(false);
  return <section className="mas-model-readiness" aria-label="模型就绪状态" aria-busy={loading}>
    <div className="mas-model-summary">
      <strong>{data?.model.model || '尚未确认模型'}</strong>
      {data && <span>{kinds[data.model.kind] || data.model.kind}</span>}
      <StatusBadge tone={error ? 'danger' : loading ? 'neutral' : data?.ready ? 'success' : 'warning'}>
        {error ? '检查失败' : loading ? '检查中' : data?.ready ? '可尝试调试' : '需要准备'}
      </StatusBadge>
      <Button size="sm" variant="ghost" onClick={onConfigureModel}>配置模型</Button>
      <Button size="sm" variant="ghost" disabled={loading} onClick={refresh}>{error ? '重试检查' : '刷新'}</Button>
      <Button size="sm" variant="ghost" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>
        {expanded ? '收起详情' : '连接与诊断'}
      </Button>
    </div>
    {error && <InlineNotice tone="danger">
      暂时无法确认调试条件，请检查 Control 服务连接后重试。
      <DownloadText text={error} filename="readiness-error.txt" label="下载检查错误" />
    </InlineNotice>}
    {!error && data && !data.ready && <InlineNotice tone="warning">
      暂不能真实调试：{data.blocking_issues.some(issue => issue.code === 'live_dependencies')
        ? '服务端推理依赖需要准备。' : '模型配置尚未满足执行条件。'}
      {!expanded && <Button size="sm" variant="ghost" onClick={() => setExpanded(true)}>查看原因</Button>}
    </InlineNotice>}
    {expanded && data && <div className="mas-model-details">
      <dl>
        <div><dt>端点</dt><dd>{data.model.base_url || '未配置'}</dd></div>
        <div><dt>凭据来源</dt><dd>{credentials[data.model.credential_source]}（不回显）</dd></div>
        <div><dt>检查时间</dt><dd>{new Date(data.checked_at).toLocaleString()}</dd></div>
      </dl>
      {data.blocking_issues.map((issue, index) => <div className="mas-model-dependency" key={`${issue.code}-${index}`}>
        <p>{issue.message}</p>{issue.hint && <p className="field-hint">{issue.hint}</p>}
      </div>)}
      {data.warnings.filter(issue => includeData || issue.code !== 'parquet_dependencies').map((issue, index) =>
        <p className="field-hint" key={`${issue.code}-${index}`}>{issue.message}{issue.hint && <> {issue.hint}</>}</p>)}
      {data.probe ? <p className="field-hint">最近端点探测：{data.probe.message}</p>
        : <p className="field-hint">尚未探测端点，不代表连接失败。</p>}
      <p className="field-hint">本地检查不调用模型；/models 探测不验证生成或工具调用。token、概率与策略版本以实际记录为准。</p>
    </div>}
  </section>;
});

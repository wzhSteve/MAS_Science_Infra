import type { ModelReadinessState } from '../model/useModelReadiness';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';

const kinds: Record<string, string> = { api: '远程 API', local: '本地模型服务', rl_endpoint: '训练注入' };
const credentials = { experiment: '实验密钥', service: '服务默认密钥', none: '未提供密钥' };

export function ModelReadiness({ readiness, onConfigureModel }: {
  readiness: ModelReadinessState;
  onConfigureModel: () => void;
}) {
  const { data, error, loading, refresh } = readiness;
  return <section className="mas-model-readiness" aria-label="模型就绪状态" aria-busy={loading}>
    <div className="mas-model-summary">
      <strong>模型：{data?.model.model || '未确认'}</strong>
      {data && <span>{kinds[data.model.kind] || data.model.kind}</span>}
      <StatusBadge tone={error ? 'danger' : loading ? 'neutral' : data?.ready ? 'success' : 'warning'}>
        {error ? '检查失败' : loading ? '检查本地就绪状态…' : data?.ready ? '本地前置条件已满足' : '尚未就绪'}
      </StatusBadge>
      <Button size="sm" variant="ghost" onClick={onConfigureModel}>配置模型</Button>
      <Button size="sm" variant="ghost" disabled={loading} onClick={refresh}>{error ? '重试' : '刷新检查'}</Button>
    </div>
    {loading && <p className="field-hint" role="status">只检查本地配置与依赖，不请求模型。示例模拟执行不受影响。</p>}
    {error && <InlineNotice tone="danger">无法确认真实运行条件：{error} 示例模拟执行仍可使用。</InlineNotice>}
    {data && <>
      <p className="field-hint">{data.model.kind === 'api'
        ? '远程 API 无需本机显卡、VERL 或 AGL 训练服务。'
        : data.model.kind === 'local'
          ? '本地模式需要已运行的模型服务及其算力；保存配置不会自动启动服务。'
          : data.model.kind === 'rl_endpoint'
            ? '此模式由训练过程注入，独立真实运行请选择 API 或本地服务。'
            : '无法识别当前模型模式，请在 LLM 页面检查配置。'}
        {' '}模拟执行不使用模型；本地就绪不代表推理或工具调用已验证。</p>
      <details className="mas-model-details" open={!data.ready}>
        <summary>诊断与能力 · {data.probe ? data.probe.status === 'unsupported' ? '不支持模型列表探测' : data.probe.ok ? '模型列表可访问' : '最近探测有问题' : '尚未探测端点'}</summary>
        <dl>
          <div><dt>端点</dt><dd>{data.model.base_url || '未配置'}</dd></div>
          <div><dt>凭据来源</dt><dd>{credentials[data.model.credential_source]}（不回显）</dd></div>
          <div><dt>检查时间</dt><dd>{new Date(data.checked_at).toLocaleString()}</dd></div>
        </dl>
        {data.blocking_issues.map((issue, index) => <InlineNotice key={`${issue.code}-${index}`} tone="danger">
          {issue.message}{issue.hint && <p>{issue.hint}</p>}
        </InlineNotice>)}
        {data.warnings.map((issue, index) => <InlineNotice key={`${issue.code}-${index}`} tone="warning">
          {issue.message}{issue.hint && <p>{issue.hint}</p>}
        </InlineNotice>)}
        {(['live', 'parquet'] as const).map(name => {
          const dependency = data.dependencies[name];
          return <div key={name} className="mas-model-dependency">
            <strong>{name === 'live' ? '真实执行依赖' : 'parquet 数据读取依赖'}：{dependency.available ? '可用' : '不可用'}</strong>
            {!dependency.available && <>
              {dependency.missing.length > 0 && <p>缺少：{dependency.missing.join('、')}</p>}
              {dependency.error && <p>{dependency.error}</p>}
              <p>{dependency.hint}</p>
            </>}
          </div>;
        })}
        {data.probe && <p>最近模型列表探测：{data.probe.message} · {new Date(data.probe.checked_at).toLocaleString()}</p>}
        <p>端点探测仅检查 /models，不验证推理或工具调用。不支持模型列表不等于无法推理；可在 LLM 页面手动探测。</p>
        <p>工具调用、token IDs、logprobs、策略版本：均未确认。后续以真实调用记录为准，不阻止基础运行。</p>
      </details>
    </>}
  </section>;
}

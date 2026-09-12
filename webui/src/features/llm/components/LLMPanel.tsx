import { useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, HealthResponse } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { JsonDetails } from '../../../shared/components/JsonDetails';
import { LoadingState } from '../../../shared/components/LoadingState';
import { PageHeader } from '../../../shared/components/PageHeader';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';

type Props = { expId: string; bundle: Bundle | null; onReload: () => void };

const probeLabels: Record<HealthResponse['status'], string> = {
  reachable: '模型列表可访问',
  authentication_failed: '鉴权或权限失败',
  rate_limited: '请求限流',
  unsupported: '不支持模型列表探测',
  network_error: '网络连接失败',
  timeout: '请求超时',
  server_error: '服务端错误',
  invalid_response: '模型列表响应无效',
  configuration_error: '配置不完整或不支持',
  request_failed: '探测请求失败',
};

interface ProbeSnapshot {
  kind: string;
  model: string;
  baseUrl: string;
  inputRevision: number;
  configRevision: string | undefined;
}

export function LLMPanel(props: Props) {
  if (!props.bundle || props.bundle.id !== props.expId) return <LoadingState label="加载 llm.yaml…" />;
  return <LlmSettings key={props.expId} {...props} bundle={props.bundle} />;
}

function LlmSettings({ expId, bundle, onReload }: Props & { bundle: Bundle }) {
  const [kind, setKind] = useState(String(bundle.llm.kind || 'api'));
  const [model, setModel] = useState(String(bundle.llm.model || ''));
  const [baseUrl, setBaseUrl] = useState(String(bundle.llm.base_url || ''));
  const [modelPath, setModelPath] = useState(String(bundle.llm.model_path || ''));
  const [port, setPort] = useState(Number(bundle.llm.port || 8000));
  const [gpuMem, setGpuMem] = useState(Number(bundle.llm.gpu_memory_utilization || 0.45));
  const [apiKey, setApiKey] = useState('');
  const [probe, setProbe] = useState<{ result: HealthResponse; snapshot: ProbeSnapshot } | null>(null);
  const inputRevision = useRef(0);
  const keyRevision = useRef(0);
  const health = probe?.result;
  const probeStale = Boolean(probe && (probe.snapshot.inputRevision !== inputRevision.current
    || probe.snapshot.kind !== kind || probe.snapshot.model !== model || probe.snapshot.baseUrl !== baseUrl
    || probe.snapshot.configRevision !== bundle.llm.config_revision));
  const { pending, notice, run } = useAction();
  const stopAction = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const save = async () => {
    const submittedKeyRevision = keyRevision.current;
    const submitted = {
      kind, model, base_url: baseUrl, model_path: modelPath, port,
      gpu_memory_utilization: gpuMem, ...(apiKey ? { api_key: apiKey } : {}),
    };
    await api.putSection(expId, 'llm', submitted);
    if (mounted.current) {
      if (keyRevision.current === submittedKeyRevision && apiKey) {
        inputRevision.current += 1;
        keyRevision.current += 1;
        setApiKey('');
      }
      await onReload();
    }
  };

  return <div className="page-stack settings-page">
    <PageHeader title="LLM" eyebrow="模型连接" description="配置模型端点，或管理当前实验的本地 vLLM 进程。" />
    <Section title="连接配置">
      <FormField label="模式">
        <Select value={kind} onChange={(event) => { inputRevision.current += 1; setKind(event.target.value); }}>
          <option value="api">API（第三方 OpenAI-compat）</option>
          <option value="local">Local（一键启动 vLLM）</option>
          <option value="rl_endpoint">RL endpoint（训练时由 AGL 注入）</option>
        </Select>
      </FormField>
      {kind === 'api' && <p className="field-hint">远程 API 执行不需要本机显卡、VERL 或 AGL 训练服务。</p>}
      {kind === 'local' && <InlineNotice tone="info">需要可用的模型服务及其算力。保存只更新配置，不自动启动本地服务；已有兼容服务可直接填写端点。</InlineNotice>}
      {kind === 'rl_endpoint' && <InlineNotice tone="warning">此模式由训练过程注入。独立真实运行请选择 API 或本地服务；示例模拟执行不受影响。</InlineNotice>}
      <div className="form-grid">
        <FormField label="model"><Input value={model} onChange={(event) => { inputRevision.current += 1; setModel(event.target.value); }} /></FormField>
        <FormField label="base_url"><Input value={baseUrl} onChange={(event) => { inputRevision.current += 1; setBaseUrl(event.target.value); }} /></FormField>
      </div>
      {kind !== 'rl_endpoint' && <div className="grid gap-2">
        <FormField label="API Key" hint="写入实验 .secrets.env，不会回显。留空保留已有密钥；保存后仅清空未再次编辑的输入。">
          <Input type="password" autoComplete="off" placeholder={bundle.llm.api_key_set ? '••••（已配置）' : '可选：无鉴权服务可留空'} value={apiKey}
            onChange={(event) => { inputRevision.current += 1; keyRevision.current += 1; setApiKey(event.target.value); }} />
        </FormField>
        <div><StatusBadge tone={bundle.llm.api_key_set ? 'success' : 'neutral'}>
          {bundle.llm.credential_source === 'experiment' ? '使用实验密钥'
            : bundle.llm.credential_source === 'service' ? '使用服务默认密钥'
              : bundle.llm.api_key_set ? '密钥已配置' : '未提供密钥'}
        </StatusBadge></div>
      </div>}
      {kind === 'local' && <div className="form-grid">
        <FormField label="model_path"><Input value={modelPath} onChange={(event) => setModelPath(event.target.value)} /></FormField>
        <FormField label="port"><Input type="number" value={port} onChange={(event) => setPort(Number(event.target.value))} /></FormField>
        <FormField label="gpu_memory_utilization"><Input type="number" step="0.05" value={gpuMem} onChange={(event) => setGpuMem(Number(event.target.value))} /></FormField>
      </div>}
      <ActionBar>
        <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={() => {
          void run('save', async () => { await save(); return '已保存 llm.yaml；未发起模型探测。'; });
        }}>保存 llm.yaml</Button>
        <Button loading={pending === 'health'} disabled={pending !== null || kind === 'rl_endpoint'} onClick={() => {
          void run('health', async () => {
            const snapshot = { kind, model, baseUrl, inputRevision: inputRevision.current, configRevision: bundle.llm.config_revision };
            const result = await api.llmHealth(expId, { kind, model, base_url: baseUrl, api_key: apiKey });
            if (mounted.current) setProbe({ result, snapshot });
          });
        }}>探测连接</Button>
      </ActionBar>
      <p className="field-hint">手动探测使用当前模式、模型、端点和临时 Key；Key 留空时使用实验密钥或服务默认密钥，不保存配置。
        只请求 /models，不生成内容，不验证实际推理或工具调用。不支持模型列表不等于无法推理。</p>
      {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
    </Section>
    {kind === 'local' && <Section title="本地进程" description="启动前先保存当前配置；保存失败不会启动。Train 与本地 vLLM 互斥。">
      <ActionBar>
        <Button loading={pending === 'start'} disabled={pending !== null} onClick={() => {
          void run('start', async () => {
            await save();
            if (!mounted.current) return;
            const result = await api.llmStart(expId);
            if (mounted.current) onReload();
            return `LLM starting run=${result.run_id} → ${result.base_url}`;
          });
        }}>一键启动 LLM</Button>
        <Button variant="danger" loading={stopAction.pending !== null} onClick={() => {
          void stopAction.run('stop', async () => { await api.llmStop(expId); return 'LLM stopped'; });
        }}>停止 LLM</Button>
      </ActionBar>
      {stopAction.notice && <InlineNotice tone={stopAction.notice.tone}>{stopAction.notice.message}</InlineNotice>}
    </Section>}
    <Section title="连接探测结果" actions={<StatusBadge tone={!health ? 'neutral' : probeStale || health.status === 'unsupported' ? 'warning' : health.ok ? 'success' : 'danger'}>
      {!health ? '尚未探测' : probeStale ? '探测结果已过期' : probeLabels[health.status]}
    </StatusBadge>}>
      {!health ? <p className="field-hint">页面加载与配置保存不会自动发起探测。尚未探测不代表失败。</p> : <>
        {probeStale && <InlineNotice tone="warning">输入或已保存配置已变化，以下结果不代表当前配置；请按需重新探测。</InlineNotice>}
        <InlineNotice tone={probeStale || health.status === 'unsupported' ? 'warning' : health.ok ? 'success' : 'danger'}>
          {probeLabels[health.status]}：{health.message}
        </InlineNotice>
        <p className="field-hint">探测时间：{new Date(health.checked_at).toLocaleString()} · 仅模型列表；实际推理与工具调用均未验证。</p>
        <JsonDetails value={{
          status: health.status, status_code: health.status_code, models: health.models,
          url: health.url, checked_at: health.checked_at, probe_type: health.probe_type,
          inference_verified: health.inference_verified, tool_calling_verified: health.tool_calling_verified,
        }} label="查看探测详情（不含密钥）" />
      </>}
    </Section>
  </div>;
}

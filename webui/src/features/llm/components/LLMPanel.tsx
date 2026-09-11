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
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const { pending, notice, run } = useAction();
  const stopAction = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const save = async () => {
    await api.putSection(expId, 'llm', {
      kind, model, base_url: baseUrl, model_path: modelPath, port,
      gpu_memory_utilization: gpuMem, ...(apiKey ? { api_key: apiKey } : {}),
    });
    if (mounted.current) {
      setApiKey('');
      onReload();
    }
  };

  return <div className="page-stack settings-page">
    <PageHeader title="LLM" eyebrow="模型连接" description="配置模型端点，或管理当前实验的本地 vLLM 进程。" />
    <Section title="连接配置">
      <FormField label="模式">
        <Select value={kind} onChange={(event) => setKind(event.target.value)}>
          <option value="api">API（第三方 OpenAI-compat）</option>
          <option value="local">Local（一键启动 vLLM）</option>
          <option value="rl_endpoint">RL endpoint（训练时由 AGL 注入）</option>
        </Select>
      </FormField>
      {kind === 'rl_endpoint' && <InlineNotice tone="warning">Collect 阶段不可用；训练中由 Agent-Lightning ProxyLLM 注入。</InlineNotice>}
      <div className="form-grid">
        <FormField label="model"><Input value={model} onChange={(event) => setModel(event.target.value)} /></FormField>
        <FormField label="base_url"><Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></FormField>
      </div>
      {kind === 'api' && <div className="grid gap-2">
        <FormField label="API Key" hint="写入实验 .secrets.env，不会回显；保存成功后清空输入。">
          <Input type="password" autoComplete="off" placeholder={bundle.llm.api_key_set ? '••••（已配置）' : 'sk-...'} value={apiKey} onChange={(event) => setApiKey(event.target.value)} />
        </FormField>
        <div><StatusBadge tone={bundle.llm.api_key_set ? 'success' : 'neutral'}>{bundle.llm.api_key_set ? '密钥已配置' : '密钥未配置'}</StatusBadge></div>
      </div>}
      {kind === 'local' && <div className="form-grid">
        <FormField label="model_path"><Input value={modelPath} onChange={(event) => setModelPath(event.target.value)} /></FormField>
        <FormField label="port"><Input type="number" value={port} onChange={(event) => setPort(Number(event.target.value))} /></FormField>
        <FormField label="gpu_memory_utilization"><Input type="number" step="0.05" value={gpuMem} onChange={(event) => setGpuMem(Number(event.target.value))} /></FormField>
      </div>}
      <ActionBar>
        <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={() => {
          void run('save', async () => { await save(); return 'saved llm.yaml'; });
        }}>保存 llm.yaml</Button>
        <Button loading={pending === 'health'} disabled={pending !== null} onClick={() => {
          void run('health', async () => {
            const result = await api.llmHealth(expId, { base_url: baseUrl, api_key: apiKey || undefined });
            if (mounted.current) setHealth(result);
            if (!result.ok) throw new Error(`health fail: ${result.error || result.status_code}`);
            return 'health ok';
          });
        }}>探测连接</Button>
      </ActionBar>
      <p className="field-hint">探测使用当前输入的 base_url 与 API Key，不会保存配置。</p>
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
    {health && <Section title="连接探测结果" actions={<StatusBadge tone={health.ok ? 'success' : 'danger'}>{health.ok ? '连接正常' : '连接失败'}</StatusBadge>}>
      <JsonDetails value={health} label="查看完整探测返回" />
    </Section>}
  </div>;
}

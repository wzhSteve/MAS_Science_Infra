import { useEffect, useRef, useState } from 'react';
import { api, type Bundle } from '../api/client';

type Props = { expId: string; bundle: Bundle | null; onReload: () => void };

export function LLMPanel({ expId, bundle, onReload }: Props) {
  const [kind, setKind] = useState('api');
  const [model, setModel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [port, setPort] = useState(8000);
  const [gpuMem, setGpuMem] = useState(0.45);
  const [apiKey, setApiKey] = useState('');
  const [msg, setMsg] = useState('');
  const [health, setHealth] = useState<any>(null);
  const llm = bundle?.llm || {};
  const synced = useRef<string | null>(null);

  useEffect(() => {
    if (!bundle) return;
    if (synced.current === expId) return;
    setKind(String(bundle.llm.kind || 'api'));
    setModel(String(bundle.llm.model || ''));
    setBaseUrl(String(bundle.llm.base_url || ''));
    setModelPath(String(bundle.llm.model_path || ''));
    setPort(Number(bundle.llm.port || 8000));
    setGpuMem(Number(bundle.llm.gpu_memory_utilization || 0.45));
    synced.current = expId;
  }, [expId, bundle]);

  const save = async () => {
    try {
      await api.putSection(expId, 'llm', {
        kind,
        model,
        base_url: baseUrl,
        model_path: modelPath,
        port,
        gpu_memory_utilization: gpuMem,
        ...(apiKey ? { api_key: apiKey } : {}),
      });
      setApiKey('');
      setMsg('saved llm.yaml');
      onReload();
    } catch (e) {
      setMsg(String(e));
    }
  };

  return (
    <div className="card">
      <h2>LLM</h2>
      <div className="field">
        <label>模式</label>
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="api">API（第三方 OpenAI-compat）</option>
          <option value="local">Local（一键启动 vLLM）</option>
          <option value="rl_endpoint">RL endpoint（训练时由 AGL 注入）</option>
        </select>
      </div>
      {kind === 'rl_endpoint' ? (
        <p className="muted">Collect 阶段不可用；训练中由 Agent-Lightning ProxyLLM 注入。</p>
      ) : null}
      <div className="grid2">
        <div className="field">
          <label>model</label>
          <input value={model} onChange={(e) => setModel(e.target.value)} />
        </div>
        <div className="field">
          <label>base_url</label>
          <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
      </div>
      {kind === 'api' ? (
        <div className="field">
          <label>API Key（写入实验 .secrets.env，不会回显）</label>
          <input
            type="password"
            placeholder={llm.api_key_set ? '••••（已配置）' : 'sk-...'}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
      ) : null}
      {kind === 'local' ? (
        <div className="grid2">
          <div className="field">
            <label>model_path</label>
            <input value={modelPath} onChange={(e) => setModelPath(e.target.value)} />
          </div>
          <div className="field">
            <label>port</label>
            <input type="number" value={port} onChange={(e) => setPort(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>gpu_memory_utilization</label>
            <input
              type="number"
              step="0.05"
              value={gpuMem}
              onChange={(e) => setGpuMem(Number(e.target.value))}
            />
          </div>
        </div>
      ) : null}
      <div className="row">
        <button type="button" className="primary" onClick={save}>
          保存 llm.yaml
        </button>
        <button
          type="button"
          onClick={async () => {
            try {
              const h = await api.llmHealth(expId, {
                base_url: baseUrl,
                api_key: apiKey || undefined,
              });
              setHealth(h);
              setMsg(h.ok ? 'health ok' : `health fail: ${h.error || h.status_code}`);
            } catch (e) {
              setMsg(String(e));
            }
          }}
        >
          探测连接
        </button>
        {kind === 'local' ? (
          <>
            <button
              type="button"
              className="primary"
              onClick={async () => {
                try {
                  await save();
                  const r = await api.llmStart(expId);
                  setMsg(`LLM starting run=${r.run_id} → ${r.base_url}`);
                  onReload();
                } catch (e) {
                  setMsg(String(e));
                }
              }}
            >
              一键启动 LLM
            </button>
            <button
              type="button"
              className="danger"
              onClick={async () => {
                try {
                  await api.llmStop(expId);
                  setMsg('LLM stopped');
                } catch (e) {
                  setMsg(String(e));
                }
              }}
            >
              停止 LLM
            </button>
          </>
        ) : null}
      </div>
      {msg ? <p className="muted">{msg}</p> : null}
      {health ? (
        <pre style={{ fontSize: 12, overflow: 'auto' }}>{JSON.stringify(health, null, 2)}</pre>
      ) : null}
    </div>
  );
}

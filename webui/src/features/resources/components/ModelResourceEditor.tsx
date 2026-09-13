import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { modelResourcesApi, type ModelResource, type ModelResourceType, type ResourceDetail, type ResourceWrite } from '../api';
import { ApiError } from '../../../shared/api/http';
import type { HealthResponse } from '../../../shared/api/types';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { FormField } from '../../../shared/components/FormField';
import { Section } from '../../../shared/components/Section';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';

export interface ModelEditorTarget { id: string | null; type: ModelResourceType }
export interface ModelEditorHandle { canLeave: () => boolean }
interface EditorProps {
  target: ModelEditorTarget;
  active: boolean;
  returnExperimentId: string | null;
  requestedType?: ModelResourceType;
  onUse: (resource: ModelResource) => void;
  onClose: () => void;
  onSaved: (resource: ModelResource) => void;
  onDeleted: () => void;
}
interface Draft {
  name: string;
  kind: 'api' | 'local';
  model: string;
  baseUrl: string;
  modelPath: string;
  port: string;
  gpuMemory: string;
  credentialMode: ModelResource['credential_mode'];
}
const draftFor = (resource: ModelResource | null, type: ModelResourceType): Draft => ({
  name: resource?.name || '',
  kind: resource?.config.kind || (type === 'training' ? 'local' : 'api'),
  model: resource?.config.model || '',
  baseUrl: resource?.config.base_url || '',
  modelPath: resource?.config.model_path || '',
  port: resource?.config.port?.toString() || '',
  gpuMemory: resource?.config.gpu_memory_utilization?.toString() || '',
  credentialMode: resource?.credential_mode || (type === 'training' ? 'none' : 'saved'),
});
function failure(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) return '资源修订或引用状态已变化，操作未完成。草稿已保留；请核对引用或重新载入最新资源后再修改。';
    if (error.status === 404) return '资源已不存在。未保存内容仍保留，请关闭后刷新资源列表。';
    if (error.status === 400 || error.status === 422) return '资源配置未通过校验，请核对名称、路径、端点及凭据方式。';
    if (error.status === 401 || error.status === 403) return '没有权限完成此操作，请检查服务访问权限。';
  }
  return '操作失败，请检查服务状态后重试。未保存内容仍保留。';
}
const probeLabels: Record<HealthResponse['status'], string> = {
  reachable: '模型列表接口可达',
  authentication_failed: '身份认证失败',
  rate_limited: '服务限流',
  unsupported: '服务不支持此检查',
  network_error: '无法连接服务',
  timeout: '连接超时',
  server_error: '模型服务错误',
  invalid_response: '响应格式无效',
  configuration_error: '连接配置无效',
  request_failed: '检查请求失败',
};

export const ModelResourceEditor = memo(forwardRef<ModelEditorHandle, EditorProps>(function ModelResourceEditor({
  target, active, returnExperimentId, requestedType, onUse, onClose, onSaved, onDeleted,
}, ref) {
  const [base, setBase] = useState<ModelResource | null>(null);
  const [references, setReferences] = useState<ResourceDetail['references'] | null>(target.id ? null : []);
  const [draft, setDraft] = useState<Draft>(() => draftFor(null, target.type));
  const [apiKey, setApiKey] = useState('');
  const [clearKey, setClearKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadVersion, setLoadVersion] = useState(0);
  const [pending, setPending] = useState<'save' | 'probe' | 'delete' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [probe, setProbe] = useState<{ revision: number; result: HealthResponse } | null>(null);
  const [conflict, setConflict] = useState(false);
  const loadedVersion = useRef(-1);
  const operation = useRef(false);
  const mounted = useRef(true);
  const currentBase = useRef(base);
  currentBase.current = base;
  const type = base?.type || target.type;
  const initial = draftFor(base, type);
  const ready = !target.id || base !== null;
  const dirty = ready && (JSON.stringify(draft) !== JSON.stringify(initial) || apiKey.length > 0 || clearKey);
  const busy = loading || pending !== null;

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    if (!active) { setLoading(false); return; }
    const id = currentBase.current?.id || target.id;
    if (!id || loadedVersion.current === loadVersion) return;
    const controller = new AbortController();
    setLoading(true); setError(null);
    void modelResourcesApi.get(id, controller.signal).then(result => {
      if (controller.signal.aborted) return;
      setBase(result); setDraft(draftFor(result, result.type)); setReferences(result.references);
      setApiKey(''); setClearKey(false); setProbe(null); setConflict(false); setNotice(null);
      loadedVersion.current = loadVersion;
    }).catch(reason => {
      if (!controller.signal.aborted) {
        // A failed explicit reload must not later overwrite newly edited input on reactivation.
        loadedVersion.current = loadVersion;
        setError(failure(reason));
      }
    })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [active, target.id, loadVersion]);

  const canLeave = useCallback(() => {
    if (operation.current) return false;
    return !dirty || window.confirm('放弃此资源未保存的修改及已输入的密钥？此操作不会修改服务端资源。');
  }, [dirty]);
  useImperativeHandle(ref, () => ({ canLeave }), [canLeave]);

  const save = useCallback(async (): Promise<boolean> => {
    if (!ready || loading || operation.current || conflict) return false;
    setError(null); setNotice(null);
    if (!draft.name.trim()) { setError('请填写资源名称。'); return false; }
    if (type === 'training' && !draft.modelPath.trim()) { setError('训练模型来源必须填写权重 / checkpoint 路径。'); return false; }
    if (type === 'inference' && !draft.model.trim()) { setError('请填写模型名称。'); return false; }
    if (type === 'inference' && !draft.baseUrl.trim()) { setError('请填写 API 端点；本地模型服务也需要显式地址。'); return false; }
    if (type === 'inference' && draft.baseUrl.trim()) {
      try {
        const url = new URL(draft.baseUrl.trim());
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
      } catch { setError('端点须为 HTTP(S) 地址，不能包含账号、密钥、查询参数或片段。'); return false; }
    }
    const port = draft.port.trim() ? Number(draft.port) : undefined;
    const memory = draft.gpuMemory.trim() ? Number(draft.gpuMemory) : undefined;
    if (type === 'inference' && draft.kind === 'local' && (port !== undefined && (!Number.isInteger(port) || port < 1 || port > 65535))) {
      setError('端口须为 1–65535 的整数。'); return false;
    }
    if (type === 'inference' && draft.kind === 'local' && (memory !== undefined && (!Number.isFinite(memory) || memory <= 0 || memory > 1))) {
      setError('GPU 显存利用率须大于 0 且不超过 1。'); return false;
    }
    if (type === 'inference' && draft.credentialMode === 'saved' && !apiKey.trim() && (!base?.api_key_set || clearKey)) {
      setError('保存资源密钥模式需要填写密钥；不使用密钥时请选择“不使用凭据”，或显式选择服务默认凭据。'); return false;
    }
    const body: ResourceWrite = {
      name: draft.name.trim(), type,
      config: type === 'training' ? { model_path: draft.modelPath.trim() } : {
        kind: draft.kind, model: draft.model.trim(), base_url: draft.baseUrl.trim(),
        ...(draft.kind === 'local' ? { model_path: draft.modelPath.trim(), port, gpu_memory_utilization: memory } : {}),
      },
      credential_mode: type === 'training' ? 'none' : draft.credentialMode,
      ...(type === 'inference' && draft.credentialMode === 'saved' && apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      ...(type === 'inference' && clearKey ? { clear_api_key: true } : {}),
    };
    operation.current = true; setPending('save');
    try {
      const result = base ? await modelResourcesApi.update(base.id, base.revision, body) : await modelResourcesApi.create(body);
      if (!mounted.current) return true;
      setBase(result); setDraft(draftFor(result, result.type)); setApiKey(''); setClearKey(false);
      setProbe(null); setConflict(false);
      loadedVersion.current = loadVersion;
      setNotice('资源已保存。未自动检查连接、绑定实验或执行模型。');
      onSaved(result);
      return true;
    } catch (reason) {
      if (mounted.current) {
        setError(failure(reason));
        if (reason instanceof ApiError && reason.status === 409) setConflict(true);
      }
      return false;
    } finally { operation.current = false; if (mounted.current) setPending(null); }
  }, [ready, loading, conflict, draft, type, apiKey, base, clearKey, loadVersion, onSaved]);

  useUnsavedChanges('model-resource-editor', {
    label: '个人模型资源草稿', resource: 'personal-model', dirty, busy, save,
  });
  const change = <K extends keyof Draft,>(key: K, value: Draft[K]) => {
    setDraft(previous => ({ ...previous, [key]: value })); setNotice(null);
  };
  const reload = () => {
    if (operation.current || loading) return;
    if (!canLeave()) return;
    setLoadVersion(value => value + 1);
  };
  const check = async () => {
    if (!base || dirty || busy || conflict || operation.current) return;
    operation.current = true; setPending('probe'); setError(null); setNotice(null); setProbe(null);
    try {
      const result = await modelResourcesApi.probe(base.id, base.revision);
      if (mounted.current) setProbe({ revision: base.revision, result });
    } catch (reason) {
      if (mounted.current) {
        setError(failure(reason));
        if (reason instanceof ApiError && reason.status === 409) setConflict(true);
      }
    } finally { operation.current = false; if (mounted.current) setPending(null); }
  };
  const remove = async () => {
    if (!base || busy || operation.current || references === null || references.length > 0 || conflict) return;
    if (!window.confirm(`删除个人资源“${base.name}”？${dirty ? '未保存修改和已输入的密钥也会丢弃。' : ''}不会删除模型文件或历史运行；服务端将再次检查实验引用。`)) return;
    operation.current = true; setPending('delete'); setError(null); setNotice(null);
    try {
      await modelResourcesApi.remove(base.id, base.revision);
      if (mounted.current) onDeleted();
    } catch (reason) {
      if (mounted.current) {
        setError(failure(reason));
        if (reason instanceof ApiError && reason.status === 409) setConflict(true);
      }
    } finally { operation.current = false; if (mounted.current) setPending(null); }
  };
  const clearCredential = () => {
    if (!window.confirm('确认清除该资源已保存的密钥？当前输入也会清空，并切换为不使用凭据。点击保存后才在服务端生效。')) return;
    setApiKey(''); setClearKey(true); change('credentialMode', 'none');
  };
  const changeCredentialMode = (mode: Draft['credentialMode']) => {
    if (apiKey && mode !== 'saved') {
      if (!window.confirm('切换凭据方式会丢弃当前输入的密钥，是否继续？服务端已有密钥不会因此自动清除。')) return;
      setApiKey('');
    }
    change('credentialMode', mode);
  };

  return <Section className="model-catalog-editor" title={base ? `编辑资源 · ${base.name}` : target.id ? '资源详情' : `新增${type === 'training' ? '训练模型来源' : '推理连接'}`}
    actions={<div className="model-catalog-actions">
      <StatusBadge tone={dirty ? 'warning' : 'neutral'}>{busy ? '处理中' : dirty ? '未保存' : base ? '已保存' : '尚未创建'}</StatusBadge>
      <Button size="sm" disabled={pending !== null} onClick={onClose}>关闭详情</Button>
    </div>}>
    {loading && <p className="field-hint" role="status">正在读取资源及引用关系…</p>}
    {error && <InlineNotice tone="danger">{error}</InlineNotice>}
    {notice && <InlineNotice tone="success">{notice}</InlineNotice>}
    {(base || target.id) && <div className="model-catalog-actions">
      <Button size="sm" disabled={busy} onClick={reload}>重新载入资源 / 引用</Button>
      {base && <span className="field-hint">修订 {base.revision} · {base.id}</span>}
    </div>}
    {ready && <>
      {base && <div className="model-catalog-scope">
        <InlineNotice tone="warning">
          当前已知有 {references?.length ?? '未知数量的'} 个实验引用。修改会影响引用实验之后发起的调用；
          已开始的调用和历史记录不改用新值。只想修改某一个实验时，请新增独立资源再绑定。
        </InlineNotice>
        {references && references.length > 0 && <ul className="model-catalog-references" aria-label="引用此资源的实验" tabIndex={0}>
          {references.map(reference => <li key={`${reference.experiment_id}:${reference.purpose}`}>
            <span>{reference.experiment_id}</span><StatusBadge>{reference.purpose === 'training' ? '训练来源' : '默认推理'}</StatusBadge>
          </li>)}
        </ul>}
      </div>}
      {type === 'training' && <InlineNotice>这里只登记服务端权重 / checkpoint 路径，不上传或下载文件，不保证文件存在、GPU 可用或实际可训练。</InlineNotice>}
      <form onSubmit={event => { event.preventDefault(); void save(); }}>
        <fieldset disabled={busy} className="model-catalog-fields">
          <FormField label="资源名称"><Input value={draft.name} maxLength={120} autoComplete="off"
            placeholder="例如：日常调试模型" onChange={event => change('name', event.target.value)} /></FormField>
          <FormField label="资源类型" hint="类型创建后不可更改；不同用途请另建资源。">
            <Select value={type} disabled><option value="inference">推理连接</option><option value="training">训练模型来源</option></Select>
          </FormField>
          {type === 'inference' && <>
            <FormField label="连接方式"><Select value={draft.kind} onChange={event => change('kind', event.target.value as Draft['kind'])}>
              <option value="api">API 服务</option><option value="local">本地模型服务</option>
            </Select></FormField>
            <FormField label="模型名称"><Input value={draft.model} maxLength={256} autoComplete="off"
              onChange={event => change('model', event.target.value)} /></FormField>
            <FormField label="API 端点" hint={draft.kind === 'local' ? '必须填写。实验内启动要求 http://127.0.0.1:<端口>/v1 或 localhost，端口须与下方一致（默认 8000）。保存不启动服务，也不自动改写地址。' : '填写 HTTP(S) 基础地址，不要把密钥放入地址。'}>
              <Input value={draft.baseUrl} maxLength={2048} autoComplete="off" placeholder="https://example.com/v1"
                onChange={event => change('baseUrl', event.target.value)} />
            </FormField>
          </>}
          {(type === 'training' || draft.kind === 'local') && <FormField
            label={type === 'training' ? '权重 / checkpoint 路径' : '权重 / checkpoint 路径（可选）'}
            hint={type === 'training' ? '填写训练进程可读取的路径，不是浏览器本机文件上传。'
              : '连接已启动的本地服务可留空；从实验启动新服务时必须填写模型服务可读取的权重路径。'}>
            <Input value={draft.modelPath} maxLength={4096} autoComplete="off" onChange={event => change('modelPath', event.target.value)} />
          </FormField>}
          {type === 'inference' && draft.kind === 'local' && <>
            <FormField label="本地服务端口（可选）" hint="留空使用服务端默认端口 8000。"><Input type="number" min={1} max={65535} step={1} value={draft.port} placeholder="8000"
              onChange={event => change('port', event.target.value)} /></FormField>
            <FormField label="GPU 显存利用率（可选）" hint="取值大于 0 且不超过 1；留空使用服务端默认值 0.45。">
              <Input type="number" min={0.001} max={1} step="any" value={draft.gpuMemory} placeholder="0.45" onChange={event => change('gpuMemory', event.target.value)} />
            </FormField>
          </>}
          {type === 'inference' && <>
            <FormField label="凭据方式" hint="不继承任何实验的密钥；服务默认凭据必须显式选择。">
              <Select value={draft.credentialMode} onChange={event => changeCredentialMode(event.target.value as Draft['credentialMode'])}>
                <option value="saved">保存资源专用密钥</option><option value="service">使用服务默认凭据</option><option value="none">不使用凭据</option>
              </Select>
            </FormField>
            {draft.credentialMode === 'saved' && <FormField label={base?.api_key_set ? '替换 API Key（可选）' : 'API Key'}
              hint={base?.api_key_set && !clearKey ? '留空保留服务端已有密钥；如需删除请使用显式清除。' : '此凭据方式需要填写密钥。密钥仅保留在当前表单内存，刷新不会恢复。'}>
              <Input type="password" autoComplete="new-password" spellCheck={false} value={apiKey} maxLength={8192}
                onChange={event => { setApiKey(event.target.value); if (event.target.value) setClearKey(false); setNotice(null); }} />
            </FormField>}
            <div className="model-catalog-credential">
              <span className="field-hint">{clearKey ? '待清除密钥（保存后生效）' : base?.api_key_set ? '服务端已保存资源密钥；切换凭据来源不等于删除密钥。' : '未保存资源专用密钥。'}</span>
              {base?.api_key_set && !clearKey && <Button size="sm" variant="danger" onClick={clearCredential}>清除已保存密钥…</Button>}
              {clearKey && <Button size="sm" onClick={() => { setClearKey(false); change('credentialMode', base?.credential_mode || 'saved'); }}>撤销清除</Button>}
            </div>
          </>}
        </fieldset>
        <div className="model-catalog-editor-footer">
          <Button type="submit" variant="primary" loading={pending === 'save'} disabled={busy || conflict || (Boolean(base) && !dirty)}>保存资源</Button>
          {type === 'inference' && <Button loading={pending === 'probe'} disabled={!base || busy || dirty || conflict} onClick={() => void check()}>手动检查已保存连接</Button>}
          {base && <Button variant="danger" disabled={busy || conflict || references === null || references.length > 0}
            onClick={() => void remove()}>删除资源…</Button>}
          {returnExperimentId && base && <Button disabled={busy || dirty || conflict || (requestedType !== undefined && requestedType !== type)}
            onClick={() => onUse(base)}>选择用于原实验</Button>}
        </div>
      </form>
      {dirty && <p className="field-hint">请先保存修改，再手动检查或选择用于原实验。保存不会自动检查。</p>}
      {references && references.length > 0 && <p className="field-hint">资源仍被引用，不能删除。请先在对应实验中显式解除或更换绑定。</p>}
      {returnExperimentId && requestedType && requestedType !== type && <p className="field-hint">资源类型与原实验选择器不匹配，不能用于该选择器。</p>}
      {type === 'inference' && <div className="model-catalog-probe">
        <StatusBadge tone={dirty || conflict ? 'warning' : probe ? (probe.result.ok ? 'success' : 'danger') : 'neutral'}>
          {dirty || conflict ? '修改未确认，检查状态不可用于当前草稿' : probe ? (probe.result.ok ? '最近检查成功' : '最近检查失败') : '尚未检查'}
        </StatusBadge>
        {probe && <p>已保存修订 {probe.revision} · {probeLabels[probe.result.status] || '检查已完成'}
          {probe.result.status_code ? ` · HTTP ${probe.result.status_code}` : ''}</p>}
        <p className="field-hint">仅检查 /models 接口；不验证生成或工具调用能力。保存、切换凭据或启动服务后，请重新手动检查。</p>
      </div>}
    </>}
  </Section>;
}));

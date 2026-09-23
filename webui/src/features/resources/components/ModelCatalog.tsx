import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { modelResourcesApi, type ModelResource, type ModelResourceType, type ResourceList } from '../api';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { ModelResourceEditor, type ModelEditorHandle, type ModelEditorTarget } from './ModelResourceEditor';

export interface ModelCatalogProps {
  active: boolean;
  returnExperimentId: string | null;
  requestedType?: 'inference' | 'training';
  onUse: (resource: ModelResource) => void;
}

const PAGE_SIZE = 12;
const typeLabel = (type: ModelResourceType) => type === 'inference' ? '推理连接' : '训练模型来源';

const ModelCard = memo(function ModelCard({ resource, selected, onSelect }: {
  resource: ModelResource; selected: boolean; onSelect: (resource: ModelResource) => void;
}) {
  return <article className={`resource-card model-catalog-card${selected ? ' model-catalog-card--selected' : ''}`}>
    <header><h3>{resource.name}</h3><StatusBadge>{typeLabel(resource.type)}</StatusBadge></header>
    <dl>
      <dt>配置摘要</dt>
      <dd>{resource.type === 'training' ? (resource.config.model_path ? '权重 / checkpoint 路径已填写' : '未填写路径')
        : `${resource.config.kind === 'local' ? '本地服务' : 'API'} · ${resource.config.model || '未填写模型名'}`}</dd>
      <dt>凭据来源</dt>
      <dd>{resource.credential_mode === 'service' ? '显式使用服务默认凭据'
        : resource.credential_mode === 'none' ? '不使用凭据' : resource.api_key_set ? '资源密钥已保存' : '资源密钥未配置'}</dd>
      <dt>资源修订</dt><dd>{resource.revision}</dd>
    </dl>
    <p>{resource.type === 'training' ? '登记路径不代表权重可用或能够训练。' : '连接状态请在详情中手动检查。'}</p>
    <footer><span>引用实验按需读取</span>
      <Button size="sm" aria-pressed={selected} onClick={() => onSelect(resource)}>查看 / 编辑</Button>
    </footer>
  </article>;
});

const ModelList = memo(function ModelList({ active, refreshVersion, selectedId, onSelect }: {
  active: boolean; refreshVersion: number; selectedId: string | null; onSelect: (resource: ModelResource) => void;
}) {
  const [type, setType] = useState<ModelResourceType | ''>('');
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [reload, setReload] = useState(0);
  const [data, setData] = useState<ResourceList | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const loadedKey = useRef('');
  const requestKey = JSON.stringify([type, search, offset, refreshVersion, reload]);

  useEffect(() => {
    if (!active) { setLoading(false); return; }
    if (loadedKey.current === requestKey) return;
    const controller = new AbortController();
    setLoading(true); setError(false);
    void modelResourcesApi.list({ type: type || undefined, q: search, offset, limit: PAGE_SIZE }, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        if (result.total === 0 && offset > 0) { setOffset(0); return; }
        if (result.total > 0 && offset >= result.total) {
          setOffset(Math.floor((result.total - 1) / PAGE_SIZE) * PAGE_SIZE);
          return;
        }
        setData(result); loadedKey.current = requestKey;
      })
      .catch(() => { if (!controller.signal.aborted) setError(true); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [active, requestKey, type, search, offset]);

  return <section className="model-catalog-list" aria-label="个人模型资源列表" aria-busy={loading}>
    <form className="model-catalog-filters" onSubmit={event => {
      event.preventDefault(); setSearch(query.trim()); setOffset(0); setReload(value => value + 1);
    }}>
      <label>用途<Select value={type} onChange={event => {
        setType(event.target.value as ModelResourceType | ''); setOffset(0);
      }}>
        <option value="">全部类型</option><option value="inference">推理连接</option><option value="training">训练模型来源</option>
      </Select></label>
      <Input aria-label="搜索个人模型资源" placeholder="按名称搜索资源…" maxLength={120}
        value={query} onChange={event => setQuery(event.target.value)} />
      <Button type="submit" disabled={loading}>搜索</Button>
      <Button disabled={loading} onClick={() => setReload(value => value + 1)}>刷新列表</Button>
    </form>
    {error && <InlineNotice tone="danger">资源列表读取失败，请刷新重试。已显示的列表可能不是最新结果。</InlineNotice>}
    {loading && <p className="field-hint" role="status">正在读取资源列表…</p>}
    {data && data.items.length > 0 && <div className="resource-grid">
      {data.items.map(resource => <ModelCard key={resource.id} resource={resource}
        selected={selectedId === resource.id} onSelect={onSelect} />)}
    </div>}
    {!loading && !error && data?.items.length === 0 && <div className="resource-empty">
      {search || type ? '没有符合条件的资源，可调整筛选或新增资源。' : '还没有个人模型资源。可先新增连接，无需创建或选择实验。'}
    </div>}
    {data && <nav className="model-catalog-pagination" aria-label="模型资源分页">
      <Button size="sm" disabled={loading || offset === 0} onClick={() => setOffset(value => Math.max(0, value - PAGE_SIZE))}>上一页</Button>
      <span>{data.total ? `${data.offset + 1}–${Math.min(data.offset + data.items.length, data.total)}` : '0'} / {data.total}</span>
      <Button size="sm" disabled={loading || offset + PAGE_SIZE >= data.total} onClick={() => setOffset(value => value + PAGE_SIZE)}>下一页</Button>
    </nav>}
  </section>;
});

export const ModelCatalog = memo(function ModelCatalog({ active, returnExperimentId, requestedType, onUse }: ModelCatalogProps) {
  const [target, setTarget] = useState<(ModelEditorTarget & { session: number }) | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const editor = useRef<ModelEditorHandle>(null);
  const sequence = useRef(0);
  const open = useCallback(async (next: ModelEditorTarget) => {
    if (editor.current && !(await editor.current.canLeave())) return;
    setSelectedId(next.id);
    setTarget({ ...next, session: ++sequence.current });
  }, []);
  const select = useCallback((resource: ModelResource) => {
    if (resource.id === selectedId) return;
    void open({ id: resource.id, type: resource.type });
  }, [open, selectedId]);
  const close = useCallback(async () => {
    if (editor.current && !(await editor.current.canLeave())) return;
    setTarget(null); setSelectedId(null);
  }, []);
  const saved = useCallback((resource: ModelResource) => {
    setSelectedId(resource.id); setRefreshVersion(value => value + 1);
  }, []);
  const removed = useCallback(() => {
    setTarget(null); setSelectedId(null); setRefreshVersion(value => value + 1);
  }, []);

  return <div className="model-catalog" hidden={!active}>
    <div className="model-catalog-heading">
      <div><h2>个人模型资源库</h2></div>
      <div className="model-catalog-actions">
        <Button variant="primary" onClick={() => void open({ id: null, type: 'inference' })}>新增推理连接</Button>
        <Button onClick={() => void open({ id: null, type: 'training' })}>登记训练模型来源</Button>
      </div>
    </div>
    {returnExperimentId && <InlineNotice>
      来自实验 {returnExperimentId} 的{requestedType === 'training' ? '训练来源' : requestedType === 'inference' ? '推理模型' : '模型'}选择器。
      保存资源后仍需点击“选择用于原实验”，并回到实验显式保存绑定；这里不会自动绑定或执行。
    </InlineNotice>}
    <ModelList active={active} refreshVersion={refreshVersion} selectedId={selectedId} onSelect={select} />
    {target && <ModelResourceEditor key={target.session} ref={editor} target={target} active={active}
      returnExperimentId={returnExperimentId} requestedType={requestedType} onUse={onUse}
      onClose={() => void close()} onSaved={saved} onDeleted={removed} />}
  </div>;
});

import { memo, useCallback, useState } from 'react';
import { Database, Plus, RefreshCw, X } from 'lucide-react';
import { datasetResourcesApi, type DatasetResource } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { DataPreview } from './DataPreview';

type EditTarget = { resource: DatasetResource | null; revision: number };

function DatasetEditor({ target, onSaved, onClose }: {
  target: EditTarget; onSaved: () => void; onClose: () => void;
}) {
  const [name, setName] = useState(target.resource?.name || '');
  const [path, setPath] = useState(target.resource?.path || '');
  const action = useAction();
  const dirty = name !== (target.resource?.name || '') || path !== (target.resource?.path || '');
  const save = useCallback(() => action.run('save', async () => {
    await datasetResourcesApi.save({ name, path, revision: target.revision }, target.resource?.id);
    onSaved();
  }), [action.run, name, path, target, onSaved]);
  useUnsavedChanges('dataset-resource-editor', { label: '共享数据目录', resource: 'dataset-catalog', dirty, busy: action.pending !== null, save });
  return <section className="resource-details">
    <header><h2>{target.resource ? '编辑数据' : '登记数据'}</h2>
      <Button size="sm" variant="ghost" disabled={action.pending !== null} aria-label="关闭数据编辑" onClick={() => {
        if (!dirty || window.confirm('放弃未保存的数据目录修改？')) onClose();
      }}><X size={16} /></Button>
    </header>
    <form className="page-stack" onSubmit={event => { event.preventDefault(); void save(); }}>
      <FormField label="名称"><Input required disabled={action.pending !== null} value={name} onChange={event => setName(event.target.value)} /></FormField>
      <FormField label="服务器 Parquet 路径">
        <Input required disabled={action.pending !== null} value={path} placeholder="/root/autodl-tmp/MAS_Science_Infra/data/train.parquet" onChange={event => setPath(event.target.value)} />
      </FormField>
      {action.notice && <InlineNotice tone={action.notice.tone}>{action.notice.message}</InlineNotice>}
      <div className="action-bar"><Button type="submit" variant="primary" disabled={!dirty || action.pending !== null} loading={action.pending === 'save'}>保存数据</Button></div>
    </form>
  </section>;
}

export const DatasetCatalog = memo(function DatasetCatalog({ active }: { active: boolean }) {
  const catalog = usePollingResource('shared-datasets', datasetResourcesApi.list, undefined, active);
  const [editing, setEditing] = useState<EditTarget | null>(null);
  const [preview, setPreview] = useState<DatasetResource | null>(null);
  const saved = useCallback(() => { setEditing(null); void catalog.refresh(); }, [catalog.refresh]);
  return <>
    <div className="resources-toolbar">
      <Button size="sm" disabled={!catalog.data || editing !== null} onClick={() => {
        if (catalog.data) setEditing({ resource: null, revision: catalog.data.revision });
        setPreview(null);
      }}><Plus size={14} />登记数据</Button>
      <Button size="sm" variant="ghost" onClick={() => void catalog.refresh()}><RefreshCw size={14} />刷新</Button>
    </div>
    {catalog.error && <InlineNotice tone="danger">数据目录读取失败：{catalog.error}</InlineNotice>}
    {editing && <DatasetEditor key={editing.resource?.id || 'new'} target={editing} onSaved={saved} onClose={() => setEditing(null)} />}
    <div className="resource-grid">{catalog.data?.items.map(resource => <article className="resource-card" key={resource.id}>
      <header><span className="resource-icon"><Database size={20} /></span><span>Parquet</span></header>
      <h2>{resource.name}</h2><p className="mono break-all">{resource.path}</p>
      <footer>
        <Button size="sm" variant="ghost" disabled={editing !== null} onClick={() => {
          if (catalog.data) setEditing({ resource, revision: catalog.data.revision });
          setPreview(null);
        }}>编辑</Button>
        <Button size="sm" disabled={editing !== null} onClick={() => setPreview(resource)}>预览</Button>
      </footer>
    </article>)}</div>
    {!catalog.error && !catalog.data?.items.length && <p className="resource-empty">{catalog.loading ? '读取数据目录…' : '暂无数据资源'}</p>}
    {preview && <section className="resource-details"><header><h2>{preview.name}</h2>
      <Button size="sm" variant="ghost" aria-label="关闭数据预览" onClick={() => setPreview(null)}><X size={16} /></Button></header>
      <DataPreview key={preview.id} path={preview.path} active={active} />
    </section>}
  </>;
});

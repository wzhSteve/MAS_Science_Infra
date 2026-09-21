import { memo, useCallback, useState } from 'react';
import { ArrowLeft, Cpu, Database, RefreshCw, X } from 'lucide-react';
import type { ResourceCategory, ResourceRoute } from '../app/navigation';
import type { SettingsSection } from '../features/settings/model/sections';
import type { ModelResource, ModelResourceType } from '../features/resources/api';
import { ModelCatalog } from '../features/resources/components/ModelCatalog';
import { experimentApi } from '../features/experiment/api';
import { usePollingResource } from '../shared/hooks/usePollingResource';
import { useResourceOverview } from '../features/resources/model/useResourceOverview';
import { DataPreview } from '../features/resources/components/DataPreview';
import { Button } from '../shared/ui/button';
import { Select } from '../shared/ui/select';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { StatusBadge } from '../shared/components/StatusBadge';

const ExperimentData = memo(function ExperimentData({ experimentId, active, onConfigure }: {
  experimentId: string; active: boolean; onConfigure: (id: string, section: SettingsSection) => void;
}) {
  const resource = useResourceOverview(experimentId, active);
  const [detail, setDetail] = useState<{ kind: 'training' | 'validation' | 'preview'; path: string } | null>(null);
  const data = resource.data;
  if (!data) return resource.error ? <InlineNotice tone="danger">读取数据配置失败：{resource.error}
    <Button size="sm" onClick={() => void resource.refresh()}>重试</Button></InlineNotice> : <LoadingState label="读取数据配置…" />;
  const paths = detail && detail.kind !== 'preview' ? data.data[detail.kind] : [];
  return <>
    <div className="resource-scope"><strong>{data.experimentName} · 数据配置</strong>
      <Button size="sm" variant="ghost" onClick={() => void resource.refresh()}><RefreshCw size={13} />刷新配置</Button></div>
    <p className="field-hint resource-saved-note">数据入口暂保留实验内已保存路径，不包含未保存草稿；尚未确定上传 / 导入方式。</p>
    {resource.error && <InlineNotice tone="warning">刷新失败：{resource.error}</InlineNotice>}
    <div className="resource-grid">
      {(['training', 'validation'] as const).map(kind => {
        const values = data.data[kind];
        const unsupported = kind === 'training' ? data.data.unsupportedTraining : data.data.unsupportedValidation;
        return <article className="resource-card" key={kind}>
          <header><span className="resource-icon"><Database size={20} /></span><StatusBadge tone="neutral">{kind === 'training' ? '训练用途' : '训练内验证用途'}</StatusBadge></header>
          <h2>{kind === 'training' ? '训练数据' : '验证数据'}</h2>
          <p>rl.yaml / data.{kind === 'training' ? 'train_files' : 'val_files'}</p>
          {values.length ? <ul className="resource-paths">{values.slice(0, 3).map((path, index) => <li key={index}>{path}</li>)}</ul>
            : <p className="resource-unconfigured">{unsupported ? '字段格式暂无法展示为路径。'
              : '未显式保存路径，训练入口可能使用默认配置；文件是否存在尚未确认。'}</p>}
          {values.length > 3 && <span className="field-hint">共 {values.length} 个路径</span>}
          <footer><span>规模和版本未验证</span><Button size="sm" onClick={() => setDetail({ kind, path: values[0] || '' })}>查看 / 预览</Button></footer>
        </article>;
      })}
    </div>
    <div className="resource-preview-entry"><span>已有服务端任务文件？只预览，不注册或保存训练路径。</span>
      <Button size="sm" onClick={() => setDetail({ kind: 'preview', path: '' })}>预览 parquet 文件</Button></div>
    {detail && <section className="resource-details">
      <header><h2>{detail.kind === 'training' ? '训练数据' : detail.kind === 'validation' ? '训练内验证数据' : '任务文件预览'}</h2>
        <Button size="sm" variant="ghost" onClick={() => setDetail(null)} aria-label="关闭预览"><X size={16} /></Button></header>
      <p className="field-hint">验证数据不等于已执行独立评估；任务预览不保证完整训练格式兼容性。</p>
      {paths.length > 1 && <Select aria-label="已配置路径" value={detail.path} onChange={event => setDetail({ ...detail, path: event.target.value })}>
        {paths.map((path, index) => <option key={index} value={path}>{path}</option>)}
      </Select>}
      <DataPreview key={`${detail.kind}:${detail.path}`} path={detail.path} active={active} />
      <footer><Button size="sm" onClick={() => onConfigure(experimentId, 'data')}>返回数据与采样设置</Button></footer>
    </section>}
  </>;
});

function DatasetResources({ experimentId, active, currentExperimentId, onNavigate, onConfigure }: {
  experimentId?: string; active: boolean; currentExperimentId: string | null;
  onNavigate: (category: ResourceCategory, id?: string) => void;
  onConfigure: (id: string, section: SettingsSection) => void;
}) {
  const listing = usePollingResource('dataset-experiments', experimentApi.listExperiments, undefined, active);
  const ids = listing.data?.experiments;
  const selected = experimentId || (currentExperimentId && ids?.includes(currentExperimentId) ? currentExperimentId : ids?.[0]) || '';
  const valid = ids?.includes(selected);
  return <>
    <div className="resources-toolbar"><label>数据所属实验<Select value={selected} disabled={!ids?.length}
      onChange={event => onNavigate('datasets', event.target.value)}>
      {!valid && <option value={selected}>{selected || '选择实验'}</option>}
      {ids?.map(id => <option key={id} value={id}>{id}</option>)}
    </Select></label><Button size="sm" onClick={() => void listing.refresh()}>刷新列表</Button></div>
    {listing.error && <InlineNotice tone="danger">{listing.error}</InlineNotice>}
    {!ids && listing.loading && <LoadingState label="读取实验列表…" />}
    {ids?.length === 0 && <p className="resource-empty">先创建实验，再配置和查看数据。</p>}
    {ids && ids.length > 0 && !valid && <InlineNotice tone="warning">所选实验不存在，请重新选择。</InlineNotice>}
    {valid && <ExperimentData key={selected} experimentId={selected} active={active} onConfigure={onConfigure} />}
  </>;
}

export const Resources = memo(function Resources({ route, active, currentExperimentId, onNavigate, onConfigure, onReturn, onSelectModel, requestedType }: {
  route: ResourceRoute; active: boolean; currentExperimentId: string | null;
  onNavigate: (category: ResourceCategory, experimentId?: string) => void;
  onConfigure: (id: string, section: SettingsSection) => void; onReturn: () => void;
  onSelectModel: (id: string, resource: ModelResource) => void; requestedType?: ModelResourceType;
}) {
  const onUse = useCallback((resource: ModelResource) => {
    if (currentExperimentId) onSelectModel(currentExperimentId, resource);
  }, [currentExperimentId, onSelectModel]);
  const [dataVisited, setDataVisited] = useState(route.category === 'datasets');
  if (route.category === 'datasets' && !dataVisited) setDataVisited(true);
  return <div className="resources-page">
    <header className="resources-page-header"><div><h1>模型与数据</h1><p>个人模型配置一次，多个实验选择使用。数据暂保留路径与任务预览。</p></div>
      <Button size="sm" variant="ghost" onClick={onReturn}><ArrowLeft size={14} />{currentExperimentId ? '返回原实验' : '返回实验首页'}</Button></header>
    <div className="resources-toolbar"><div className="resources-tabs" role="group" aria-label="资源类别">
      <Button aria-pressed={route.category === 'models'} variant={route.category === 'models' ? 'primary' : 'ghost'} onClick={() => onNavigate('models', currentExperimentId || undefined)}><Cpu size={15} />个人模型</Button>
      <Button aria-pressed={route.category === 'datasets'} variant={route.category === 'datasets' ? 'primary' : 'ghost'} onClick={() => onNavigate('datasets', route.experimentId || currentExperimentId || undefined)}><Database size={15} />数据</Button>
    </div></div>
    <div hidden={route.category !== 'models'}>
      <ModelCatalog active={active && route.category === 'models'} returnExperimentId={currentExperimentId} onUse={onUse} requestedType={requestedType} />
    </div>
    {dataVisited && <div hidden={route.category !== 'datasets'}>
      <DatasetResources experimentId={route.experimentId} active={active && route.category === 'datasets'} currentExperimentId={currentExperimentId}
        onNavigate={onNavigate} onConfigure={onConfigure} />
    </div>}
  </div>;
});

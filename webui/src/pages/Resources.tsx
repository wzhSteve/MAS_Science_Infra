import { memo, useCallback, useEffect, useState } from 'react';
import { ArrowLeft, Cpu, Database } from 'lucide-react';
import type { ResourceCategory, ResourceRoute } from '../app/navigation';
import type { ModelResource, ModelResourceType } from '../features/resources/api';
import { ModelCatalog } from '../features/resources/components/ModelCatalog';
import { DatasetCatalog } from '../features/resources/components/DatasetCatalog';
import { Button } from '../shared/ui/button';

export const Resources = memo(function Resources({ route, active, currentExperimentId, onNavigate, onReturn, onSelectModel, requestedType }: {
  route: ResourceRoute; active: boolean; currentExperimentId: string | null;
  onNavigate: (category: ResourceCategory, experimentId?: string) => void;
  onReturn: () => void;
  onSelectModel: (id: string, resource: ModelResource) => void; requestedType?: ModelResourceType;
}) {
  const onUse = useCallback((resource: ModelResource) => {
    if (currentExperimentId) onSelectModel(currentExperimentId, resource);
  }, [currentExperimentId, onSelectModel]);
  const [dataVisited, setDataVisited] = useState(route.category === 'datasets');
  useEffect(() => { if (route.category === 'datasets') setDataVisited(true); }, [route.category]);
  return <div className="resources-page">
    <header className="resources-page-header"><div><h1>模型与数据</h1></div>
      <Button size="sm" variant="ghost" onClick={onReturn}><ArrowLeft size={14} />{currentExperimentId ? '返回原实验' : '返回实验首页'}</Button></header>
    <div className="resources-toolbar"><div className="resources-tabs" role="group" aria-label="资源类别">
      <Button aria-pressed={route.category === 'models'} variant={route.category === 'models' ? 'primary' : 'ghost'} onClick={() => onNavigate('models', currentExperimentId || undefined)}><Cpu size={15} />模型</Button>
      <Button aria-pressed={route.category === 'datasets'} variant={route.category === 'datasets' ? 'primary' : 'ghost'} onClick={() => onNavigate('datasets', currentExperimentId || undefined)}><Database size={15} />数据</Button>
    </div></div>
    <div hidden={route.category !== 'models'}>
      <ModelCatalog active={active && route.category === 'models'} returnExperimentId={currentExperimentId} onUse={onUse} requestedType={requestedType} />
    </div>
    {(dataVisited || route.category === 'datasets') && <div hidden={route.category !== 'datasets'}>
      <DatasetCatalog active={active && route.category === 'datasets'} />
    </div>}
  </div>;
});

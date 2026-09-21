import { lazy, memo, Suspense, useCallback, useEffect, useRef } from 'react';
import { Tooltip } from 'radix-ui';
import { ArrowLeft, Database, Network } from 'lucide-react';
import { experimentApi } from '../features/experiment/api';
import { usePollingResource } from '../shared/hooks/usePollingResource';
import { LoadingState } from '../shared/components/LoadingState';
import { InlineNotice } from '../shared/components/InlineNotice';
import { Button } from '../shared/ui/button';
import { WorkspaceMenu } from './layout/WorkspaceMenu';
import { TrainingBanner } from './layout/TrainingBanner';
import { StatusBar } from './layout/StatusBar';
import { RuntimeProvider } from './providers/RuntimeProvider';
import { NavigationGuardProvider } from './providers/NavigationGuard';
import { PageBoundary } from './layout/PageBoundary';
import { ExperimentHome } from '../pages/ExperimentHome';
import { useHashNavigation } from './useHashNavigation';
import { NAVIGATION, isCanvasPanel, workspaceRoute, type PanelId, type ResourceCategory } from './navigation';
import type { SettingsSection } from '../features/settings/model/sections';
import type { ModelResource } from '../features/resources/api';
import type { MetaResponse } from '../shared/api/types';

const WorkspacePanels = lazy(() => import('./layout/WorkspacePanels').then(module => ({ default: module.WorkspacePanels })));
const Resources = lazy(() => import('../pages/Resources').then(module => ({ default: module.Resources })));

const Workspace = memo(function Workspace({ expId, tab, visible, meta, setExpId, onWorkspace, onHome, onChangePanel, settings, onResources, onSettings, selectedResource }: {
  expId: string; tab: PanelId; visible: boolean; meta: MetaResponse | null; setExpId: (id: string) => void;
  onWorkspace: () => void; onHome: () => void; onChangePanel: (id: PanelId) => void;
  settings?: SettingsSection; onResources: (category: ResourceCategory) => void;
  onSettings: (section: SettingsSection) => void;
  selectedResource?: string;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    // Legacy bundle reads ensure missing experiments; deep links must never create one.
    const { experiments } = await experimentApi.listExperiments(signal);
    if (!experiments.includes(expId)) throw new Error('实验不存在，请返回首页创建或选择已有实验。');
    return experimentApi.getExperiment(expId, signal);
  }, [expId]);
  const { data: bundle, error, loading, refresh } = usePollingResource(`experiment:${expId}`, load);
  const wasVisible = useRef(visible);
  useEffect(() => {
    if (visible && !wasVisible.current) void refresh();
    wasVisible.current = visible;
  }, [visible, refresh]);
  if (!bundle || bundle.id !== expId) {
    return <div className="workspace-content">{loading ? <LoadingState label={`加载实验 ${expId}…`} /> : <>
      <InlineNotice tone="danger">{error || '实验响应与当前实验不匹配'}</InlineNotice>
      <div className="navigation-recovery"><Button onClick={() => void refresh()}>重试</Button><Button onClick={onHome}>返回实验首页</Button></div>
    </>}</div>;
  }
  return <RuntimeProvider expId={expId} onReload={refresh} active={visible}>
    <div className="experiment-session-bar">
      <Button size="sm" variant="ghost" onClick={onHome}><ArrowLeft size={14} />实验首页</Button>
      <span className="experiment-session-divider" />
      <Network size={14} aria-hidden="true" />
      <strong title={bundle.meta.name || bundle.id}>{bundle.meta.name || bundle.id}</strong>
      {bundle.meta.name && bundle.meta.name !== bundle.id && <span className="experiment-session-id mono">{bundle.id}</span>}
      {!isCanvasPanel(tab) && <span className="experiment-session-location">{NAVIGATION.find(item => item.id === tab)?.description}</span>}
      <div className="workspace-navigation"><Button size="sm" variant="ghost" onClick={() => onResources('models')}><Database size={14} />资源</Button>
        <WorkspaceMenu active={tab} onChange={onChangePanel} /></div>
    </div>
    <TrainingBanner />
    {error && <div className="runtime-error"><InlineNotice tone="warning">实验刷新失败，草稿保持不变：{error}</InlineNotice></div>}
    <PageBoundary><Suspense fallback={<LoadingState label="加载实验工作区…" />}>
      <WorkspacePanels active={tab} visible={visible} bundle={bundle} meta={meta} onReload={refresh} setExpId={setExpId} onWorkspace={onWorkspace}
        settings={settings} onResources={onResources} onSettings={onSettings} selectedResource={selectedResource} />
    </Suspense></PageBoundary>
    {!isCanvasPanel(tab) && <StatusBar />}
  </RuntimeProvider>;
});

function Application() {
  const { route, workspace, resources, navigate } = useHashNavigation();
  const expId = workspace?.experimentId ?? null;
  const visible = route.kind === 'workspace';
  const onHome = useCallback(() => { void navigate({ kind: 'home' }); }, [navigate]);
  const openExperiment = useCallback((id: string) => { void navigate(workspaceRoute(id)); }, [navigate]);
  const changePanel = useCallback((panel: PanelId) => {
    if (expId) void navigate(workspaceRoute(expId, panel));
  }, [expId, navigate]);
  const returnToWorkspace = useCallback(() => changePanel('mas'), [changePanel]);
  const browseResources = useCallback((category: ResourceCategory, experimentId?: string) => {
    void navigate({ kind: 'resources', category, experimentId });
  }, [navigate]);
  const openResources = useCallback((category: ResourceCategory) => browseResources(category, expId || undefined), [browseResources, expId]);
  const configureResource = useCallback((id: string, section: SettingsSection) => {
    void navigate(workspaceRoute(id, 'mas', section));
  }, [navigate]);
  const openSettings = useCallback((section: SettingsSection) => {
    if (expId) configureResource(expId, section);
  }, [expId, configureResource]);
  const selectModelResource = useCallback((id: string, resource: ModelResource) => {
    void navigate({ ...workspaceRoute(id, 'mas', resource.type === 'inference' ? 'model' : 'training'), selectedResource: resource.id });
  }, [navigate]);
  const returnFromResources = useCallback(() => {
    void navigate(workspace || { kind: 'home' });
  }, [workspace, navigate]);
  const meta = usePollingResource('meta', experimentApi.meta, undefined, visible);
  const content = useRef<HTMLElement>(null);
  useEffect(() => { content.current?.focus({ preventScroll: true }); }, [route]);
  return <>
    <a className="skip-link" href="#app-content" onClick={event => {
      event.preventDefault();
      content.current?.focus({ preventScroll: true });
    }}>跳到主要内容</a>
    <div className="app-shell app-shell--product">
      <main ref={content} id="app-content" className="workspace-main" tabIndex={-1}>
        <section className="experiment-home-host" hidden={route.kind !== 'home' && route.kind !== 'resources'} aria-label="实验与资源">
          <header className="studio-header">
            <div className="brand"><span className="brand-mark"><Network size={19} strokeWidth={1.6} /></span>
              <div>Science Studio<span>MAS / RL RESEARCH</span></div></div>
            <nav className="studio-product-nav" aria-label="研究空间导航">
              <Button size="sm" variant="ghost" aria-current={route.kind === 'home' ? 'page' : undefined} onClick={onHome}>实验</Button>
              <Button size="sm" variant="ghost" aria-current={route.kind === 'resources' ? 'page' : undefined} onClick={() => openResources('models')}>模型与数据</Button>
            </nav>
          </header>
          <div hidden={route.kind !== 'home'}><PageBoundary><ExperimentHome active={route.kind === 'home'} currentExperimentId={expId} onOpenExperiment={openExperiment} /></PageBoundary></div>
          {resources && <div hidden={route.kind !== 'resources'}><PageBoundary><Suspense fallback={<LoadingState label="加载资源页面…" />}>
            <Resources route={resources} active={route.kind === 'resources'} currentExperimentId={expId}
              onNavigate={browseResources} onConfigure={configureResource} onReturn={returnFromResources}
              onSelectModel={selectModelResource} requestedType={workspace?.panel === 'rl' || workspace?.settings === 'training' || workspace?.settings === 'environment' ? 'training' : 'inference'} />
          </Suspense></PageBoundary></div>}
        </section>
        {workspace && <section className="experiment-session" hidden={!visible} aria-label={`实验 ${workspace.experimentId}`}>
          {meta.error && visible && <div className="runtime-error"><InlineNotice tone="warning">配置选项加载失败：{meta.error}<Button size="sm" onClick={() => void meta.refresh()}>重试</Button></InlineNotice></div>}
          <Workspace key={workspace.experimentId} expId={workspace.experimentId} tab={workspace.panel} visible={visible} meta={meta.data}
            setExpId={openExperiment} onWorkspace={returnToWorkspace} onHome={onHome} onChangePanel={changePanel}
            settings={workspace.settings} onResources={openResources} onSettings={openSettings} selectedResource={workspace.selectedResource} />
        </section>}
        {route.kind === 'not-found' && <section className="workspace-content">
          <div className="page-stack settings-page">
            <h1>找不到这个页面</h1>
            <p className="field-hint">地址可能不完整，或对应页面尚未开放。当前实验的编辑会话仍然保留。</p>
            <div className="navigation-recovery">
              <Button variant="primary" onClick={onHome}>返回实验首页</Button>
              {expId && <Button onClick={() => openExperiment(expId)}>返回当前实验</Button>}
            </div>
          </div>
        </section>}
      </main>
    </div>
  </>;
}

export default function App() {
  return <Tooltip.Provider delayDuration={250}>
    <NavigationGuardProvider><Application /></NavigationGuardProvider>
  </Tooltip.Provider>;
}

import { lazy, memo, Suspense, useCallback, useEffect, useRef } from 'react';
import { Tooltip } from 'radix-ui';
import { ArrowLeft, Network } from 'lucide-react';
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
import { NAVIGATION, isCanvasPanel, workspaceRoute, type PanelId } from './navigation';
import type { MetaResponse } from '../shared/api/types';

const WorkspacePanels = lazy(() => import('./layout/WorkspacePanels').then(module => ({ default: module.WorkspacePanels })));

const Workspace = memo(function Workspace({ expId, tab, visible, meta, setExpId, onWorkspace, onHome, onChangePanel }: {
  expId: string; tab: PanelId; visible: boolean; meta: MetaResponse | null; setExpId: (id: string) => void;
  onWorkspace: () => void; onHome: () => void; onChangePanel: (id: PanelId) => void;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    // Legacy bundle reads ensure missing experiments; deep links must never create one.
    const { experiments } = await experimentApi.listExperiments(signal);
    if (!experiments.includes(expId)) throw new Error('实验不存在，请返回首页创建或选择已有实验。');
    return experimentApi.getExperiment(expId, signal);
  }, [expId]);
  const { data: bundle, error, loading, refresh } = usePollingResource(`experiment:${expId}`, load);
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
      <WorkspaceMenu active={tab} onChange={onChangePanel} />
    </div>
    <TrainingBanner />
    {error && <div className="runtime-error"><InlineNotice tone="warning">实验刷新失败，草稿保持不变：{error}</InlineNotice></div>}
    <PageBoundary><Suspense fallback={<LoadingState label="加载实验工作区…" />}>
      <WorkspacePanels active={tab} visible={visible} bundle={bundle} meta={meta} onReload={refresh} setExpId={setExpId} onWorkspace={onWorkspace} />
    </Suspense></PageBoundary>
    {!isCanvasPanel(tab) && <StatusBar />}
  </RuntimeProvider>;
});

function Application() {
  const { route, workspace, navigate } = useHashNavigation();
  const expId = workspace?.experimentId ?? null;
  const visible = route.kind === 'workspace';
  const onHome = useCallback(() => { void navigate({ kind: 'home' }); }, [navigate]);
  const openExperiment = useCallback((id: string) => { void navigate(workspaceRoute(id)); }, [navigate]);
  const changePanel = useCallback((panel: PanelId) => {
    if (expId) void navigate(workspaceRoute(expId, panel));
  }, [expId, navigate]);
  const returnToWorkspace = useCallback(() => changePanel('mas'), [changePanel]);
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
        <section className="experiment-home-host" hidden={route.kind !== 'home'} aria-label="实验首页">
          <header className="studio-header">
            <div className="brand"><span className="brand-mark"><Network size={19} strokeWidth={1.6} /></span>
              <div>Science Studio<span>MAS / RL RESEARCH</span></div></div>
            <span className="studio-header-section">实验空间</span>
          </header>
          <PageBoundary><ExperimentHome active={route.kind === 'home'} currentExperimentId={expId} onOpenExperiment={openExperiment} /></PageBoundary>
        </section>
        {workspace && <section className="experiment-session" hidden={!visible} aria-label={`实验 ${workspace.experimentId}`}>
          {meta.error && visible && <div className="runtime-error"><InlineNotice tone="warning">配置选项加载失败：{meta.error}<Button size="sm" onClick={() => void meta.refresh()}>重试</Button></InlineNotice></div>}
          <Workspace key={workspace.experimentId} expId={workspace.experimentId} tab={workspace.panel} visible={visible} meta={meta.data}
            setExpId={openExperiment} onWorkspace={returnToWorkspace} onHome={onHome} onChangePanel={changePanel} />
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

import { Component, memo, useCallback, useState, type ErrorInfo, type ReactNode } from 'react';
import { Tooltip } from 'radix-ui';
import { experimentApi } from '../features/experiment/api';
import { usePollingResource } from '../shared/hooks/usePollingResource';
import { LoadingState } from '../shared/components/LoadingState';
import { InlineNotice } from '../shared/components/InlineNotice';
import { Button } from '../shared/ui/button';
import { Sidebar } from './layout/Sidebar';
import { WorkspaceHeader } from './layout/WorkspaceHeader';
import { TrainingBanner } from './layout/TrainingBanner';
import { StatusBar } from './layout/StatusBar';
import { WorkspacePanels } from './layout/WorkspacePanels';
import { RuntimeProvider } from './providers/RuntimeProvider';
import type { PanelId } from './navigation';
import type { MetaResponse } from '../shared/api/types';

class WorkspaceBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error: error.message }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Workspace rendering failed', error, info); }
  render() {
    return this.state.error
      ? <div className="workspace-content"><InlineNotice tone="danger">界面加载失败：{this.state.error}</InlineNotice><Button onClick={() => window.location.reload()}>重新加载页面</Button></div>
      : this.props.children;
  }
}

const Workspace = memo(function Workspace({ expId, tab, meta, setExpId }: {
  expId: string; tab: PanelId; meta: MetaResponse | null; setExpId: (id: string) => void;
}) {
  const load = useCallback((signal: AbortSignal) => experimentApi.getExperiment(expId, signal), [expId]);
  const { data: bundle, error, loading, refresh } = usePollingResource(`experiment:${expId}`, load);
  if (!bundle || bundle.id !== expId) {
    return <div className="workspace-content">{loading ? <LoadingState label={`加载实验 ${expId}…`} /> : <>
      <InlineNotice tone="danger">{error || '实验响应与当前实验不匹配'}</InlineNotice><Button onClick={() => void refresh()}>重试</Button>
    </>}</div>;
  }
  return <RuntimeProvider expId={expId} onReload={refresh}>
    {tab !== 'mas' && <WorkspaceHeader bundle={bundle} onReload={refresh} />}
    <TrainingBanner />
    {error && <div className="runtime-error"><InlineNotice tone="warning">实验刷新失败，草稿保持不变：{error}</InlineNotice></div>}
    <WorkspaceBoundary><WorkspacePanels active={tab} bundle={bundle} meta={meta} onReload={refresh} setExpId={setExpId} /></WorkspaceBoundary>
    <StatusBar />
  </RuntimeProvider>;
});

export default function App() {
  const [tab, setTab] = useState<PanelId>('mas');
  const [expId, setExpId] = useState('demo');
  const meta = usePollingResource('meta', experimentApi.meta);
  return <Tooltip.Provider delayDuration={250}>
    <a className="skip-link" href="#workspace-content">跳到工作区</a>
    <div className="app-shell">
      <Sidebar active={tab} onChange={setTab} />
      <main className="workspace-main">
        {meta.error && <div className="runtime-error"><InlineNotice tone="warning">配置选项加载失败：{meta.error}<Button size="sm" onClick={() => void meta.refresh()}>重试</Button></InlineNotice></div>}
        <Workspace key={expId} expId={expId} tab={tab} meta={meta.data} setExpId={setExpId} />
      </main>
    </div>
  </Tooltip.Provider>;
}

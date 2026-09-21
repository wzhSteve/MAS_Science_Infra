import { memo, useCallback, useLayoutEffect, useRef, useState } from 'react';
import { ArrowRight, FlaskConical, Network, Plus, RefreshCw, Search } from 'lucide-react';
import type { Bundle } from '../shared/api/types';
import { useExperimentCatalog, type ExperimentCardState } from '../features/experiment/model/useExperimentCatalog';
import { CreateExperimentDialog } from '../features/experiment/components/home/CreateExperimentDialog';
import { Button } from '../shared/ui/button';
import { Input } from '../shared/ui/input';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { StatusBadge } from '../shared/components/StatusBadge';

const ExperimentCard = memo(function ExperimentCard({ id, state, current, onOpen, onRetry }: {
  id: string; state?: ExperimentCardState; current: boolean;
  onOpen: (id: string) => void; onRetry: (id: string) => void;
}) {
  const summary = state?.summary;
  return <article className={`experiment-card${current ? ' is-current' : ''}`}>
    <div className="experiment-card-top"><span className="experiment-card-icon"><Network size={20} /></span>
      {current && <StatusBadge tone="neutral">当前编辑会话</StatusBadge>}
    </div>
    <h2 title={summary?.name || id}>{summary?.name || id}</h2>
    <p className="experiment-card-id mono">ID · {id}</p>
    <div className="experiment-card-structure">
      {summary ? <><span>{summary.agentCount} 个 Agent</span><span>{summary.toolCount} 个工具</span>
        <span className="experiment-card-topology" title={summary.topology}>{summary.topology === 'graph' ? '图编排' : ['hub_react', 'single'].includes(summary.topology) ? '单 Agent 工作流' : summary.topology}</span></>
        : <span>{state?.error ? '结构摘要读取失败' : '正在读取结构摘要…'}</span>}
    </div>
    {state?.error && <div className="experiment-card-error">
      <p>{state.error}</p><Button size="sm" variant="ghost" onClick={() => onRetry(id)}>重试摘要</Button>
    </div>}
    <div className="experiment-card-footer">
      <span className="field-hint" title="这里只检查设计结构，不代表模型或训练环境已就绪。">{state?.loading && summary ? '正在刷新摘要…' : summary ? summary.executable ? '结构可执行' : '结构待完善 · 仍可编辑' : '不影响打开实验'}</span>
      <Button size="sm" onClick={() => onOpen(id)}>{current ? '返回工作区' : '打开工作区'}<ArrowRight size={13} /></Button>
    </div>
  </article>;
});

export const ExperimentHome = memo(function ExperimentHome({ active, currentExperimentId, onOpenExperiment }: {
  active: boolean; currentExperimentId: string | null; onOpenExperiment: (id: string) => void;
}) {
  const catalog = useExperimentCatalog(active);
  const [creating, setCreating] = useState(false);
  const [createdMessage, setCreatedMessage] = useState('');
  const activeRef = useRef(active);
  useLayoutEffect(() => { activeRef.current = active; }, [active]);
  const onCreated = useCallback((bundle: Bundle) => {
    catalog.created(bundle);
    setCreatedMessage(`已创建“${bundle.meta.name || bundle.id}”。如果取消切换，仍可从首页打开新实验。`);
    if (activeRef.current) onOpenExperiment(bundle.id);
  }, [catalog.created, onOpenExperiment]);

  return <div className="experiment-home">
    <header className="experiment-home-header">
      <div><h1>我的实验</h1>
        <p>从一个研究问题开始，设计你的 Agent 协作。</p></div>
      <Button variant="primary" onClick={() => setCreating(true)}><Plus size={16} />新建实验</Button>
    </header>
    {currentExperimentId && <div className="experiment-home-resume">
      <div><Network size={16} /><span>编辑会话已保留</span><code>{currentExperimentId}</code></div>
      <Button size="sm" variant="ghost" onClick={() => onOpenExperiment(currentExperimentId)}>继续编辑<ArrowRight size={13} /></Button>
    </div>}
    <div className="experiment-home-toolbar">
      <label className="experiment-home-search"><Search size={16} aria-hidden="true" />
        <Input aria-label="按实验 ID 搜索" placeholder="按实验 ID 搜索…" value={catalog.query} onChange={event => catalog.setQuery(event.target.value)} />
      </label>
      <span className="field-hint">{catalog.loaded ? `${catalog.total} 个实验` : '读取实验列表'}</span>
      <Button size="sm" variant="ghost" loading={catalog.loading} onClick={catalog.refresh}><RefreshCw size={14} />刷新</Button>
    </div>
    {createdMessage && <InlineNotice tone="success">{createdMessage}</InlineNotice>}
    {catalog.error && <InlineNotice tone="danger">实验列表加载失败：{catalog.error}
      <Button size="sm" onClick={catalog.refresh}>重试列表</Button>
    </InlineNotice>}
    {catalog.loading && !catalog.loaded && <LoadingState label="读取实验…" />}
    {catalog.pageIds.length > 0 && <div className="experiment-home-grid" aria-label="实验列表">
      {catalog.pageIds.map(id => <ExperimentCard key={id} id={id} state={catalog.cards[id]}
        current={currentExperimentId === id} onOpen={onOpenExperiment} onRetry={catalog.retryCard} />)}
    </div>}
    {catalog.loaded && !catalog.loading && !catalog.error && catalog.pageIds.length === 0 && <section className="experiment-home-empty">
      <FlaskConical size={30} strokeWidth={1.4} /><h2>{catalog.total ? '没有匹配的实验' : '从一个实验开始'}</h2>
      <p>{catalog.total ? '尝试其他实验 ID，或清除搜索条件。' : '先创建研究方案，再逐步配置模型、工具和训练。'}</p>
      {catalog.total ? <Button onClick={() => catalog.setQuery('')}>清除搜索</Button>
        : <Button variant="primary" onClick={() => setCreating(true)}><Plus size={15} />创建第一个实验</Button>}
    </section>}
    {catalog.pageCount > 1 && <nav className="experiment-home-pagination" aria-label="实验列表分页">
      <span>第 {catalog.page + 1} / {catalog.pageCount} 页 · {catalog.matchedCount} 个匹配</span>
      <div><Button size="sm" disabled={catalog.page === 0} onClick={() => catalog.setPage(catalog.page - 1)}>上一页</Button>
        <Button size="sm" disabled={catalog.page + 1 >= catalog.pageCount} onClick={() => catalog.setPage(catalog.page + 1)}>下一页</Button></div>
    </nav>}
    <footer className="experiment-home-note">
      打开实验进入画布，通过“实验设置”配置模型、训练和诊断；其他入口保留在“更多功能”中。
    </footer>
    <CreateExperimentDialog open={active && creating} onOpenChange={setCreating} onCreated={onCreated} />
  </div>;
});

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, type Bundle } from './api/client';
import { ExperimentPanel } from './pages/Experiment';
import { LLMPanel } from './pages/LLM';
import { MASPanel } from './pages/MAS';
import { RLPanel } from './pages/RL';
import { HarnessPanel } from './pages/Harness';
import { MonitorPanel } from './pages/Monitor';
import GpuPicker from './features/gpu/GpuPicker';
import './App.css';

const NAV = [
  { id: 'experiment', label: 'Experiment' },
  { id: 'llm', label: 'LLM' },
  { id: 'mas', label: 'MAS' },
  { id: 'rl', label: 'RL' },
  { id: 'harness', label: 'Harness' },
  { id: 'monitor', label: 'Monitor' },
] as const;

type Tab = (typeof NAV)[number]['id'];

function contractSummary(bundle: Bundle | null): string {
  const wf = bundle?.workflow || {};
  const entry = String(wf.entry_agent || 'hub');
  const agents = Array.isArray(wf.agents) ? wf.agents : [];
  const trainable = agents
    .filter((a: any) => a && a.trainable !== false)
    .map((a: any) => a.id)
    .join(',') || entry;
  const gpus = ((bundle?.rl?.devices?.ids as number[]) || [0]).join(',');
  const n =
    bundle?.rl?.rollout_per_gpu ?? bundle?.rl?.actor_rollout_ref?.rollout?.n ?? 2;
  const runners = bundle?.rl?.n_runners ?? 1;
  return `入口=${entry} · 可训=[${trainable}] · GPU=${gpus} · n=${n} · runners=${runners}`;
}

export default function App() {
  const [tab, setTab] = useState<Tab>('mas');
  const [expId, setExpId] = useState('demo');
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [meta, setMeta] = useState<any>(null);
  const [status, setStatus] = useState('idle');
  const [events, setEvents] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [trainRunId, setTrainRunId] = useState<string | null>(null);
  const [trainRunning, setTrainRunning] = useState(false);
  const [trainLog, setTrainLog] = useState('');
  const [aglOnline, setAglOnline] = useState(false);
  const [meanReward, setMeanReward] = useState<string | number | null>(null);

  const reload = useCallback(async () => {
    try {
      const b = await api.getExperiment(expId);
      setBundle(b);
    } catch (e) {
      console.error(e);
    }
  }, [expId]);

  const refreshTrain = useCallback(async () => {
    try {
      const r = await api.runs(expId);
      const trains = (r.runs || []).filter((x: any) => x.kind === 'train');
      const active = trains.find((x: any) => x.running);
      const latest = active || trains.sort((a: any, b: any) => (b.started_at || 0) - (a.started_at || 0))[0];
      if (!latest) {
        setTrainRunning(false);
        return;
      }
      setTrainRunId(latest.run_id);
      setTrainRunning(!!latest.running);
      const d = await api.run(latest.run_id);
      setTrainLog(String(d.log_tail || ''));
    } catch {
      /* ignore */
    }
  }, [expId]);

  const refreshAgl = useCallback(async () => {
    try {
      const h = await api.aglHealth();
      setAglOnline(!!h.ok);
    } catch {
      setAglOnline(false);
    }
  }, []);

  useEffect(() => {
    api.meta().then(setMeta).catch(console.error);
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  useEffect(() => {
    refreshTrain();
    const t = setInterval(refreshTrain, 4000);
    return () => clearInterval(t);
  }, [refreshTrain]);

  useEffect(() => {
    refreshAgl();
    const t = setInterval(refreshAgl, 5000);
    return () => clearInterval(t);
  }, [refreshAgl, trainRunning]);

  useEffect(() => {
    api
      .monitor(expId)
      .then((m) => setMeanReward(m.mean_reward ?? null))
      .catch(() => setMeanReward(null));
  }, [expId, status]);

  useEffect(() => {
    const es = new EventSource(`/api/events?experiment_id=${encodeURIComponent(expId)}`);
    es.onmessage = (ev) => {
      try {
        const p = JSON.parse(ev.data);
        const t = String(p.type || '');
        setStatus(t || 'event');
        setEvents((xs) => [t, ...xs].slice(0, 8));
        if (t.includes('train')) refreshTrain();
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  }, [expId, refreshTrain]);

  useEffect(() => {
    if (tab === 'mas' || tab === 'monitor') {
      const id = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
      return () => cancelAnimationFrame(id);
    }
  }, [tab]);

  const gpuIds: number[] = ((bundle?.rl?.devices?.ids as number[]) || [0]).map((x) => Number(x));

  const saveGpus = async (ids: number[]) => {
    if (!bundle) return;
    const rl = {
      ...bundle.rl,
      devices: { ids },
      trainer: { ...(bundle.rl.trainer || {}), n_gpus_per_node: ids.length },
    };
    await api.putSection(expId, 'rl', rl);
    reload();
  };

  const startTrain = async () => {
    if (!window.confirm('启动训练将停止本地 LLM（若在跑）。确认？')) return;
    try {
      setBusy(true);
      const r = await api.train(expId, { stop_llm: true, confirm_gpu: true });
      setTrainRunId(r.run_id);
      setTrainRunning(true);
      setStatus('train_started');
      await refreshTrain();
    } catch (e) {
      setStatus(String(e));
    } finally {
      setBusy(false);
    }
  };

  const stopTrain = async () => {
    try {
      await api.trainStop(expId);
      setTrainRunning(false);
      setStatus('train_stopped');
      await refreshTrain();
    } catch (e) {
      setStatus(String(e));
    }
  };

  const logTail = useMemo(() => {
    const lines = trainLog.split('\n').filter(Boolean);
    return lines.slice(-2).join(' · ');
  }, [trainLog]);

  return (
    <div className="layout">
      <aside className="nav">
        <div className="brand">
          Science Studio
          <span>零代码 MAS / RL</span>
        </div>
        {NAV.map((n) => (
          <button
            key={n.id}
            type="button"
            className={tab === n.id ? 'nav-item active' : 'nav-item'}
            onClick={() => setTab(n.id)}
          >
            {n.label}
          </button>
        ))}
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="topbar-meta">
            <div>
              <strong>{expId}</strong>
              <span className="muted" style={{ marginLeft: 8 }}>
                seed={bundle?.meta?.seed ?? '—'}
              </span>
            </div>
            <span className="muted">{contractSummary(bundle)}</span>
          </div>
          <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <GpuPicker compact selected={gpuIds} onChange={(ids) => saveGpus(ids)} />
            <button
              type="button"
              disabled={busy || bundle?.executable?.ok === false}
              onClick={async () => {
                try {
                  setBusy(true);
                  await api.collect(expId, { mock: true, n: 1 });
                  setStatus('collect_done');
                  reload();
                } catch (e) {
                  setStatus(String(e));
                } finally {
                  setBusy(false);
                }
              }}
              title={bundle?.executable?.ok === false ? String(bundle.executable.reason || 'not executable') : ''}
            >
              Collect
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={async () => {
                try {
                  setBusy(true);
                  await api.diagnose(expId);
                  setStatus('diagnose_done');
                  reload();
                } catch (e) {
                  setStatus(String(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              Diagnose
            </button>
            <button type="button" className="primary" disabled={busy || trainRunning} onClick={startTrain}>
              Train
            </button>
            <span className={`chip ${trainRunning ? 'ok' : ''}`}>{trainRunning ? 'train running' : status || 'idle'}</span>
          </div>
        </header>
        {trainRunning ? (
          <div className="train-banner">
            <span className="chip ok">训练中</span>
            <span className="muted">run={trainRunId}</span>
            <button type="button" className="danger" onClick={stopTrain}>
              Stop
            </button>
            {aglOnline ? (
              <a href="/agl/metrics" target="_blank" rel="noreferrer">
                <button type="button" className="primary">
                  AGL Metrics
                </button>
              </a>
            ) : (
              <button type="button" disabled title="LightningStore 仅在训练进程存活时可用">
                Metrics 未就绪
              </button>
            )}
          </div>
        ) : null}
        <div className="content">
          <div className={tab === 'experiment' ? 'pane' : 'pane pane-hidden'}>
            <ExperimentPanel expId={expId} bundle={bundle} onReload={reload} setExpId={setExpId} />
          </div>
          <div className={tab === 'llm' ? 'pane' : 'pane pane-hidden'}>
            <LLMPanel expId={expId} bundle={bundle} onReload={reload} />
          </div>
          <div className={tab === 'mas' ? 'pane' : 'pane pane-hidden'}>
            <MASPanel expId={expId} bundle={bundle} onReload={reload} />
          </div>
          <div className={tab === 'rl' ? 'pane' : 'pane pane-hidden'}>
            <RLPanel
              expId={expId}
              bundle={bundle}
              meta={meta}
              onReload={reload}
              trainRunId={trainRunId}
              trainRunning={trainRunning}
              trainLog={trainLog}
              aglOnline={aglOnline}
              onStartTrain={startTrain}
              onStopTrain={stopTrain}
              onRefreshLog={refreshTrain}
            />
          </div>
          <div className={tab === 'harness' ? 'pane' : 'pane pane-hidden'}>
            <HarnessPanel expId={expId} bundle={bundle} meta={meta} onReload={reload} />
          </div>
          <div className={tab === 'monitor' ? 'pane' : 'pane pane-hidden'}>
            <MonitorPanel
              expId={expId}
              trainRunId={trainRunId}
              trainRunning={trainRunning}
              trainLog={trainLog}
              aglOnline={aglOnline}
              onRefreshLog={refreshTrain}
              visible={tab === 'monitor'}
            />
          </div>
        </div>
        <footer className="status-strip">
          <span className={trainRunning ? 'chip ok' : 'chip'}>{trainRunning ? 'running' : 'idle'}</span>
          <span>reward={meanReward ?? '—'}</span>
          <span className="tail" title={logTail}>
            {logTail || events[0] || '等待事件…'}
          </span>
          {trainRunning ? (
            <button type="button" className="danger" onClick={stopTrain}>
              Stop
            </button>
          ) : null}
        </footer>
      </main>
    </div>
  );
}

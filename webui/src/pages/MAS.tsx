import { useEffect, useRef, useState } from 'react';
import { api, type Bundle } from '../api/client';
import MasGraphEditor from '../features/graph/MasGraphEditor';
import type { WorkflowSpec } from '../features/graph/workflowGraph';
import RolloutSamplingPanel from '../features/sampling/RolloutSamplingPanel';

type Props = { expId: string; bundle: Bundle | null; onReload: () => void };

export function MASPanel({ expId, bundle, onReload }: Props) {
  const [wf, setWf] = useState<WorkflowSpec | null>(null);
  const [rl, setRl] = useState<any>(null);
  const [palette, setPalette] = useState<any>({
    skills: [],
    roles: [],
    tools: [],
    edge_kinds: [],
    sampling_modes: ['grpo_n', 'arpo', 'aepo', 'appo', 'rae'],
    templates: [],
  });
  const [msg, setMsg] = useState('');
  const [n, setN] = useState(1);
  const [algo, setAlgo] = useState('grpo');
  const [parquet, setParquet] = useState('data/val.parquet');
  const [dataN, setDataN] = useState(5);
  const [source, setSource] = useState('gsm8k');
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<any[]>([]);
  const [wfDirty, setWfDirty] = useState(false);
  const [rlDirty, setRlDirty] = useState(false);
  const synced = useRef<string | null>(null);

  useEffect(() => {
    api.palette().then(setPalette).catch((e) => setMsg(String(e)));
  }, []);

  useEffect(() => {
    if (!bundle?.workflow) return;
    if (synced.current === expId) return;
    setWf(bundle.workflow as WorkflowSpec);
    setRl(bundle.rl ? structuredClone(bundle.rl) : {});
    setWfDirty(false);
    setRlDirty(false);
    synced.current = expId;
  }, [expId, bundle]);

  if (!wf) return <div className="card">加载 workflow…</div>;

  const save = async () => {
    try {
      await api.putWorkflow(expId, wf as unknown as Record<string, unknown>);
      setWfDirty(false);
      setMsg('saved workflow.yaml');
      onReload();
    } catch (e) {
      setMsg(String(e));
    }
  };

  const saveRl = async () => {
    if (!rl) return;
    try {
      const payload = { ...rl };
      for (const k of Object.keys(payload)) {
        if (k.includes('.')) delete payload[k];
      }
      const sel = (payload.devices?.ids || [0]).map((x: any) => Number(x));
      payload.devices = { ids: sel };
      payload.trainer = { ...(payload.trainer || {}), n_gpus_per_node: sel.length };
      if (payload.rollout_per_gpu != null) {
        payload.actor_rollout_ref = {
          ...(payload.actor_rollout_ref || {}),
          rollout: {
            ...((payload.actor_rollout_ref || {}).rollout || {}),
            n: Number(payload.rollout_per_gpu),
          },
        };
      }
      await api.putSection(expId, 'rl', payload);
      setRl(payload);
      setRlDirty(false);
      setMsg('saved rl.yaml（训练简参）');
      onReload();
    } catch (e) {
      setMsg(String(e));
    }
  };

  const collect = async (mock: boolean) => {
    try {
      setBusy(true);
      await save();
      const r = await api.collect(expId, { mock, n, algo });
      setRows(r.rows || []);
      setMsg(`collect done n=${r.n} mean_reward=${r.mean_reward}`);
      onReload();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  const collectParquet = async () => {
    try {
      setBusy(true);
      await save();
      const r = await api.collect(expId, {
        mock: false,
        n: 1,
        algo,
        parquet,
        data_n: dataN,
        source,
        sequential: true,
      });
      setRows(r.rows || []);
      const ok = (r.rows || []).filter((x: any) => x.tool && Number(x.reward) > 0).length;
      setMsg(`live-api-data via UI: ${ok}/${r.n} pass · mean_reward=${r.mean_reward} · ${r.path}`);
      onReload();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0 }}>MAS · Workflow</h2>
          <div className="row">
            {wfDirty ? <span className="chip warn">图未保存</span> : null}
            {rlDirty ? <span className="chip warn">训练简参未保存</span> : null}
          </div>
        </div>
        <p className="muted">
          拖拽 Agent / Tool 并连线；序列化为 WorkflowSpec。采集入口=
          {String((wf as any).entry_agent || 'hub')} · topology=
          <code>{wf.topology}</code>
          。每题采样条数在 Inspector 训练简参 / RL 页，不要和采集入口混用。
        </p>
        <MasGraphEditor
          workflow={wf}
          palette={palette}
          onChange={(next) => {
            setWf(next);
            setWfDirty(true);
          }}
          rl={rl}
          onRlPatch={(patch) => {
            setRl((prev: any) => ({ ...(prev || {}), ...patch }));
            setRlDirty(true);
          }}
          onRlSave={saveRl}
        />
        <RolloutSamplingPanel
          workflow={wf}
          samplingModes={palette.sampling_modes}
          onChange={(next) => {
            setWf(next);
            setWfDirty(true);
          }}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <button type="button" className="primary" onClick={save} disabled={busy}>
            保存 workflow.yaml
          </button>
          <label className="muted">采集条数</label>
          <input
            type="number"
            value={n}
            min={1}
            style={{ width: 70 }}
            onChange={(e) => setN(Number(e.target.value))}
          />
          <label className="muted">algo</label>
          <input value={algo} style={{ width: 100 }} onChange={(e) => setAlgo(e.target.value)} />
          <button type="button" onClick={() => collect(true)} disabled={busy || !bundle?.executable?.ok}>
            Collect (mock)
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => collect(false)}
            disabled={busy || !bundle?.executable?.ok}
          >
            Collect (live)
          </button>
        </div>
        {msg ? <p className="muted">{msg}</p> : null}
        {!bundle?.executable?.ok ? <p className="chip bad">{bundle?.executable?.reason}</p> : null}
      </div>

      <div className="card">
        <h3>live-api-data（parquet → TirAgent）</h3>
        <p className="muted">
          对齐 <code>./run.sh live-api-data</code>：从 val.parquet 抽 N 条 gsm8k，经 .env API 跑 LLM→tool→answer→reward。
        </p>
        <div className="grid2">
          <div className="field">
            <label>parquet</label>
            <input value={parquet} onChange={(e) => setParquet(e.target.value)} />
          </div>
          <div className="field">
            <label>data_n</label>
            <input type="number" min={1} value={dataN} onChange={(e) => setDataN(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>source</label>
            <input value={source} onChange={(e) => setSource(e.target.value)} />
          </div>
        </div>
        <div className="row">
          <button type="button" className="primary" onClick={collectParquet} disabled={busy || !bundle?.executable?.ok}>
            {busy ? '采集中…' : 'Collect from parquet (live)'}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={async () => {
              try {
                const r = await api.sampleData({ parquet, data_n: dataN, source });
                setMsg(`sampled ${r.n}: ${(r.tasks || []).map((t: any) => t.id).join(', ')}`);
              } catch (e) {
                setMsg(String(e));
              }
            }}
          >
            仅预览抽样
          </button>
        </div>
        {rows.length > 0 ? (
          <table style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>group</th>
                <th>idx</th>
                <th>id</th>
                <th>answer</th>
                <th>reward</th>
                <th>branch</th>
                <th>tool</th>
              </tr>
            </thead>
            <tbody>
              {[...rows]
                .sort((a, b) => String(a.group_id || a.id).localeCompare(String(b.group_id || b.id)))
                .map((r, i) => (
                  <tr key={`${r.group_id || r.id}-${r.sample_index ?? i}`}>
                    <td>{r.group_id ?? r.id}</td>
                    <td>{r.sample_index ?? ''}</td>
                    <td>{r.id}</td>
                    <td>{String(r.answer ?? '')}</td>
                    <td>{r.reward ?? ''}</td>
                    <td>{r.is_branch ? '✓' : ''}</td>
                    <td>{r.tool ? '✓' : '✗'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </div>
  );
}

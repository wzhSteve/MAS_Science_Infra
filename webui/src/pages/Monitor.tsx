import { useEffect, useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';

type Props = {
  expId: string;
  trainRunId: string | null;
  trainRunning: boolean;
  trainLog: string;
  aglOnline: boolean;
  onRefreshLog: () => Promise<void>;
  visible?: boolean;
};

function RewardChart({ data, color = '#0969da' }: { data: Array<{ i: number; reward: number }>; color?: string }) {
  if (!data.length) {
    return <p className="muted">暂无数据点。</p>;
  }
  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="i" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="reward" stroke={color} dot />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MonitorPanel({
  expId,
  trainRunId,
  trainRunning,
  trainLog,
  aglOnline,
  onRefreshLog,
  visible = true,
}: Props) {
  const [model, setModel] = useState<any>(null);
  const [err, setErr] = useState('');
  const [idx, setIdx] = useState<number | null>(null);
  const [plugin, setPlugin] = useState('');

  const load = async () => {
    try {
      const m = await api.monitor(expId);
      setModel(m);
      setErr('');
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [expId]);

  useEffect(() => {
    if (visible) {
      const id = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
      return () => cancelAnimationFrame(id);
    }
  }, [visible]);

  const hyps = useMemo(() => {
    const all = model?.hypotheses || [];
    if (!plugin) return all;
    return all.filter((h: any) => h.plugin === plugin);
  }, [model, plugin]);

  const collectChart = (model?.rewards || []).map((r: any) => ({
    i: r.index,
    reward: r.reward,
  }));
  const stepChart = (model?.step_rewards || []).map((r: any) => ({
    i: r.index,
    reward: r.reward,
  }));
  const trainRolloutChart = (model?.train_rewards || []).map((r: any) => ({
    i: r.index,
    reward: r.reward,
  }));
  const trainChart = stepChart.length ? stepChart : trainRolloutChart;
  const hasCollect = model && !model.empty && (model.n || 0) > 0;
  const hasTrainReward = trainChart.length > 0;

  const metricsBtn = aglOnline ? (
    <a href="/agl/metrics" target="_blank" rel="noreferrer">
      <button type="button">AGL Metrics</button>
    </a>
  ) : (
    <button type="button" disabled title="训练进程拉起 LightningStore 后可用">
      Metrics 未就绪
    </button>
  );

  const trainCard = (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0 }}>训练日志</h3>
        <div className="row">
          <span className={trainRunning ? 'chip ok' : 'chip'}>{trainRunning ? 'running' : 'idle'}</span>
          {trainRunId ? <span className="muted">run={trainRunId}</span> : null}
          <button type="button" onClick={() => onRefreshLog()}>
            刷新日志
          </button>
          {metricsBtn}
        </div>
      </div>
      {trainLog ? (
        <pre className="log-pre">{trainLog}</pre>
      ) : (
        <p className="muted">尚无训练 stdout。若已点 Train，日志在 experiments/&lt;id&gt;/artifacts/runs/&lt;run&gt;/stdout.log；重启 UI 后会从磁盘恢复。</p>
      )}
    </div>
  );

  if (err) {
    return (
      <div>
        <div className="card chip bad">{err}</div>
        {trainCard}
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0 }}>Monitor</h2>
          <div className="row">
            <button type="button" onClick={load}>
              刷新
            </button>
            {metricsBtn}
          </div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <span className="chip">collect n={model?.n ?? 0}</span>
          <span className="chip ok">collect mean={model?.mean_reward ?? '—'}</span>
          <span className="chip">train pts={trainChart.length}</span>
          <span className="chip">errors={model?.n_error ?? 0}</span>
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          训练 Reward 与 AGL Metrics 同源（LightningStore）。Collect 曲线只反映 MAS 采集，不是 GRPO 训练步。
        </p>
      </div>
      {trainCard}
      <div className="card">
        <h3>训练 Reward（AGL）</h3>
        {visible ? (
          hasTrainReward ? (
            <RewardChart data={trainChart} color="#1a7f37" />
          ) : (
            <p className="muted">
              {aglOnline
                ? '训练已连接，但还没有带 final_reward 的 rollout。等第一批 episode 结束即可。'
                : '无 train / LightningStore 未就绪时这里为空。AGL Metrics 能开时会同步到这里。'}
            </p>
          )
        ) : (
          <div style={{ height: 220 }} />
        )}
      </div>
      <div className="card">
        <h3>Collect Reward</h3>
        {visible ? (
          hasCollect && collectChart.length ? (
            <RewardChart data={collectChart} />
          ) : (
            <p className="muted">尚无 collect.json。请先在 MAS 面板 Collect。</p>
          )
        ) : (
          <div style={{ height: 220 }} />
        )}
      </div>
      {hasCollect ? (
        <div className="card">
          <h3>Trajectories</h3>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>id</th>
                <th>reward</th>
                <th>format</th>
                <th>answer</th>
              </tr>
            </thead>
            <tbody>
              {(model.trajectories || []).map((t: any, i: number) => (
                <tr key={t.trajectory_id} onClick={() => setIdx(i)} style={{ cursor: 'pointer' }}>
                  <td>{i + 1}</td>
                  <td>{String(t.trajectory_id).slice(0, 10)}</td>
                  <td>{t.reward ?? ''}</td>
                  <td>{t.format_ok ? 'yes' : 'no'}</td>
                  <td>{String(t.answer || '').slice(0, 60)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {idx != null && model.trajectories?.[idx] ? (
            <div style={{ marginTop: 12 }}>
              <h4>Events</h4>
              <pre className="log-pre">{JSON.stringify(model.trajectories[idx].events, null, 2)}</pre>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="card">
        <div className="row">
          <h3 style={{ margin: 0 }}>Harness</h3>
          <select value={plugin} onChange={(e) => setPlugin(e.target.value)} style={{ width: 200 }}>
            <option value="">(all)</option>
            {[...new Set((model?.hypotheses || []).map((h: any) => h.plugin))].map((p) => (
              <option key={String(p)} value={String(p)}>
                {String(p)}
              </option>
            ))}
          </select>
        </div>
        <table>
          <thead>
            <tr>
              <th>plugin</th>
              <th>message</th>
            </tr>
          </thead>
          <tbody>
            {hyps.length === 0 ? (
              <tr>
                <td colSpan={2} className="muted">
                  (no hypotheses)
                </td>
              </tr>
            ) : (
              hyps.map((h: any, i: number) => (
                <tr key={i}>
                  <td>{h.plugin}</td>
                  <td>{h.message}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
        {model?.train_signal ? (
          <p className="muted" style={{ marginTop: 8 }}>
            TrainSignal advantage={JSON.stringify(model.train_signal.advantage)} loss=
            {JSON.stringify(model.train_signal.loss)}
          </p>
        ) : null}
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from 'react';
import { api, type Bundle } from '../api/client';

type Props = {
  expId: string;
  bundle: Bundle | null;
  meta: any;
  onReload: () => void;
  trainRunId: string | null;
  trainRunning: boolean;
  trainLog: string;
  aglOnline: boolean;
  onStartTrain: () => Promise<void>;
  onStopTrain: () => Promise<void>;
  onRefreshLog: () => Promise<void>;
};

function getPath(obj: any, path: string): any {
  return path.split('.').reduce((a, k) => (a == null ? undefined : a[k]), obj);
}

function setPath(obj: any, path: string, value: any): any {
  const parts = path.split('.');
  const out = structuredClone(obj);
  let cur = out;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] == null || typeof cur[parts[i]] !== 'object') cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
  return out;
}

const FIELDS: Array<{ path: string; label: string; type?: string }> = [
  { path: 'actor_rollout_ref.actor.optim.lr', label: 'actor.optim.lr', type: 'number' },
  { path: 'actor_rollout_ref.actor.clip_ratio_low', label: 'clip_ratio_low', type: 'number' },
  { path: 'actor_rollout_ref.actor.clip_ratio_high', label: 'clip_ratio_high', type: 'number' },
  { path: 'actor_rollout_ref.actor.entropy_coeff', label: 'entropy_coeff', type: 'number' },
  { path: 'actor_rollout_ref.actor.kl_loss_coef', label: 'kl_loss_coef', type: 'number' },
  { path: 'data.train_batch_size', label: 'train_batch_size', type: 'number' },
  { path: 'actor_rollout_ref.rollout.n', label: 'rollout.n', type: 'number' },
  { path: 'actor_rollout_ref.rollout.gpu_memory_utilization', label: 'gpu_memory_utilization', type: 'number' },
  { path: 'trainer.n_gpus_per_node', label: 'n_gpus_per_node', type: 'number' },
  { path: 'trainer.total_epochs', label: 'total_epochs', type: 'number' },
  { path: 'trainer.experiment_name', label: 'experiment_name' },
];

export function RLPanel({
  expId,
  bundle,
  meta,
  onReload,
  trainRunId,
  trainRunning,
  trainLog,
  aglOnline,
  onStartTrain,
  onStopTrain,
  onRefreshLog,
}: Props) {
  const [rl, setRl] = useState<any>(null);
  const [msg, setMsg] = useState('');
  const [dirty, setDirty] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const synced = useRef<string | null>(null);

  useEffect(() => {
    if (!bundle?.rl) return;
    if (synced.current !== expId) {
      setRl(structuredClone(bundle.rl));
      setDirty(false);
      synced.current = expId;
    }
  }, [expId, bundle]);

  const patchRl = (next: any) => {
    setRl(next);
    setDirty(true);
  };

  if (!rl) return <div className="card">加载 rl.yaml…</div>;

  const ids: number[] = (rl.devices?.ids || [0]).map((x: any) => Number(x));

  const save = async (next = rl): Promise<boolean> => {
    try {
      const payload = { ...next };
      for (const k of Object.keys(payload)) {
        if (k.includes('.')) delete payload[k];
      }
      const sel = (payload.devices?.ids || ids).map((x: any) => Number(x));
      payload.devices = { ids: sel };
      payload.trainer = {
        ...(payload.trainer || {}),
        n_gpus_per_node: sel.length,
      };
      if (payload.rollout_per_gpu != null) {
        payload.actor_rollout_ref = setPath(
          payload,
          'actor_rollout_ref.rollout.n',
          Number(payload.rollout_per_gpu),
        ).actor_rollout_ref;
      }
      await api.putSection(expId, 'rl', payload);
      setRl(payload);
      setDirty(false);
      setMsg('saved rl.yaml');
      onReload();
      return true;
    } catch (e) {
      setMsg(String(e));
      return false;
    }
  };

  const prefillsFromTrainSignal = async () => {
    try {
      const mon = await api.monitor(expId);
      const ts = mon.train_signal || {};
      const adv = ts.advantage || {};
      const loss = ts.loss || {};
      let next = { ...rl };
      if (adv.name) next.algo = adv.name;
      if (loss.clip_ratio_low != null) {
        next = setPath(next, 'actor_rollout_ref.actor.clip_ratio_low', loss.clip_ratio_low);
      }
      if (loss.clip_ratio_high != null) {
        next = setPath(next, 'actor_rollout_ref.actor.clip_ratio_high', loss.clip_ratio_high);
      }
      if (loss.entropy_coeff != null) {
        next = setPath(next, 'actor_rollout_ref.actor.entropy_coeff', loss.entropy_coeff);
      }
      if (loss.kl_loss_coef != null) {
        next = setPath(next, 'actor_rollout_ref.actor.kl_loss_coef', loss.kl_loss_coef);
      }
      patchRl(next);
      setMsg('prefilled from TrainSignal（需点保存）');
    } catch (e) {
      setMsg(String(e));
    }
  };

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>RL</h2>
        {dirty ? <span className="chip warn">未保存</span> : null}
      </div>
      <p className="muted">占用卡数 = 已选 GPU 数（{ids.length}）。Train 与本地 vLLM 互斥。每题采样 ≠ 画布上的采集入口。</p>
      <div className="grid2">
        <div className="field">
          <label>algo</label>
          <select value={rl.algo || 'grpo'} onChange={(e) => patchRl({ ...rl, algo: e.target.value })}>
            {(meta?.algos || ['grpo']).map((a: string) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>profile</label>
          <select value={rl.profile || 'fast'} onChange={(e) => patchRl({ ...rl, profile: e.target.value })}>
            {(meta?.profiles || ['fast']).map((p: string) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>每题采样条数（GRPO 组大小 / rollout.n）</label>
          <input
            type="number"
            min={1}
            value={rl.rollout_per_gpu ?? getPath(rl, 'actor_rollout_ref.rollout.n') ?? 2}
            onChange={(e) => {
              const v = Number(e.target.value);
              patchRl(setPath({ ...rl, rollout_per_gpu: v }, 'actor_rollout_ref.rollout.n', v));
            }}
          />
        </div>
        <div className="field">
          <label>并行采集进程（n_runners）</label>
          <input
            type="number"
            min={1}
            value={rl.n_runners ?? 1}
            onChange={(e) => patchRl({ ...rl, n_runners: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>model_path</label>
          <input value={rl.model_path || ''} onChange={(e) => patchRl({ ...rl, model_path: e.target.value })} />
        </div>
      </div>
      <div className="row">
        <button
          type="button"
          onClick={async () => {
            try {
              const g = await api.gpus();
              const rec = g.recommend || {};
              let next = { ...rl, ...rec };
              if (rec.devices) next.devices = rec.devices;
              if (rec['actor_rollout_ref.rollout.n'] != null) {
                next = setPath(next, 'actor_rollout_ref.rollout.n', rec['actor_rollout_ref.rollout.n']);
                next.rollout_per_gpu = rec.rollout_per_gpu;
              }
              if (rec['actor_rollout_ref.rollout.gpu_memory_utilization'] != null) {
                next = setPath(
                  next,
                  'actor_rollout_ref.rollout.gpu_memory_utilization',
                  rec['actor_rollout_ref.rollout.gpu_memory_utilization'],
                );
              }
              if (rec['trainer.n_gpus_per_node'] != null) {
                next = setPath(next, 'trainer.n_gpus_per_node', rec['trainer.n_gpus_per_node']);
              }
              patchRl(next);
              setMsg('已按当前机器推荐档位（需点保存）');
            } catch (e) {
              setMsg(String(e));
            }
          }}
        >
          按当前机器推荐
        </button>
        <button type="button" onClick={() => setShowAdvanced((s) => !s)}>
          {showAdvanced ? '收起高级' : '高级 Hydra 字段'}
        </button>
      </div>
      {showAdvanced ? (
        <div className="grid2">
          {FIELDS.map((f) => (
            <div className="field" key={f.path}>
              <label>{f.label}</label>
              <input
                type={f.type || 'text'}
                step={f.type === 'number' ? 'any' : undefined}
                value={getPath(rl, f.path) ?? ''}
                onChange={(e) => {
                  const v = f.type === 'number' ? Number(e.target.value) : e.target.value;
                  patchRl(setPath(rl, f.path, v));
                }}
              />
            </div>
          ))}
        </div>
      ) : null}
      <div className="row">
        <button type="button" className="primary" onClick={() => save()}>
          保存超参
        </button>
        <button type="button" onClick={prefillsFromTrainSignal}>
          从 TrainSignal 预填
        </button>
        <button
          type="button"
          className="primary"
          disabled={trainRunning}
          onClick={async () => {
            const ok = await save();
            if (ok) await onStartTrain();
          }}
        >
          一键启动训练
        </button>
        <button type="button" className="danger" onClick={onStopTrain}>
          一键中断训练
        </button>
        {aglOnline ? (
          <a href="/agl/metrics" target="_blank" rel="noreferrer">
            <button type="button">打开 AGL Metrics</button>
          </a>
        ) : (
          <button type="button" disabled title="训练进程拉起 LightningStore 后可用（同源 /agl/metrics）">
            Metrics 未就绪
          </button>
        )}
        {trainRunId ? (
          <button type="button" onClick={() => onRefreshLog()}>
            刷新日志
          </button>
        ) : null}
      </div>
      {msg ? <p className="muted">{msg}</p> : null}
      {trainLog ? <pre className="log-pre">{trainLog}</pre> : null}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { api } from '../../api/client';

export type GpuInfo = {
  id: number;
  name: string;
  mem_total_mb: number;
  mem_used_mb: number;
  util: number;
};

type Props = {
  selected: number[];
  onChange: (ids: number[]) => void;
  compact?: boolean;
};

export default function GpuPicker({ selected, onChange, compact = false }: Props) {
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  const [err, setErr] = useState('');
  const [count, setCount] = useState(0);

  const refresh = async () => {
    try {
      const r = await api.gpus();
      setGpus(r.gpus || []);
      setCount(r.count || 0);
      setErr(r.error || '');
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const toggle = (id: number) => {
    if (selected.includes(id)) {
      const next = selected.filter((x) => x !== id);
      onChange(next.length ? next : [id]);
    } else {
      onChange([...selected, id].sort((a, b) => a - b));
    }
  };

  const shortName = (name: string) => name.replace(/^NVIDIA\s+/i, '').split('-')[0] || name;

  return (
    <div>
      {compact ? null : (
        <div className="row">
          <span className="muted">GPU（{count} 张）</span>
          <button type="button" onClick={refresh}>
            探测
          </button>
        </div>
      )}
      {err ? <p className="muted">{err}</p> : null}
      <div className="row" style={{ marginTop: compact ? 0 : 8, flexWrap: 'wrap' }}>
        {gpus.length === 0 ? <span className="chip">未检测到 GPU</span> : null}
        {gpus.map((g) => {
          const on = selected.includes(g.id);
          const util = Math.max(0, Math.min(100, Math.round(g.util)));
          return (
            <button
              key={g.id}
              type="button"
              className={compact ? `gpu-chip${on ? ' on' : ''}` : on ? 'primary' : undefined}
              onClick={() => toggle(g.id)}
              title={`${g.name} ${Math.round(g.mem_used_mb)}/${Math.round(g.mem_total_mb)} MB · ${util}%`}
            >
              {compact ? (
                <>
                  <span>
                    GPU {g.id} · {util}%
                  </span>
                  <span className="gpu-util">
                    <span style={{ width: `${util}%` }} />
                  </span>
                </>
              ) : (
                `GPU ${g.id} · ${shortName(g.name)} · ${util}%`
              )}
            </button>
          );
        })}
        {compact ? (
          <button type="button" onClick={refresh} title="重新探测 GPU">
            探测
          </button>
        ) : null}
      </div>
    </div>
  );
}

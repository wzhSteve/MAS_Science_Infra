import { useEffect, useRef, useState } from 'react';
import { api, type Bundle } from '../api/client';

type Props = { expId: string; bundle: Bundle | null; meta: any; onReload: () => void };

export function HarnessPanel({ expId, bundle, meta, onReload }: Props) {
  const pluginsAll: string[] = meta?.harness_plugins || [];
  const stubs: string[] = meta?.stub_harness || [];
  const [selected, setSelected] = useState<string[]>([]);
  const [hyps, setHyps] = useState<any[]>([]);
  const [msg, setMsg] = useState('');
  const synced = useRef<string | null>(null);

  useEffect(() => {
    if (bundle?.harness?.plugins && synced.current !== expId) {
      setSelected([...bundle.harness.plugins]);
      synced.current = expId;
    }
  }, [expId, bundle]);

  const toggle = (name: string) => {
    if (stubs.includes(name)) return;
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));
  };

  return (
    <div className="card">
      <h2>Harness</h2>
      <div className="row" style={{ marginBottom: 12 }}>
        {pluginsAll.map((p) => (
          <label key={p} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              disabled={stubs.includes(p)}
              checked={selected.includes(p)}
              onChange={() => toggle(p)}
            />
            {p}
            {stubs.includes(p) ? <span className="chip warn">stub</span> : null}
          </label>
        ))}
      </div>
      <div className="row">
        <button
          type="button"
          className="primary"
          onClick={async () => {
            try {
              await api.putSection(expId, 'harness', { plugins: selected });
              setMsg('saved harness.yaml');
              onReload();
            } catch (e) {
              setMsg(String(e));
            }
          }}
        >
          保存
        </button>
        <button
          type="button"
          onClick={async () => {
            try {
              await api.putSection(expId, 'harness', { plugins: selected });
              const r = await api.diagnose(expId);
              setHyps(r.hypotheses || []);
              setMsg(`diagnose n=${r.n}`);
            } catch (e) {
              setMsg(String(e));
            }
          }}
        >
          对最近 Collect 诊断
        </button>
      </div>
      {msg ? <p className="muted">{msg}</p> : null}
      <table>
        <thead>
          <tr>
            <th>plugin</th>
            <th>event_id</th>
            <th>message</th>
          </tr>
        </thead>
        <tbody>
          {hyps.length === 0 ? (
            <tr>
              <td colSpan={3} className="muted">
                (no hypotheses)
              </td>
            </tr>
          ) : (
            hyps.map((h, i) => (
              <tr key={i}>
                <td>{h.plugin}</td>
                <td>{h.event_id || ''}</td>
                <td>{h.message}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

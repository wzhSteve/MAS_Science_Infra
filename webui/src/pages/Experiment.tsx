import { useEffect, useRef, useState } from 'react';
import { api, type Bundle } from '../api/client';

type Props = {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  setExpId: (id: string) => void;
};

export function ExperimentPanel({ expId, bundle, onReload, setExpId }: Props) {
  const [list, setList] = useState<string[]>([]);
  const [newId, setNewId] = useState('');
  const [seed, setSeed] = useState(42);
  const [name, setName] = useState('');
  const [msg, setMsg] = useState('');
  const synced = useRef<string | null>(null);

  useEffect(() => {
    api.listExperiments().then((r) => setList(r.experiments)).catch((e) => setMsg(String(e)));
  }, [bundle]);

  const saveMeta = async () => {
    if (!bundle) return;
    try {
      await api.putSection(expId, 'meta', {
        ...bundle.meta,
        seed,
        name: name || bundle.meta.name,
      });
      setMsg('saved experiment.yaml');
      onReload();
    } catch (e) {
      setMsg(String(e));
    }
  };

  useEffect(() => {
    if (bundle) {
      if (synced.current !== expId) {
        setSeed(Number(bundle.meta.seed || 42));
        setName(String(bundle.meta.name || ''));
        synced.current = expId;
      }
    }
  }, [expId, bundle]);

  return (
    <div>
      <div className="card">
        <h2>Experiment</h2>
        <div className="field">
          <label>当前实验</label>
          <select value={expId} onChange={(e) => setExpId(e.target.value)}>
            {list.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>
        <div className="grid2">
          <div className="field">
            <label>name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="field">
            <label>seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
        </div>
        <div className="row">
          <button type="button" className="primary" onClick={saveMeta}>
            保存 experiment.yaml
          </button>
          <button type="button" onClick={onReload}>
            重新加载
          </button>
        </div>
        {msg ? <p className="muted">{msg}</p> : null}
        {bundle ? (
          <p className="muted">
            path={bundle.path} · topology={bundle.workflow?.topology} ·{' '}
            <span className={`chip ${bundle.executable?.ok ? 'ok' : 'bad'}`}>
              {bundle.executable?.ok ? 'executable' : 'blocked'}
            </span>
          </p>
        ) : null}
      </div>
      <div className="card">
        <h3>新建实验</h3>
        <div className="row">
          <input
            placeholder="exp id"
            value={newId}
            onChange={(e) => setNewId(e.target.value)}
            style={{ maxWidth: 220 }}
          />
          <button
            type="button"
            className="primary"
            onClick={async () => {
              try {
                await api.createExperiment(newId, 42, newId);
                setExpId(newId);
                setNewId('');
                onReload();
              } catch (e) {
                setMsg(String(e));
              }
            }}
          >
            创建
          </button>
        </div>
      </div>
    </div>
  );
}

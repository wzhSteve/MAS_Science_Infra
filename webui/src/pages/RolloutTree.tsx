import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { api } from '../api/client';

type Props = { expId: string; visible: boolean };

type TreeNode = {
  node_id: string;
  parent_id: string | null;
  depth: number;
  role: string;
  agent_path?: string[];
  metrics?: Record<string, any>;
  reward?: number | null;
  verdict?: string | null;
};

type TreePayload = { file: string; tree: { tree_id: string; query?: string; nodes: TreeNode[]; outcomes?: Record<string, any> } };

const METRIC_KEYS = ['event_kind', 'h_tool', 'site_id', 'reward_scheme'];

function nodeLabel(n: TreeNode): string {
  const parts: string[] = [n.node_id.slice(0, 10)];
  const m = n.metrics || {};
  for (const k of METRIC_KEYS) {
    if (m[k] !== undefined && m[k] !== null && m[k] !== '') {
      parts.push(`${k}=${String(m[k]).slice(0, 18)}`);
    }
  }
  if (n.reward != null) parts.push(`r=${Number(n.reward).toFixed(2)}`);
  if (n.verdict) parts.push(n.verdict);
  return parts.join('\n');
}

export function RolloutTreePanel({ expId, visible }: Props) {
  const [trees, setTrees] = useState<TreePayload[]>([]);
  const [sel, setSel] = useState<string>('');
  const [err, setErr] = useState('');
  const [live, setLive] = useState('');
  const [sseMsg, setSseMsg] = useState('');

  const reload = useCallback(async () => {
    try {
      const r = await api.rolloutTrees(expId);
      setTrees(r.trees || []);
      setErr('');
      if (!r.trees?.length) setSel('');
      else if (!r.trees.some((t) => t.tree.tree_id === sel)) setSel(r.trees[0].tree.tree_id);
    } catch (e: any) {
      setErr(String(e?.message || e));
    }
  }, [expId, sel]);

  useEffect(() => {
    if (visible) reload();
  }, [visible, reload]);

  useEffect(() => {
    if (!visible) return;
    const t = setInterval(reload, 8000);
    return () => clearInterval(t);
  }, [visible, reload]);

  // P3: SSE live updates (pull mode above stays as fallback).
  useEffect(() => {
    if (!visible || typeof EventSource === 'undefined') return;
    const es = new EventSource(`/api/events?experiment_id=${encodeURIComponent(expId)}`);
    es.onmessage = (m) => {
      try {
        const payload = JSON.parse(m.data);
        if (payload?.type === 'rollout_tree') {
          const ev = payload.data || {};
          setLive(
            `${new Date().toLocaleTimeString()} · ${ev.event || '?'} · tree=${String(ev.tree_id || '').slice(0, 14)}${
              ev.payload?.n_nodes != null ? ` · nodes=${ev.payload.n_nodes}` : ''
            }`,
          );
          // tree updated in another process → refresh the pull view soon
          if (ev.event === 'node_added' || ev.event === 'outcome') {
            setTimeout(reload, 500);
          }
          if (ev.event === 'loss' && ev.payload?.metrics) {
            const keys = Object.keys(ev.payload.metrics).slice(0, 4);
            if (keys.length) {
              setSseMsg(
                keys.map((k) => `${k}=${Number(ev.payload.metrics[k]).toFixed(3)}`).join('  '),
              );
            }
          }
        }
      } catch {
        /* ignore malformed frames */
      }
    };
    es.onerror = () => setLive('');
    return () => es.close();
  }, [visible, expId, reload]);

  const { nodes, edges } = useMemo(() => {
    const payload = trees.find((t) => t.tree.tree_id === sel);
    if (!payload) return { nodes: [], edges: [] };
    const ns = payload.tree.nodes || [];
    const depthCount: Record<number, number> = {};
    const rfNodes: Node[] = ns.map((n) => {
      const d = Math.max(0, n.depth || 0);
      const col = depthCount[d] || 0;
      depthCount[d] = col + 1;
      return {
        id: n.node_id,
        position: { x: d * 260, y: col * 120 },
        data: { label: nodeLabel(n) },
        style: {
          fontSize: 10,
          whiteSpace: 'pre-line',
          background: n.role === 'root' ? 'rgba(96,165,250,0.15)' : undefined,
          border: n.role === 'root' ? '1px solid rgba(96,165,250,0.6)' : undefined,
        },
      };
    });
    const rfEdges: Edge[] = ns
      .filter((n) => n.parent_id)
      .map((n) => ({
        id: `e_${n.parent_id}_${n.node_id}`,
        source: n.parent_id as string,
        target: n.node_id,
        animated: false,
      }));
    return { nodes: rfNodes, edges: rfEdges };
  }, [trees, sel]);

  const current = trees.find((t) => t.tree.tree_id === sel);

  return (
    <div className="card" style={{ minHeight: 480 }}>
      <h2>RolloutTree</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        每 query 一棵树：root=query，叶子=outcome。Collect 产出为单链树（树≠branch）；
        branch 见 BRANCH_ROLLOUT_UI_TEST.md L0 原则。
      </p>
      <div className="row" style={{ marginBottom: 12, flexWrap: 'wrap' }}>
        {trees.length === 0 ? (
          <span className="muted">暂无树（跑 Collect 或 branch-ui-test --train 后出现）</span>
        ) : (
          trees.map((t) => (
            <button
              key={t.tree.tree_id}
              type="button"
              className={sel === t.tree.tree_id ? 'chip ok' : 'chip'}
              onClick={() => setSel(t.tree.tree_id)}
            >
              {t.tree.tree_id.slice(0, 14)}
            </button>
          ))
        )}
        <button type="button" onClick={reload} style={{ marginLeft: 'auto' }}>
          刷新
        </button>
      </div>
      {live ? (
        <div className="chip ok" style={{ marginBottom: 8, fontSize: 11 }}>
          SSE {live}
        </div>
      ) : null}
      {sseMsg ? (
        <div className="chip" style={{ marginBottom: 8, fontSize: 11 }}>
          {sseMsg}
        </div>
      ) : null}
      {err ? <p style={{ color: '#f87171' }}>{err}</p> : null}
      {current ? (
        <div className="muted" style={{ marginBottom: 8 }}>
          file={current.file} · nodes={current.tree.nodes?.length || 0} · leaves=
          {(current.tree.nodes || []).filter(
            (n) => !(current.tree.nodes || []).some((m) => m.parent_id === n.node_id),
          ).length}
          {current.tree.query ? ` · query=${String(current.tree.query).slice(0, 60)}` : ''}
        </div>
      ) : null}
      <div style={{ height: 420, border: '1px solid rgba(148,163,184,0.25)', borderRadius: 8 }}>
        {nodes.length > 0 ? (
          <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
            <Background gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
        ) : (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: '#94a3b8',
            }}
          >
            无选中树
          </div>
        )}
      </div>
    </div>
  );
}

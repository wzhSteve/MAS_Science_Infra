import { useCallback, useEffect, useMemo, useState, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import {
  executableInfo,
  flowToWorkflow,
  workflowToFlow,
  type WorkflowSpec,
} from './workflowGraph';

const nodeTypes = { agent: AgentNode, tool: ToolNode };

type Palette = {
  skills?: string[];
  roles?: string[];
  tools?: string[];
  edge_kinds?: string[];
  templates?: Array<{ id: string; label: string; workflow?: WorkflowSpec }>;
};

type Props = {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (wf: WorkflowSpec) => void;
  rl?: any;
  onRlPatch?: (patch: Record<string, unknown>) => void;
  onRlSave?: () => void;
};

export default function MasGraphEditor({ workflow, palette, onChange, rl, onRlPatch, onRlSave }: Props) {
  const initial = useMemo(() => workflowToFlow(workflow), []);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [selected, setSelected] = useState<Node | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [edgeKind, setEdgeKind] = useState('message');

  useEffect(() => {
    const { nodes: n, edges: e } = workflowToFlow(workflow);
    setNodes(n);
    setEdges(e);
  }, [
    workflow.topology,
    workflow.entry_agent,
    JSON.stringify(workflow.agents),
    JSON.stringify(workflow.hub),
    JSON.stringify(workflow.edges),
  ]);

  const emit = useCallback(
    (ns: Node[], es: typeof edges) => {
      onChange(flowToWorkflow(ns, es, workflow));
    },
    [onChange, workflow],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      const src = nodes.find((n) => n.id === c.source);
      const dst = nodes.find((n) => n.id === c.target);
      let kind = edgeKind;
      if (dst?.type === 'tool' || src?.type === 'tool') kind = 'tool_call';
      setEdges((eds) => {
        const next = addEdge({ ...c, label: kind, data: { kind } }, eds);
        emit(nodes, next);
        return next;
      });
    },
    [edgeKind, emit, nodes, setEdges],
  );

  const addNode = (kind: 'agent' | 'tool', id: string, role: string) => {
    if (nodes.some((n) => n.id === id)) return;
    const node: Node = {
      id,
      type: kind,
      position: { x: 120 + nodes.length * 36, y: 90 + nodes.length * 28 },
      data: {
        label: id,
        role,
        skills: role === 'verifier' ? ['verifier'] : kind === 'agent' ? ['react_loop'] : [],
        tools: [],
        trainable: role !== 'verifier',
        entry: false,
        system_prompt: '',
      },
    };
    const next = [...nodes, node];
    setNodes(next);
    emit(next, edges);
  };

  const setEntry = (id: string) => {
    const next = nodes.map((n) =>
      n.type === 'tool' ? n : { ...n, data: { ...n.data, entry: n.id === id } },
    );
    setNodes(next);
    setSelected(next.find((n) => n.id === id) || null);
    emit(next, edges);
  };

  const updateSelected = (patch: Record<string, unknown>) => {
    if (!selected) return;
    const next = nodes.map((n) =>
      n.id === selected.id ? { ...n, data: { ...n.data, ...patch } } : n,
    );
    setNodes(next);
    setSelected(next.find((n) => n.id === selected.id) || null);
    emit(next, edges);
  };

  const applyTemplate = (wf: WorkflowSpec) => {
    const { nodes: n, edges: e } = workflowToFlow({ ...workflow, ...wf });
    setNodes(n);
    setEdges(e);
    onChange({ ...workflow, ...wf });
  };

  const onDragOver = (ev: DragEvent) => {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = 'move';
  };

  const onDrop = (ev: DragEvent) => {
    ev.preventDefault();
    const raw = ev.dataTransfer.getData('application/science-node');
    if (!raw) return;
    try {
      const payload = JSON.parse(raw) as { kind: 'agent' | 'tool'; id: string; role: string };
      addNode(payload.kind, payload.id, payload.role);
    } catch {
      /* ignore */
    }
  };

  const exe = executableInfo(flowToWorkflow(nodes, edges, workflow));
  const entryId = nodes.find((n) => n.data?.entry)?.id || workflow.entry_agent || 'hub';

  return (
    <div>
      <div className="row" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
        {(palette.templates || []).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => t.workflow && applyTemplate(t.workflow)}
            disabled={!t.workflow}
          >
            模板: {t.label}
          </button>
        ))}
        <span className={`chip ${exe.ok ? 'ok' : 'bad'}`}>
          {exe.ok ? 'executable' : 'not executable'}: {exe.reason}
        </span>
        <span className="chip ok">入口={entryId}</span>
      </div>
      <div className="studio-graph">
        <div>
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="muted">连线</span>
            <select value={edgeKind} onChange={(e) => setEdgeKind(e.target.value)} style={{ width: 130 }}>
              {(palette.edge_kinds || ['message', 'route', 'feedback', 'tool_call']).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <p className="muted">从左侧拖 Agent / Tool 到画布。把「采集入口」拖到 Agent 上，表示 Episode 从哪个节点开始（不是 GRPO 每题采样条数）。</p>
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="muted">添加</span>
            {(palette.roles || ['hub', 'planner', 'executor', 'verifier']).map((r) => (
              <button
                key={r}
                type="button"
                draggable
                onDragStart={(ev) =>
                  ev.dataTransfer.setData(
                    'application/science-node',
                    JSON.stringify({ kind: 'agent', id: r === 'hub' ? 'hub' : `${r}_${nodes.length + 1}`, role: r }),
                  )
                }
                onClick={() => addNode('agent', r === 'hub' ? 'hub' : `${r}_${nodes.length + 1}`, r)}
              >
                + {r}
              </button>
            ))}
            {(palette.tools || []).map((t) => (
              <button
                key={t}
                type="button"
                draggable
                onDragStart={(ev) =>
                  ev.dataTransfer.setData(
                    'application/science-node',
                    JSON.stringify({ kind: 'tool', id: t, role: 'tool' }),
                  )
                }
                onClick={() => addNode('tool', t, 'tool')}
              >
                + {t}
              </button>
            ))}
            <button
              type="button"
              draggable
              onDragStart={(ev) =>
                ev.dataTransfer.setData(
                  'application/science-pin',
                  JSON.stringify({ pin: 'entry' }),
                )
              }
            >
              采集入口
            </button>
          </div>
          <div
            className="canvas-wrap"
            onDragOver={onDragOver}
            onDrop={(ev) => {
              const pin = ev.dataTransfer.getData('application/science-pin');
              if (pin) {
                ev.preventDefault();
                const el = document.elementFromPoint(ev.clientX, ev.clientY);
                const nodeEl = el?.closest('.react-flow__node');
                const hitId = nodeEl?.getAttribute('data-id');
                const hit = nodes.find((n) => n.id === hitId);
                const targetId =
                  hit && hit.type !== 'tool'
                    ? hit.id
                    : hoverId && nodes.find((n) => n.id === hoverId)?.type !== 'tool'
                      ? hoverId
                      : selected && selected.type !== 'tool'
                        ? selected.id
                        : '';
                if (targetId) setEntry(targetId);
                return;
              }
              onDrop(ev);
            }}
          >
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              onNodeClick={(_, n) => setSelected(n)}
              onNodeMouseEnter={(_, n) => setHoverId(n.id)}
              onNodeMouseLeave={() => setHoverId(null)}
              onPaneClick={() => setSelected(null)}
              onNodeDragStop={() => emit(nodes, edges)}
              fitView
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
          </div>
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>节点属性</h3>
          {!selected ? (
            <p className="muted">选中 Agent 可编辑 prompt / tools / 是否训练；拖「采集入口」到节点上设 Episode 起点。</p>
          ) : selected.type === 'tool' ? (
            <p className="muted">工具节点 {selected.id}。用 tool_call 边连到 Agent。</p>
          ) : (
            <>
              <div className="field">
                <label>id</label>
                <input value={selected.id} disabled />
              </div>
              <div className="field">
                <label>role</label>
                <input
                  value={String(selected.data?.role || '')}
                  onChange={(e) => updateSelected({ role: e.target.value })}
                />
              </div>
              <div className="field">
                <label>system prompt（profile）</label>
                <textarea
                  rows={5}
                  value={String(selected.data?.system_prompt || '')}
                  onChange={(e) => updateSelected({ system_prompt: e.target.value })}
                />
              </div>
              <div className="field">
                <label>skills（逗号分隔）</label>
                <input
                  value={((selected.data?.skills as string[]) || []).join(', ')}
                  onChange={(e) =>
                    updateSelected({
                      skills: e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </div>
              <div className="field">
                <label>tools</label>
                <div className="row">
                  {(palette.tools || []).map((t) => {
                    const tools = (selected.data?.tools as string[]) || [];
                    const on = tools.includes(t);
                    return (
                      <label key={t} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                        <input
                          type="checkbox"
                          checked={on}
                          onChange={() => {
                            const next = on ? tools.filter((x) => x !== t) : [...tools, t];
                            updateSelected({ tools: next });
                          }}
                        />
                        {t}
                      </label>
                    );
                  })}
                </div>
              </div>
              <div className="row">
                <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={selected.data?.trainable !== false}
                    onChange={(e) => updateSelected({ trainable: e.target.checked })}
                  />
                  参与 RL（trainable）
                </label>
                <button type="button" className="primary" onClick={() => setEntry(selected.id)}>
                  设为采集入口
                </button>
              </div>
              {selected.id === 'hub' ? (
                <div className="field">
                  <label>verify skill（空=关闭）</label>
                  <select
                    value={String(selected.data?.verify || '')}
                    onChange={(e) => updateSelected({ verify: e.target.value || null })}
                  >
                    <option value="">(none)</option>
                    {(palette.skills || []).map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              {selected.data?.entry || selected.id === entryId ? (
                <details open className="field" style={{ marginTop: 12 }}>
                  <summary>训练简参（写入 rl.yaml，不画成图节点）</summary>
                  <div className="field" style={{ marginTop: 8 }}>
                    <label>每题采样条数（GRPO 组大小）</label>
                    <input
                      type="number"
                      min={1}
                      value={rl?.rollout_per_gpu ?? rl?.actor_rollout_ref?.rollout?.n ?? 2}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        onRlPatch?.({
                          rollout_per_gpu: v,
                          actor_rollout_ref: {
                            ...(rl?.actor_rollout_ref || {}),
                            rollout: { ...(rl?.actor_rollout_ref?.rollout || {}), n: v },
                          },
                        });
                      }}
                    />
                  </div>
                  <div className="field">
                    <label>并行采集进程 n_runners</label>
                    <input
                      type="number"
                      min={1}
                      value={rl?.n_runners ?? 1}
                      onChange={(e) => onRlPatch?.({ n_runners: Number(e.target.value) })}
                    />
                  </div>
                  <p className="muted">算法={String(rl?.algo || 'grpo')} · 可训 Agent → --active-agent</p>
                  {onRlSave ? (
                    <button type="button" onClick={onRlSave}>
                      写入 rl.yaml
                    </button>
                  ) : null}
                </details>
              ) : null}
            </>
          )}
          <p className="muted" style={{ marginTop: 12 }}>
            图序列化为 WorkflowSpec YAML。连线 kind：tool_call / route / message / feedback。
          </p>
        </div>
      </div>
    </div>
  );
}

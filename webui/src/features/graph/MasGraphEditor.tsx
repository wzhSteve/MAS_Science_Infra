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
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import RouterNode from './RouterNode';
import {
  executableInfo,
  flowToWorkflow,
  workflowToFlow,
  type WorkflowSpec,
} from './workflowGraph';
import {
  GATE_TYPES,
  deriveBranchCandidates,
  findSiteForCandidate,
  upsertSiteFromCandidate,
  type BranchCandidate,
  type BranchSiteConfig,
} from './branchSites';

const nodeTypes = { agent: AgentNode, tool: ToolNode, router: RouterNode };

type Palette = {
  skills?: string[];
  roles?: string[];
  tools?: string[];
  edge_kinds?: string[];
  templates?: Array<{ id: string; label: string; workflow?: WorkflowSpec }>;
  // agent-framework W2/W3: unified agent palette (schema 0.3)
  agent_templates?: Array<{
    id: string;
    kind: string;
    label: string;
    hint?: string;
  }>;
  tool_agents?: Array<{
    id: string;
    backend?: string;
    llm_required?: boolean;
    description?: string;
  }>;
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
  const [branchCand, setBranchCand] = useState<BranchCandidate | null>(null);

  const sites = useMemo(
    () => ((workflow.sampling?.sites || []) as BranchSiteConfig[]),
    [JSON.stringify(workflow.sampling?.sites || [])],
  );
  const candidates = useMemo(() => deriveBranchCandidates(workflow), [workflow]);

  useEffect(() => {
    const { nodes: n, edges: e } = workflowToFlow(workflow);
    const enabledByNode: Record<string, string> = {};
    for (const c of deriveBranchCandidates(workflow)) {
      const s = findSiteForCandidate(sites, c);
      if (s?.enabled) enabledByNode[c.nodeId] = s.gate?.type || 'on';
    }
    const decorated = n.map((node) => ({
      ...node,
      data: {
        ...node.data,
        branchEnabled: !!enabledByNode[node.id],
        branchGate: enabledByNode[node.id],
      },
    }));
    setNodes(decorated);
    setEdges(e);
  }, [
    workflow.topology,
    workflow.entry_agent,
    JSON.stringify(workflow.agents),
    JSON.stringify(workflow.hub),
    JSON.stringify(workflow.edges),
    JSON.stringify(workflow.sampling?.sites || []),
  ]);

  const patchSites = (nextSites: BranchSiteConfig[]) => {
    onChange({
      ...workflow,
      sampling: {
        ...(workflow.sampling || {}),
        sites: nextSites as any,
      },
    });
  };

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
      // W5: router -> agent creates a candidate edge (routing sugar).
      if (src?.type === 'router' && dst?.type === 'agent') {
        // kind=blank candidates are referenced as blank:<id> in RouterSpec.
        const dstKind = String(dst.data?.kind || 'blank');
        const candValue = dstKind === 'blank' ? `blank:${dst.id}` : dst.id;
        setNodes((ns) => {
          const merged = Array.from(
            new Set([...((ns.find((n) => n.id === src.id)?.data?.candidates as string[]) || []), candValue]),
          );
          const next = ns.map((n) =>
            n.id === src.id
              ? { ...n, data: { ...n.data, candidates: merged, n_candidates: merged.length } }
              : n,
          );
          const nextEdges = addEdge(
            { ...c, label: 'candidate', data: { kind: 'candidate' } },
            edges,
          );
          emit(next, nextEdges);
          return next;
        });
        return;
      }
      let kind = edgeKind;
      if (dst?.type === 'tool' || src?.type === 'tool') kind = 'tool_call';
      setEdges((eds) => {
        const next = addEdge({ ...c, label: kind, data: { kind } }, eds);
        emit(nodes, next);
        return next;
      });
    },
    [edgeKind, emit, nodes, edges, setEdges],
  );

  const addNode = (
    kind: 'agent' | 'tool',
    id: string,
    role: string,
    agentKind?: string,
    profile?: Record<string, unknown>,
  ) => {
    if (nodes.some((n) => n.id === id)) return;
    // W3: tool-agent templates create agent nodes (kind=tool), not legacy
    // tool nodes — same data structure as any other agent.
    const k = agentKind || (kind === 'tool' ? 'tool' : 'blank');
    const node: Node = {
      id,
      type: 'agent',
      position: { x: 120 + nodes.length * 36, y: 90 + nodes.length * 28 },
      data: {
        label: id,
        kind: k,
        role,
        skills: k === 'verifier' ? ['verifier'] : ['react_loop'],
        tools: [],
        trainable: k !== 'verifier' && k !== 'tool',
        entry: false,
        system_prompt: '',
        memory_scope: 'agent',
        profile: profile || {},
      },
    };
    const next = [...nodes, node];
    setNodes(next);
    emit(next, edges);
  };

  const addRouterNode = (id?: string) => {
    const rid = id || (nodes.some((n) => n.id === 'router_1') ? `router_${Date.now() % 1000}` : 'router_1');
    if (nodes.some((n) => n.id === rid)) return;
    const node: Node = {
      id: rid,
      type: 'router',
      position: { x: 420 + (nodes.length % 3) * 160, y: 200 + Math.floor(nodes.length / 3) * 120 },
      data: {
        label: rid,
        strategy: 'llm_choice',
        candidates: [],
        n_candidates: 0,
      },
    };
    const next = [...nodes, node];
    setNodes(next);
    setSelected(node);
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

  // W5: rename a router node — candidate edges follow the new id.
  const renameRouter = (oldId: string, newId: string) => {
    const rid = newId.trim() || oldId;
    if (rid === oldId) return;
    if (nodes.some((n) => n.id === rid)) return; // id collision: keep old
    const next = nodes.map((n) =>
      n.id === oldId
        ? { ...n, id: rid, data: { ...n.data, label: rid } }
        : n,
    );
    const nextEdges = edges.map((e) => ({
      ...e,
      source: e.source === oldId ? rid : e.source,
      target: e.target === oldId ? rid : e.target,
    }));
    setNodes(next);
    setEdges(nextEdges);
    setSelected(next.find((n) => n.id === rid) || null);
    emit(next, nextEdges);
  };

  // W5: create/remove a candidate edge (router -> agent) on the canvas.
  const toggleCandidateEdge = (routerId: string, target: string, on: boolean) => {
    if (on) {
      if (edges.some((e) => e.source === routerId && e.target === target && e.data?.kind === 'candidate')) return;
      const next = [
        ...edges,
        {
          id: `e-router-${routerId}-${target}`,
          source: routerId,
          target,
          label: 'candidate',
          data: { kind: 'candidate' },
        } as Edge,
      ];
      setEdges(next);
      emit(nodes, next);
    } else {
      const next = edges.filter(
        (e) => !(e.source === routerId && e.target === target && e.data?.kind === 'candidate'),
      );
      setEdges(next);
      emit(nodes, next);
    }
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
      const payload = JSON.parse(raw) as {
        kind: 'agent' | 'tool' | 'router';
        id: string;
        role: string;
        agentKind?: string;
        profile?: Record<string, unknown>;
      };
      if (payload.kind === 'router') {
        addRouterNode(payload.id);
        return;
      }
      addNode(payload.kind, payload.id, payload.role, payload.agentKind, payload.profile);
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
          <div className="row" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
            <span className="muted">添加</span>
            {(palette.agent_templates || []).length > 0
              ? (palette.agent_templates || []).map((t) => {
                  if (t.kind === 'router') {
                    // W5: router template is handled in its own section below.
                    return null;
                  }
                  const baseId =
                    t.kind === 'tool' ? 'tool_agent' : t.kind === 'hub' ? 'hub' : `${t.kind}_${nodes.length + 1}`;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      title={t.hint || t.label}
                      draggable
                      onDragStart={(ev) =>
                        ev.dataTransfer.setData(
                          'application/science-node',
                          JSON.stringify({
                            kind: 'agent',
                            id: nodes.some((n) => n.id === baseId) ? `${baseId}_${Date.now() % 1000}` : baseId,
                            role: t.kind,
                            agentKind: t.kind,
                          }),
                        )
                      }
                      onClick={() =>
                        addNode(
                          'agent',
                          nodes.some((n) => n.id === baseId) ? `${baseId}_${Date.now() % 1000}` : baseId,
                          t.kind,
                          t.kind,
                        )
                      }
                    >
                      + [{t.kind}] {t.label}
                    </button>
                  );
                })
              : (palette.roles || ['hub', 'planner', 'executor', 'verifier']).map((r) => (
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
            {(palette.tool_agents || palette.tools || []).length > 0 &&
              (palette.tool_agents && palette.tool_agents.length > 0
                ? palette.tool_agents.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      title={t.description || `${t.backend} backend`}
                      draggable
                      onDragStart={(ev) =>
                        ev.dataTransfer.setData(
                          'application/science-node',
                          JSON.stringify({ kind: 'agent', id: t.id, role: 'tool', agentKind: 'tool' }),
                        )
                      }
                      onClick={() => addNode('agent', t.id, 'tool', 'tool')}
                    >
                      + [tool] {t.id}
                      {t.llm_required ? ' (llm)' : ''}
                    </button>
                  ))
                : (palette.tools || []).map((t) => (
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
                  )))}
            <button
              type="button"
              title="多专家路由：从 candidates 中选择一个 agent（W5）"
              draggable
              onDragStart={(ev) =>
                ev.dataTransfer.setData(
                  'application/science-node',
                  JSON.stringify({
                    kind: 'router',
                    id: nodes.some((n) => n.id === 'router_1')
                      ? `router_${Date.now() % 1000}`
                      : 'router_1',
                    role: 'router',
                  }),
                )
              }
              onClick={() => addRouterNode()}
            >
              + [router] Router
            </button>
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
          ) : selected.type === 'router' ? (
            <>
              <div className="field">
                <label>id</label>
                <input
                  value={selected.id}
                  onChange={(e) => renameRouter(selected.id, e.target.value)}
                />
              </div>
              <div className="field">
                <label>strategy</label>
                <select
                  value={String(selected.data?.strategy || 'llm_choice')}
                  onChange={(e) => updateSelected({ strategy: e.target.value })}
                >
                  <option value="llm_choice">llm_choice（LLM 工具调用选择）</option>
                  <option value="score">score（打分 agent 排序）</option>
                  <option value="round_robin">round_robin（轮询）</option>
                </select>
              </div>
              {String(selected.data?.strategy || 'llm_choice') === 'score' ? (
                <div className="field">
                  <label>scorer（打分 agent id）</label>
                  <select
                    value={String(selected.data?.scorer || '')}
                    onChange={(e) => updateSelected({ scorer: e.target.value || null })}
                  >
                    <option value="">(none)</option>
                    {nodes
                      .filter((n) => n.type === 'agent')
                      .map((n) => (
                        <option key={n.id} value={n.id}>
                          {n.id}
                        </option>
                      ))}
                  </select>
                </div>
              ) : null}
              <div className="field">
                <label>candidates（画布 Agent 节点；kind=tool 值为 id，kind=blank 值为 blank:id）</label>
                <div className="row" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  {nodes
                    .filter((n) => n.type === 'agent')
                    .map((n) => {
                      const kind = String(n.data?.kind || 'blank');
                      const value = kind === 'blank' ? `blank:${n.id}` : n.id;
                      const cands = (selected.data?.candidates as string[]) || [];
                      const on = cands.includes(value);
                      return (
                        <label key={n.id} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                          <input
                            type="checkbox"
                            checked={on}
                            onChange={() => {
                              const next = on ? cands.filter((x) => x !== value) : [...cands, value];
                              toggleCandidateEdge(selected.id, value, !on);
                              updateSelected({ candidates: next, n_candidates: next.length });
                            }}
                          />
                          [{kind}] {n.id}
                        </label>
                      );
                    })}
                </div>
              </div>
              <p className="muted">
                连线：从 Router 拖到候选 Agent 会自动创建 candidate 边；Del 键删除节点即从 routers[] 移除。
              </p>
            </>
          ) : selected.type === 'tool' ? (
            <>
              <div className="field">
                <label>id</label>
                <input value={selected.id} disabled />
              </div>
              <p className="muted">
                旧式 tool 节点 {selected.id}。推荐用 tool_call 边连到 Agent；如需 LLM 后端请改用 [tool] Agent 模板节点。
              </p>
            </>
          ) : (
            <>
              <div className="field">
                <label>id</label>
                <input value={selected.id} disabled />
              </div>
              <div className="field">
                <label>kind</label>
                <select
                  value={String(selected.data?.kind || 'blank')}
                  disabled={selected.id === 'hub'}
                  onChange={(e) => updateSelected({ kind: e.target.value })}
                >
                  {['tool', 'planner', 'verifier', 'hub', 'blank'].map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
                {selected.id === 'hub' ? (
                  <p className="muted">hub 节点 kind 固定不可改。</p>
                ) : null}
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
                <label>memory scope</label>
                <select
                  value={String(selected.data?.memory_scope || 'agent')}
                  onChange={(e) => updateSelected({ memory_scope: e.target.value })}
                >
                  <option value="agent">agent（仅本 agent 可见）</option>
                  <option value="shared">shared（hub 共享缓冲）</option>
                </select>
              </div>
              {String(selected.data?.kind || 'blank') === 'tool' ? (
                <div className="field">
                  <label>profile.llm_required（测试态切 epc_aw LLM 后端）</label>
                  <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <input
                      type="checkbox"
                      checked={!!((selected.data?.profile as Record<string, unknown>) || {}).llm_required}
                      onChange={(e) =>
                        updateSelected({
                          profile: {
                            ...((selected.data?.profile as Record<string, unknown>) || {}),
                            llm_required: e.target.checked,
                          },
                        })
                      }
                    />
                    LLM 后端（测试态生效，训练保持 pure）
                  </label>
                </div>
              ) : null}
              {String(selected.data?.kind || 'blank') === 'blank' ? (
                <details open className="field" style={{ marginTop: 4 }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 650 }}>blank agent profile</summary>
                  <div className="field" style={{ marginTop: 8 }}>
                    <label>skills（逗号分隔，写入 profile.skills）</label>
                    <input
                      value={
                        (((selected.data?.profile as Record<string, unknown>) || {}).skills as string[] || [])
                          .join(', ')
                      }
                      onChange={(e) =>
                        updateSelected({
                          profile: {
                            ...((selected.data?.profile as Record<string, unknown>) || {}),
                            skills: e.target.value
                              .split(',')
                              .map((s) => s.trim())
                              .filter(Boolean),
                          },
                        })
                      }
                    />
                  </div>
                  <div className="field">
                    <label>memory policy</label>
                    <select
                      value={String(
                        ((selected.data?.profile as Record<string, any>) || {}).memory?.policy || 'append_latest'
                      )}
                      onChange={(e) =>
                        updateSelected({
                          profile: {
                            ...((selected.data?.profile as Record<string, unknown>) || {}),
                            memory: {
                              ...(((selected.data?.profile as Record<string, any>) || {}).memory || {}),
                              policy: e.target.value,
                            },
                          },
                        })
                      }
                    >
                      <option value="append_latest">append_latest</option>
                      <option value="none">none</option>
                    </select>
                  </div>
                  <div className="field">
                    <label>memory max_items</label>
                    <input
                      type="number"
                      min={1}
                      value={Number(
                        ((selected.data?.profile as Record<string, any>) || {}).memory?.max_items || 4
                      )}
                      onChange={(e) =>
                        updateSelected({
                          profile: {
                            ...((selected.data?.profile as Record<string, unknown>) || {}),
                            memory: {
                              ...(((selected.data?.profile as Record<string, any>) || {}).memory || {}),
                              max_items: Number(e.target.value),
                            },
                          },
                        })
                      }
                    />
                  </div>
                  <p className="muted">
                    Router 候选引用本节点时用 <code>blank:{'{id}'}</code> 形式（画布 Router Inspector 自动处理）。
                  </p>
                </details>
              ) : null}
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
                <label>tools（tool-agents）</label>
                <div className="row">
                  {(palette.tool_agents && palette.tool_agents.length > 0
                    ? palette.tool_agents.map((t) => t.id)
                    : palette.tools || []
                  ).map((t) => {
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

          <details style={{ marginTop: 16 }}>
            <summary style={{ cursor: 'pointer', fontWeight: 650 }}>
              Branch Rollout 站点（Advanced）
            </summary>
            <p className="muted">
              主交互请用下方「Rollout Sampling」轨迹小窗；此处为完整候选列表同步视图。
            </p>
          <div style={{ maxHeight: 220, overflow: 'auto' }}>
            {candidates.map((c) => {
              const site = findSiteForCandidate(sites, c);
              const on = !!site?.enabled;
              return (
                <div
                  key={c.id}
                  className="row"
                  style={{
                    justifyContent: 'space-between',
                    marginBottom: 6,
                    padding: 6,
                    border: branchCand?.id === c.id ? '1px solid #0969da' : '1px solid #d0d7de',
                    borderRadius: 6,
                    cursor: 'pointer',
                  }}
                  onClick={() => setBranchCand(c)}
                >
                  <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={(e) => {
                        e.stopPropagation();
                        patchSites(
                          upsertSiteFromCandidate(sites, c, {
                            enabled: e.target.checked,
                            gate: {
                              type: site?.gate?.type || c.recommendedGate,
                              params: site?.gate?.params || {},
                            },
                          }),
                        );
                      }}
                    />
                    <span>
                      {c.label} <span className="muted">({c.recommendedGate})</span>
                    </span>
                  </label>
                </div>
              );
            })}
          </div>
          {branchCand ? (
            <div className="field" style={{ marginTop: 8 }}>
              <label>选中站点：{branchCand.label}</label>
              <select
                value={findSiteForCandidate(sites, branchCand)?.gate?.type || branchCand.recommendedGate}
                onChange={(e) =>
                  patchSites(
                    upsertSiteFromCandidate(sites, branchCand, {
                      enabled: true,
                      gate: {
                        type: e.target.value,
                        params: findSiteForCandidate(sites, branchCand)?.gate?.params || {},
                      },
                    }),
                  )
                }
              >
                {GATE_TYPES.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
              <label style={{ marginTop: 8 }}>reward.scheme</label>
              <select
                value={findSiteForCandidate(sites, branchCand)?.reward?.scheme || 'scalar_grpo'}
                onChange={(e) =>
                  patchSites(
                    upsertSiteFromCandidate(sites, branchCand, {
                      enabled: true,
                      reward: {
                        ...(findSiteForCandidate(sites, branchCand)?.reward || {}),
                        scheme: e.target.value,
                      },
                    }),
                  )
                }
              >
                <option value="scalar_grpo">scalar_grpo (R0)</option>
                <option value="rae_adjudicate">rae_adjudicate (R1)</option>
              </select>
              <label style={{ marginTop: 8 }}>fork.beam_size</label>
              <input
                type="number"
                min={1}
                value={Number(findSiteForCandidate(sites, branchCand)?.fork?.beam_size ?? 2)}
                onChange={(e) =>
                  patchSites(
                    upsertSiteFromCandidate(sites, branchCand, {
                      enabled: true,
                      fork: {
                        ...(findSiteForCandidate(sites, branchCand)?.fork || {}),
                        beam_size: Number(e.target.value),
                        share_observation: true,
                      },
                    }),
                  )
                }
              />
              <label style={{ marginTop: 8 }}>fork.resume_mode</label>
              <select
                value={String(findSiteForCandidate(sites, branchCand)?.fork?.resume_mode || 'messages')}
                onChange={(e) =>
                  patchSites(
                    upsertSiteFromCandidate(sites, branchCand, {
                      enabled: true,
                      fork: {
                        ...(findSiteForCandidate(sites, branchCand)?.fork || {}),
                        resume_mode: e.target.value,
                        share_observation: true,
                      },
                    }),
                  )
                }
              >
                <option value="messages">messages</option>
                <option value="token_prefix">token_prefix (Advanced)</option>
              </select>
            </div>
          ) : null}
          <p className="muted">
            已启用 {sites.filter((s) => s.enabled).length} / {candidates.length} 候选
          </p>
          </details>
          <p className="muted" style={{ marginTop: 12 }}>
            图序列化为 WorkflowSpec YAML。连线 kind：tool_call / route / message / feedback。
          </p>
        </div>
      </div>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyEdgeChanges, applyNodeChanges,
  type Connection, type EdgeChange, type NodeChange, type XYPosition,
} from '@xyflow/react';
import type { AgentKind, Palette, WorkflowSpec } from '../../../shared/api/types';
import type { EdgeKind, EdgePatch, GraphNodeData, GraphNode, GraphEdge, GraphNodePreset, GraphSelection } from '../types';
import { flowToWorkflow, graphEdge, poolNodeId, routerIdFromPool, workflowGraphRevision, workflowToFlow } from './workflowGraph';
import { createEdgeRules, edgeConnection } from './edgeRules';
import { defaultHandles, resolveHandles, serializeEdges } from './edgeGeometry';

type Graph = { nodes: GraphNode[]; edges: GraphEdge[] };

export function useGraphEditor(
  workflow: WorkflowSpec,
  onChange: (workflow: WorkflowSpec) => void,
  palette: Palette = {},
  modelNames: ReadonlyMap<string, string> = new Map(),
) {
  const [graph, setGraph] = useState<Graph>(() => workflowToFlow(workflow, modelNames));
  const [selection, setSelection] = useState<GraphSelection>(null);
  const [notice, setNotice] = useState('');
  const graphRef = useRef(graph);
  const workflowRef = useRef(workflow);
  const graphRevisionRef = useRef(workflowGraphRevision(workflow));
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const rules = useMemo(() => createEdgeRules(graph.nodes, graph.edges, palette), [graph.nodes, graph.edges, palette]);
  const rulesRef = useRef(rules);
  rulesRef.current = rules;

  const replaceGraph = useCallback((next: Graph) => {
    graphRef.current = next;
    setGraph(next);
  }, []);

  useEffect(() => {
    if (workflow === workflowRef.current) return;
    workflowRef.current = workflow;
    const revision = workflowGraphRevision(workflow);
    if (revision === graphRevisionRef.current) return;
    graphRevisionRef.current = revision;
    const next = workflowToFlow(workflow, modelNames);
    const previous = new Map(graphRef.current.nodes.map((node) => [node.id, node]));
    next.nodes = next.nodes.map((node) => ({
      ...node, position: previous.get(node.id)?.position ?? node.position,
      selected: previous.get(node.id)?.selected,
    }));
    replaceGraph(next);
  }, [workflow, replaceGraph, modelNames]);

  useEffect(() => {
    const current = graphRef.current;
    const nodes = current.nodes.map(node => {
      if (node.type !== 'agent') return node;
      const model = node.data.model;
      const modelName = model && model !== 'inherit' ? modelNames.get(model) || model : undefined;
      return node.data.model_name === modelName ? node : { ...node, data: { ...node.data, model_name: modelName } };
    });
    if (nodes.some((node, index) => node !== current.nodes[index])) replaceGraph({ ...current, nodes });
  }, [modelNames, replaceGraph]);

  const commit = useCallback((next: Graph, geometryOnly = false) => {
    const nodeById = new Map(next.nodes.map((node) => [node.id, node]));
    next = { ...next, edges: next.edges.map((edge) => resolveHandles(edge, nodeById)) };
    // Tool edges are the canvas representation of each Agent's tool bindings.
    if (!geometryOnly) {
      const bindings = new Map<string, string[]>();
      for (const edge of next.edges) {
        if (edge.data?.kind !== 'tool_call') continue;
        const tools = bindings.get(edge.source) || [];
        tools.push(edge.target);
        bindings.set(edge.source, tools);
      }
      next = { ...next, nodes: next.nodes.map((node) => {
        if (node.type !== 'agent') return node;
        const tools = bindings.get(node.id) || [];
        return tools.join('\0') === (node.data.tools || []).join('\0')
          ? node : { ...node, data: { ...node.data, tools } };
      }) };
    }
    const value = geometryOnly
      ? { ...workflowRef.current, edges: serializeEdges(next.edges) }
      : flowToWorkflow(next.nodes, next.edges, workflowRef.current);
    workflowRef.current = value;
    graphRevisionRef.current = workflowGraphRevision(value);
    rulesRef.current = createEdgeRules(next.nodes, next.edges, palette);
    replaceGraph(next);
    onChangeRef.current(value);
    setNotice('');
  }, [replaceGraph, palette]);

  const select = useCallback((target: GraphSelection) => {
    setSelection(target);
    const current = graphRef.current;
    replaceGraph({
      nodes: current.nodes.map((n) => ({ ...n, selected: target?.kind === 'node' && target.id === n.id })),
      edges: current.edges.map((e) => ({ ...e, selected: target?.kind === 'edge' && target.id === e.id })),
    });
  }, [replaceGraph]);

  const remove = useCallback((nodeIds: Set<string>, edgeIds: Set<string>) => {
    const current = graphRef.current;
    for (const id of [...nodeIds]) {
      const node = current.nodes.find((item) => item.id === id);
      if (node?.type === 'router') nodeIds.add(poolNodeId(id));
    }
    const edges = current.edges.filter((e) => !nodeIds.has(e.source) && !nodeIds.has(e.target) && !edgeIds.has(e.id));
    const removedFeedback = current.edges.some((e) =>
      e.source === 'verifier' && e.target === 'planner' && e.data?.kind === 'feedback' && !edges.includes(e));
    let nodes = current.nodes.filter((n) => !nodeIds.has(n.id)).map((n) => {
      if ((n.data.kind === 'planner' || n.id === 'planner') && (nodeIds.has('verifier') || removedFeedback)) {
        return { ...n, data: { ...n.data, verify: null } };
      }
      if (n.type === 'router' || n.type === 'pool') {
        const members = (n.data.members || n.data.candidates || []).filter((id) => !nodeIds.has(id.replace(/^blank:/, '')));
        return {
          ...n,
          data: {
            ...n.data,
            members: n.type === 'pool' ? members : n.data.members,
            candidates: n.type === 'router' ? members : n.data.candidates,
            scorer: n.data.scorer && nodeIds.has(n.data.scorer) ? null : n.data.scorer,
          },
        };
      }
      return n;
    });
    if (!nodes.some((n) => n.type === 'agent' && n.data.entry)) {
      const entry = nodes.find((n) => n.type === 'agent');
      nodes = nodes.map((n) => n === entry ? { ...n, data: { ...n.data, entry: true } } : n);
    }
    commit({ nodes, edges });
    setSelection((selected) => selected && (selected.kind === 'node' ? nodeIds : edgeIds).has(selected.id) ? null : selected);
  }, [commit]);

  const onNodesChange = useCallback((changes: NodeChange<GraphNode>[]) => {
    const removed = new Set(changes.filter((c) => c.type === 'remove').map((c) => c.id));
    if (removed.size) remove(removed, new Set());
    const current = graphRef.current;
    replaceGraph({ ...current, nodes: applyNodeChanges(changes.filter((c) => c.type !== 'remove'), current.nodes) });
    for (const change of changes) {
      if (change.type === 'select' && change.selected) setSelection({ kind: 'node', id: change.id });
    }
  }, [replaceGraph, remove]);

  const onEdgesChange = useCallback((changes: EdgeChange<GraphEdge>[]) => {
    const removed = new Set(changes.filter((c) => c.type === 'remove').map((c) => c.id));
    if (removed.size) remove(new Set(), removed);
    const current = graphRef.current;
    replaceGraph({ ...current, edges: applyEdgeChanges(changes.filter((c) => c.type !== 'remove'), current.edges) });
    for (const change of changes) {
      if (change.type === 'select' && change.selected) setSelection({ kind: 'edge', id: change.id });
    }
  }, [replaceGraph, remove]);

  const connect = useCallback((raw: Connection, kind: EdgeKind) => {
    const connection = kind === 'tool_call' ? rulesRef.current.normalize(raw) : raw;
    const error = rulesRef.current.error(connection, kind);
    if (error) { setNotice(error); return false; }
    const current = graphRef.current;
    const edge = { ...graphEdge(connection.source, connection.target, kind), ...connection };
    commit({ ...current, edges: [...current.edges, edge] });
    select({ kind: 'edge', id: edge.id });
    return true;
  }, [commit, select]);

  const addNode = useCallback((preset: GraphNodePreset, position: XYPosition) => {
    const current = graphRef.current;
    const kind = preset.nodeType;
    let id = preset.id;
    if (kind === 'tool' || preset.agentKind === 'tool' || preset.id === 'planner' || preset.agentKind === 'planner') {
      if (current.nodes.some((n) => n.id === id || (preset.agentKind === 'planner' && n.data.kind === 'planner'))) {
        setNotice(`${preset.id} 已在画布中。`); return;
      }
    } else {
      let suffix = 1;
      while (current.nodes.some((n) => n.id === id)) id = `${preset.id}_${suffix++}`;
    }
    const agentKind: AgentKind = preset.agentKind || 'blank';
    const toolProfile = preset.profile
      ? { ...preset.profile }
      : agentKind === 'tool'
        ? { backend: preset.backend || 'llm', llm_required: preset.llmRequired ?? true }
        : {};
    if (kind === 'tool' || agentKind === 'tool') {
      const selectedPool = current.nodes.find((node) => node.selected && node.type === 'pool');
      const selectedRouter = current.nodes.find((node) => node.selected && node.type === 'router');
      const pools = current.nodes.filter((node) => node.type === 'pool');
      const pool = selectedPool
        || (selectedRouter ? pools.find((node) => node.id === poolNodeId(selectedRouter.id)) : undefined)
        || (pools.length === 1 ? pools[0] : undefined);
      if (!pool) {
        setNotice(pools.length ? '请先选中要加入的 tool-agent pool。' : '请先添加 Router，再把 tool-agent 放进 pool。');
        return;
      }
      const members = pool.data.members || [];
      if (members.includes(preset.id)) {
        setNotice(`${preset.id} 已在这个 pool 中。`);
        return;
      }
      const nextMembers = [...members, preset.id];
      const routerId = pool.data.routerId || routerIdFromPool(pool.id);
      commit({
        ...current,
        nodes: current.nodes.map((node) => {
          if (node.id === pool.id) return { ...node, data: { ...node.data, members: nextMembers, candidates: nextMembers } };
          if (node.id === routerId) return { ...node, data: { ...node.data, candidates: nextMembers } };
          return node;
        }),
      });
      select({ kind: 'node', id: pool.id });
      return pool.id;
    }
    const node: GraphNode = kind === 'router'
      ? {
          id, type: 'router', position,
          data: { label: id, candidates: [], strategy: 'from_plan', scorer: null, meta: {} },
        }
      : {
          id, type: kind, position,
          data: {
            label: id, kind: agentKind, role: preset.role || 'agent',
            skills: [],
            tools: [], memory_scope: 'agent', system_prompt: '', model: 'inherit',
            trainable: agentKind !== 'verifier', profile: toolProfile, meta: {},
            backend: preset.backend, llm_required: preset.llmRequired, description: preset.description,
            entry: !current.nodes.some((n) => n.type === 'agent'),
          },
        };
    const pool = kind === 'router' ? {
      id: poolNodeId(id), type: 'pool' as const, position: { x: position.x + 280, y: position.y },
      data: { label: 'tool-agent pool', members: [] as string[], routerId: id, candidates: [] as string[] },
    } : null;
    commit({
      ...current,
      nodes: pool ? [...current.nodes, node, pool] : [...current.nodes, node],
      edges: pool ? [...current.edges, graphEdge(id, pool.id, 'route')] : current.edges,
    });
    select({ kind: 'node', id });
    return id;
  }, [commit, select]);

  const setEntry = useCallback((id: string) => {
    const current = graphRef.current;
    if (current.nodes.find((node) => node.id === id)?.type !== 'agent') return;
    commit({ ...current, nodes: current.nodes.map((n) =>
      n.type === 'agent' ? { ...n, data: { ...n.data, entry: n.id === id } } : n) });
  }, [commit]);

  const updateSelected = useCallback((patch: Partial<GraphNodeData>) => {
    if (selection?.kind !== 'node') return;
    const current = graphRef.current;
    const selected = current.nodes.find((n) => n.id === selection.id);
    if (!selected) return;
    let nodes = current.nodes.map((n) => n === selected ? { ...n, data: { ...n.data, ...patch } } : n);
    if (selected.type === 'pool' && patch.members) {
      const routerId = selected.data.routerId || routerIdFromPool(selected.id);
      nodes = nodes.map((node) => node.id === routerId
        ? { ...node, data: { ...node.data, candidates: patch.members } }
        : node.id === selected.id
          ? { ...node, data: { ...node.data, members: patch.members, candidates: patch.members } }
          : node);
    }
    let edges = current.edges;
    if (patch.tools && selected.type === 'agent') {
      edges = edges.filter((e) => e.source !== selected.id || e.data?.kind !== 'tool_call' || patch.tools?.includes(e.target));
      for (const [index, tool] of patch.tools.entries()) {
        if (!nodes.some((n) => n.id === tool)) nodes.push({
          id: tool, type: 'tool',
          position: { x: selected.position.x + 260, y: selected.position.y + index * 120 },
          data: { label: tool, role: 'tool' },
        });
        if (!edges.some((e) => e.source === selected.id && e.target === tool && e.data?.kind === 'tool_call')) {
          const candidate = { source: selected.id, target: tool, sourceHandle: null, targetHandle: null };
          const error = createEdgeRules(nodes, edges, palette).error(candidate, 'tool_call');
          if (error) { setNotice(error); return; }
          edges.push(graphEdge(selected.id, tool, 'tool_call'));
        }
      }
    }
    if (patch.verify !== undefined && selected.type === 'agent' && selected.data.kind === 'planner') {
      edges = edges.filter((e) => patch.verify || !(e.source === 'verifier' && e.target === selected.id && e.data?.kind === 'feedback'));
      if (patch.verify) {
        if (!nodes.some((n) => n.id === 'verifier')) nodes.push({
          id: 'verifier', type: 'agent',
          position: { x: selected.position.x + 280, y: selected.position.y + 160 },
          data: {
            label: 'verifier', kind: 'verifier', role: 'verifier',
            skills: ['verifier'], tools: [], memory_scope: 'agent', system_prompt: '',
            model: 'inherit', trainable: false, profile: {}, meta: {},
          },
        });
        if (!edges.some((e) => e.source === 'verifier' && e.target === selected.id && e.data?.kind === 'feedback')) {
          const error = createEdgeRules(nodes, edges, palette).error({
            source: 'verifier', target: selected.id, sourceHandle: null, targetHandle: null,
          }, 'feedback');
          if (error) { setNotice(error); return; }
          edges = [...edges, graphEdge('verifier', selected.id, 'feedback')];
        }
      }
    }
    commit({ nodes, edges });
  }, [commit, selection, palette]);

  const updateEdge = useCallback((id: string, patch: EdgePatch): boolean => {
    const current = graphRef.current;
    const edge = current.edges.find((e) => e.id === id);
    if (!edge) { setNotice('连线已不存在。'); return false; }
    const { kind: requestedKind, ...endpoints } = patch;
    const kind = requestedKind ?? edge.data?.kind ?? 'message';
    const connection = rulesRef.current.normalizeEdit({ ...edgeConnection(edge), ...endpoints }, edge, kind);
    const geometryOnly = connection.source === edge.source && connection.target === edge.target && kind === edge.data?.kind;
    if (!geometryOnly) {
      const error = rulesRef.current.error(connection, kind, id);
      if (error) { setNotice(error); return false; }
    }
    const defaults = defaultHandles(rulesRef.current.nodeById.get(connection.source), rulesRef.current.nodeById.get(connection.target));
    if (connection.source !== edge.source && patch.sourceHandle === undefined) connection.sourceHandle = defaults.sourceHandle;
    if (connection.target !== edge.target && patch.targetHandle === undefined) connection.targetHandle = defaults.targetHandle;
    const removedFeedback = edge.source === 'verifier' && edge.target === 'planner' && edge.data?.kind === 'feedback'
      && (connection.source !== 'verifier' || connection.target !== 'planner' || kind !== 'feedback');
    const nodes = removedFeedback
      ? current.nodes.map((n) => (n.data.kind === 'planner' || n.id === 'planner') ? { ...n, data: { ...n.data, verify: null } } : n)
      : current.nodes;
    commit({ nodes, edges: current.edges.map((e) =>
      e === edge ? { ...e, ...connection, data: { ...e.data, kind }, selected: true } : e) }, geometryOnly);
    return true;
  }, [commit]);

  const applyTemplate = useCallback((template: WorkflowSpec) => {
    const next = { ...workflowRef.current, ...template };
    workflowRef.current = next;
    graphRevisionRef.current = workflowGraphRevision(next);
    replaceGraph(workflowToFlow(next));
    setSelection(null);
    setNotice('');
    onChangeRef.current(next);
  }, [replaceGraph]);

  const deleteSelected = useCallback(() => {
    if (!selection) return;
    remove(new Set(selection.kind === 'node' ? [selection.id] : []), new Set(selection.kind === 'edge' ? [selection.id] : []));
  }, [selection, remove]);

  return {
    nodes: graph.nodes, edges: graph.edges, selection, select, notice, setNotice,
    selected: selection?.kind === 'node' ? graph.nodes.find((n) => n.id === selection.id) ?? null : null,
    selectedEdge: selection?.kind === 'edge' ? graph.edges.find((e) => e.id === selection.id) ?? null : null,
    onNodesChange, onEdgesChange, connect, rules, addNode, setEntry, updateSelected, updateEdge,
    deleteSelected, applyTemplate,
  };
}

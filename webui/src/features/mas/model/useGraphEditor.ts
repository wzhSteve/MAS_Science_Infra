import { useCallback, useEffect, useRef, useState } from 'react';
import {
  applyEdgeChanges, applyNodeChanges,
  type Connection, type EdgeChange, type NodeChange, type XYPosition,
} from '@xyflow/react';
import type { WorkflowSpec } from '../../../shared/api/types';
import type { GraphNodeData, GraphNode, GraphEdge, GraphSelection } from '../types';
import { flowToWorkflow, graphEdge, workflowToFlow } from './workflowGraph';

type Graph = { nodes: GraphNode[]; edges: GraphEdge[] };

export function useGraphEditor(workflow: WorkflowSpec, onChange: (workflow: WorkflowSpec) => void) {
  const [graph, setGraph] = useState<Graph>(() => workflowToFlow(workflow));
  const [selection, setSelection] = useState<GraphSelection>(null);
  const [notice, setNotice] = useState('');
  const graphRef = useRef(graph);
  const workflowRef = useRef(workflow);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const replaceGraph = useCallback((next: Graph) => {
    graphRef.current = next;
    setGraph(next);
  }, []);

  useEffect(() => {
    if (workflow === workflowRef.current) return;
    workflowRef.current = workflow;
    const next = workflowToFlow(workflow);
    const previous = new Map(graphRef.current.nodes.map((node) => [node.id, node]));
    next.nodes = next.nodes.map((node) => ({
      ...node, position: previous.get(node.id)?.position ?? node.position,
      selected: previous.get(node.id)?.selected,
    }));
    replaceGraph(next);
  }, [workflow, replaceGraph]);

  const commit = useCallback((next: Graph) => {
    // Tool edges are the canvas representation of each Agent's tool bindings.
    next = { ...next, nodes: next.nodes.map((node) => {
      if (node.type === 'tool') return node;
      const tools = next.edges.filter((e) => e.source === node.id && e.data?.kind === 'tool_call').map((e) => e.target);
      return tools.join('\0') === (node.data.tools || []).join('\0')
        ? node : { ...node, data: { ...node.data, tools } };
    }) };
    const value = flowToWorkflow(next.nodes, next.edges, workflowRef.current);
    workflowRef.current = value;
    replaceGraph(next);
    onChangeRef.current(value);
    setNotice('');
  }, [replaceGraph]);

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
    const edges = current.edges.filter((e) => !nodeIds.has(e.source) && !nodeIds.has(e.target) && !edgeIds.has(e.id));
    const removedFeedback = current.edges.some((e) =>
      e.source === 'verifier' && e.target === 'hub' && e.data?.kind === 'feedback' && !edges.includes(e));
    let nodes = current.nodes.filter((n) => !nodeIds.has(n.id)).map((n) =>
      n.id === 'hub' && (nodeIds.has('verifier') || removedFeedback)
        ? { ...n, data: { ...n.data, verify: null } } : n);
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

  const connectionError = useCallback((connection: Connection, kind: string, replacing?: string) => {
    const { nodes, edges } = graphRef.current;
    const source = nodes.find((n) => n.id === connection.source);
    const target = nodes.find((n) => n.id === connection.target);
    if (!source || !target) return '请选择存在的节点。';
    if (source.id === target.id) return '请连接两个不同的节点。';
    if (source.type === 'tool') return 'Tool 只能作为工具调用的终点。';
    if (kind === 'tool_call' ? target.type !== 'tool' : target.type === 'tool') return 'Agent 与 Tool 之间只能使用工具调用。';
    if (edges.some((e) => e.id !== replacing && e.source === source.id && e.target === target.id && e.data?.kind === kind)) {
      return '这条连线已经存在。';
    }
    if (kind === 'route' && edges.some((e) => e.id !== replacing && e.target === target.id && e.data?.kind === 'route')) {
      return '一个 Agent 只能有一条输入路由。';
    }
    return '';
  }, []);

  const connect = useCallback((connection: Connection, kind: string) => {
    const error = connectionError(connection, kind);
    if (error) { setNotice(error); return; }
    const current = graphRef.current;
    const edge = graphEdge(connection.source, connection.target, kind);
    commit({ ...current, edges: [...current.edges, edge] });
    select({ kind: 'edge', id: edge.id });
  }, [commit, connectionError, select]);

  const addNode = useCallback((kind: 'agent' | 'tool', type: string, position: XYPosition) => {
    const current = graphRef.current;
    let id = type;
    if (kind === 'tool' || type === 'hub') {
      if (current.nodes.some((n) => n.id === id)) { setNotice(`${type} 已在画布中。`); return; }
    } else {
      let suffix = 1;
      while (current.nodes.some((n) => n.id === id)) id = `${type}_${suffix++}`;
    }
    const node: GraphNode = {
      id, type: kind, position,
      data: {
        label: id, role: kind === 'tool' ? 'tool' : type,
        skills: type === 'verifier' ? ['verifier'] : kind === 'agent' ? ['react_loop'] : [],
        tools: [], trainable: type !== 'verifier',
        entry: kind === 'agent' && !current.nodes.some((n) => n.type === 'agent'),
        system_prompt: '',
      },
    };
    commit({ ...current, nodes: [...current.nodes, node] });
    select({ kind: 'node', id });
    return id;
  }, [commit, select]);

  const setEntry = useCallback((id: string) => {
    const current = graphRef.current;
    commit({ ...current, nodes: current.nodes.map((n) =>
      n.type === 'agent' ? { ...n, data: { ...n.data, entry: n.id === id } } : n) });
  }, [commit]);

  const updateSelected = useCallback((patch: Partial<GraphNodeData>) => {
    if (selection?.kind !== 'node') return;
    const current = graphRef.current;
    const selected = current.nodes.find((n) => n.id === selection.id);
    if (!selected) return;
    let nodes = current.nodes.map((n) => n === selected ? { ...n, data: { ...n.data, ...patch } } : n);
    let edges = current.edges;
    if (patch.tools) {
      edges = edges.filter((e) => e.source !== selected.id || e.data?.kind !== 'tool_call');
      for (const [index, tool] of patch.tools.entries()) {
        if (!nodes.some((n) => n.id === tool)) nodes.push({
          id: tool, type: 'tool',
          position: { x: selected.position.x + 260, y: selected.position.y + index * 120 },
          data: { label: tool, role: 'tool' },
        });
        edges.push(graphEdge(selected.id, tool, 'tool_call'));
      }
    }
    if (patch.verify !== undefined && selected.id === 'hub') {
      edges = edges.filter((e) => !(e.source === 'verifier' && e.target === 'hub' && e.data?.kind === 'feedback'));
      if (patch.verify) {
        if (!nodes.some((n) => n.id === 'verifier')) nodes.push({
          id: 'verifier', type: 'agent',
          position: { x: selected.position.x + 280, y: selected.position.y + 160 },
          data: { label: 'verifier', role: 'verifier', skills: ['verifier'], tools: [], trainable: false },
        });
        edges.push(graphEdge('verifier', 'hub', 'feedback'));
      }
    }
    commit({ nodes, edges });
  }, [commit, selection]);

  const updateEdge = useCallback((id: string, kind: string) => {
    const current = graphRef.current;
    const edge = current.edges.find((e) => e.id === id);
    if (!edge) return;
    const error = connectionError({ ...edge, sourceHandle: null, targetHandle: null }, kind, id);
    if (error) { setNotice(error); return; }
    const nodes = edge.source === 'verifier' && edge.target === 'hub' && kind !== 'feedback'
      ? current.nodes.map((n) => n.id === 'hub' ? { ...n, data: { ...n.data, verify: null } } : n)
      : current.nodes;
    commit({ nodes, edges: current.edges.map((e) =>
      e === edge ? { ...graphEdge(e.source, e.target, kind, id), selected: true } : e) });
  }, [commit, connectionError]);

  const applyTemplate = useCallback((template: WorkflowSpec) => {
    const next = { ...workflowRef.current, ...template };
    workflowRef.current = next;
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
    onNodesChange, onEdgesChange, connect, connectionError, addNode, setEntry, updateSelected, updateEdge,
    deleteSelected, applyTemplate,
  };
}

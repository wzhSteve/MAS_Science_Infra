import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  addEdge, applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type Node, type NodeChange, type XYPosition,
} from '@xyflow/react';
import type { WorkflowSpec } from '../../../shared/api/types';
import type { GraphNodeData, SelectedGraphNode } from '../types';
import { flowToWorkflow, workflowToFlow } from './workflowGraph';

type Graph = { nodes: Node[]; edges: Edge[] };

export function useGraphEditor(workflow: WorkflowSpec, onChange: (workflow: WorkflowSpec) => void) {
  const [graph, setGraph] = useState<Graph>(() => workflowToFlow(workflow));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [edgeKind, setEdgeKind] = useState('message');
  const graphRef = useRef(graph);
  const workflowRef = useRef(workflow);
  const onChangeRef = useRef(onChange);
  const kindRef = useRef(edgeKind);
  const selectedRef = useRef(selectedId);
  const lastWorkflow = useRef(workflow);
  onChangeRef.current = onChange;
  kindRef.current = edgeKind;
  selectedRef.current = selectedId;

  const replaceGraph = useCallback((next: Graph) => {
    graphRef.current = next;
    setGraph(next);
  }, []);

  useEffect(() => {
    if (workflow === lastWorkflow.current) return;
    lastWorkflow.current = workflow;
    workflowRef.current = workflow;
    const next = workflowToFlow(workflow);
    // External updates refresh data, not the user's existing canvas positions.
    const previous = new Map(graphRef.current.nodes.map((node) => [node.id, node]));
    next.nodes = next.nodes.map((node) => {
      const existing = previous.get(node.id);
      return existing ? { ...node, position: existing.position, selected: existing.selected } : node;
    });
    replaceGraph(next);
  }, [workflow, replaceGraph]);

  const emit = useCallback((next: Graph) => {
    const value = flowToWorkflow(next.nodes, next.edges, workflowRef.current);
    workflowRef.current = value;
    lastWorkflow.current = value;
    onChangeRef.current(value);
    return value;
  }, []);

  const commit = useCallback((next: Graph) => {
    const value = emit(next);
    const missingTools = value.tools.some((tool) => !next.nodes.some((node) => node.id === tool));
    const missingVerifier = value.hub.verify && (
      !next.nodes.some((node) => node.id === 'verifier') || !next.edges.some((edge) => edge.source === 'verifier'));
    if (missingTools || missingVerifier) {
      // Materialize only missing derived nodes/edges, not the whole canvas.
      const implied = workflowToFlow(value);
      const nodes = implied.nodes.filter((node) => !next.nodes.some((existing) => existing.id === node.id));
      const edges = implied.edges.filter((edge) =>
        !next.edges.some((existing) => existing.source === edge.source && existing.target === edge.target
          && existing.data?.kind === edge.data?.kind));
      replaceGraph({ nodes: [...next.nodes, ...nodes], edges: [...next.edges, ...edges] });
    } else {
      replaceGraph(next);
    }
  }, [replaceGraph, emit]);

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    const current = graphRef.current;
    replaceGraph({ ...current, nodes: applyNodeChanges(changes, current.nodes) });
    // React Flow owns keyboard selection/deletion; keep the inspector in sync.
    for (const change of changes) {
      if (change.type === 'select' && change.selected) setSelectedId(change.id);
    }
  }, [replaceGraph]);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    const current = graphRef.current;
    replaceGraph({ ...current, edges: applyEdgeChanges(changes, current.edges) });
  }, [replaceGraph]);

  const onNodeDragStop = useCallback(() => {
    // Preserve the existing drag-stop emission, never serialize drag frames.
    emit(graphRef.current);
  }, [emit]);

  const onConnect = useCallback((connection: Connection) => {
    const current = graphRef.current;
    const source = current.nodes.find((node) => node.id === connection.source);
    const target = current.nodes.find((node) => node.id === connection.target);
    const kind = source?.type === 'tool' || target?.type === 'tool' ? 'tool_call' : kindRef.current;
    const tools = (source?.data.tools as string[]) || [];
    const nodes = kind === 'tool_call' && source && source.type !== 'tool' && target && !tools.includes(target.id)
      ? current.nodes.map((node) => node === source ? { ...node, data: { ...node.data, tools: [...tools, target.id] } } : node)
      : current.nodes;
    commit({
      nodes,
      edges: addEdge({ ...connection, label: kind, data: { kind } }, current.edges),
    });
  }, [commit]);

  const addNode = useCallback((kind: 'agent' | 'tool', id: string, role: string, position?: XYPosition) => {
    const current = graphRef.current;
    if (current.nodes.some((node) => node.id === id)) return;
    const node: Node = {
      id, type: kind,
      position: position || { x: 120 + current.nodes.length * 36, y: 90 + current.nodes.length * 28 },
      data: {
        label: id, role,
        skills: role === 'verifier' ? ['verifier'] : kind === 'agent' ? ['react_loop'] : [],
        tools: [], trainable: role !== 'verifier', entry: false, system_prompt: '',
      },
    };
    commit({ ...current, nodes: [...current.nodes, node] });
  }, [commit]);

  const setEntry = useCallback((id: string) => {
    const current = graphRef.current;
    if (!current.nodes.some((node) => node.id === id && node.type !== 'tool')) return;
    const nodes = current.nodes.map((node) => {
      const entry = node.id === id;
      if (node.type === 'tool' || Boolean(node.data.entry) === entry) return node;
      return { ...node, data: { ...node.data, entry } };
    });
    setSelectedId(id);
    commit({ ...current, nodes });
  }, [commit]);

  const updateSelected = useCallback((patch: Partial<GraphNodeData>) => {
    const current = graphRef.current;
    if (!current.nodes.some((node) => node.id === selectedRef.current)) return;
    const next = {
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === selectedRef.current ? { ...node, data: { ...node.data, ...patch } } : node),
    };
    commit(next);
  }, [commit]);

  const applyTemplate = useCallback((template: WorkflowSpec) => {
    const next = { ...workflowRef.current, ...template };
    workflowRef.current = next;
    lastWorkflow.current = next;
    replaceGraph(workflowToFlow(next));
    setSelectedId(null);
    onChangeRef.current(next);
  }, [replaceGraph]);

  const onNodeClick = useCallback((_: unknown, node: Node) => setSelectedId(node.id), []);
  const onPaneClick = useCallback(() => setSelectedId(null), []);
  const selectedNode = graph.nodes.find((node) => node.id === selectedId);
  const selected = useMemo<SelectedGraphNode | null>(() =>
    selectedNode ? { id: selectedNode.id, type: selectedNode.type, data: selectedNode.data } : null,
  [selectedNode?.id, selectedNode?.type, selectedNode?.data]);

  return {
    nodes: graph.nodes, edges: graph.edges, selected, edgeKind, setEdgeKind,
    onNodesChange, onEdgesChange, onConnect, onNodeClick, onPaneClick, onNodeDragStop,
    addNode, setEntry, updateSelected, applyTemplate,
  };
}

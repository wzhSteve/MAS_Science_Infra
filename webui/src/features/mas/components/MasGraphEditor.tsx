import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ReactFlowProvider, useReactFlow, type Connection, type XYPosition, type ReactFlowProps } from '@xyflow/react';
import { AlertCircle, X } from 'lucide-react';
import type { Palette, WorkflowSpec } from '../../../shared/api/types';
import type { EditorPanel, GraphNode, GraphEdge, TraceFocusRequest } from '../types';
import { executableInfo } from '../model/workflowGraph';
import { edgeConnection } from '../model/edgeRules';
import { edgeLanes } from '../model/edgeGeometry';
import { useGraphEditor } from '../model/useGraphEditor';
import { Button } from '../../../shared/ui/button';
import { GraphPalette, type LibraryTab } from './GraphPalette';
import { GraphCanvas } from './GraphCanvas';
import { NodeInspector } from './NodeInspector';
import { EdgeInspector } from './EdgeInspector';
import { ConnectionPicker } from './ConnectionPicker';
import { GraphInteractionContext } from './GraphInteractionContext';

type Props = {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  active: boolean;
  panel: EditorPanel;
  onPanelChange: (panel: EditorPanel) => void;
  libraryOpen: boolean;
  onLibraryOpenChange: (open: boolean) => void;
  settingsContent: ReactNode;
  traceFocus?: TraceFocusRequest | null;
};

function GraphWorkbench({ workflow, palette, onChange, active, panel, onPanelChange, libraryOpen, onLibraryOpenChange, settingsContent, traceFocus }: Props) {
  const graph = useGraphEditor(workflow, onChange, palette);
  const flow = useReactFlow<GraphNode, GraphEdge>();
  const root = useRef<HTMLDivElement>(null);
  const lastTraceFocus = useRef<number | null>(null);
  const [traceInfo, setTraceInfo] = useState('');
  const [libraryTab, setLibraryTab] = useState<LibraryTab>('agent');
  const [connection, setConnection] = useState<{ value: Connection; anchor: XYPosition } | null>(null);
  const [reconnecting, setReconnecting] = useState<GraphEdge | null>(null);
  const reconnectRef = useRef<GraphEdge | null>(null);
  const executable = useMemo(() => executableInfo(workflow), [workflow]);
  const selectedEdge = graph.selectedEdge;
  const nodes = useMemo(() => graph.nodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      issue: node.id === executable.nodeId ? executable.reason : undefined,
      related: selectedEdge?.source === node.id || selectedEdge?.target === node.id,
    },
  })), [graph.nodes, executable, selectedEdge]);
  const lanes = useMemo(() => edgeLanes(graph.edges), [graph.edges]);
  const edges = useMemo(() => graph.edges.map((edge) => ({
    ...edge, reconnectable: Boolean(edge.selected),
    data: { ...edge.data, kind: edge.data?.kind || 'message', lane: lanes.get(edge.id),
      issue: graph.rules.error(edgeConnection(edge), edge.data?.kind || 'message', edge.id) },
  })), [graph.edges, graph.rules, lanes]);

  const closePanels = useCallback(() => {
    onPanelChange(null);
    if (panel !== 'settings') graph.select(null);
    setConnection(null);
  }, [onPanelChange, graph.select, panel]);

  useEffect(() => {
    if (panel) setConnection(null);
  }, [panel]);

  useEffect(() => { setConnection(null); }, [workflow]);

  useEffect(() => {
    if (!connection || !active) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Element && !event.target.closest('.mas-connection-picker')) setConnection(null);
    };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [Boolean(connection), active]);

  useEffect(() => {
    if (!active) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) closePanels();
    };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [active, closePanels]);

  const openLibrary = (templates = false) => {
    setLibraryTab(templates ? 'template' : 'agent');
    onLibraryOpenChange(true);
  };

  const visibleArea = useCallback(() => {
    const bounds = root.current?.getBoundingClientRect();
    if (!bounds) return null;
    const right = Math.max(240, bounds.width - (bounds.width > 640 ? 360 : 20));
    const toolboxWidth = root.current?.querySelector<HTMLElement>('.mas-library')?.offsetWidth ?? 0;
    const inset = toolboxWidth ? toolboxWidth + 28 : 20;
    const left = right - inset >= 220 ? inset : 20;
    return { bounds, left, right };
  }, []);

  const revealNode = useCallback((id: string) => {
    requestAnimationFrame(() => {
      const area = visibleArea();
      const node = flow.getNode(id);
      if (!area || !node) return;
      const { bounds } = area;
      const { x, y, zoom } = flow.getViewport();
      const right = node.position.x * zoom + x + (node.measured?.width || 200) * zoom;
      const left = node.position.x * zoom + x;
      const top = node.position.y * zoom + y;
      const bottom = top + (node.measured?.height || 110) * zoom;
      const dx = right > area.right ? area.right - right : left < area.left ? area.left - left : 0;
      const dy = bottom > bounds.height - 70 ? bounds.height - 90 - bottom : top < 20 ? 20 - top : 0;
      if (dx || dy) void flow.setViewport({ x: x + dx, y: y + dy, zoom });
    });
  }, [flow, visibleArea]);

  useEffect(() => {
    if (!active || !traceFocus || lastTraceFocus.current === traceFocus.token) return;
    lastTraceFocus.current = traceFocus.token;
    const agent = graph.nodes.find(node => node.id === traceFocus.agentId && node.type === 'agent');
    if (!agent) {
      setTraceInfo(`历史记录中的 Agent ${traceFocus.agentId} 不在当前画布中，未修改画布。`);
      return;
    }
    const bindings = traceFocus.toolName ? graph.edges.filter(edge =>
      edge.source === agent.id && edge.target === traceFocus.toolName && edge.data?.kind === 'tool_call') : [];
    graph.select(bindings.length === 1 ? { kind: 'edge', id: bindings[0].id } : { kind: 'node', id: agent.id });
    setTraceInfo(traceFocus.toolName && bindings.length !== 1
      ? '当前画布没有唯一对应的工具绑定，仅定位到调用者 Agent。'
      : `已定位历史执行涉及的实体${traceFocus.agentExecutionId ? ` · ${traceFocus.agentExecutionId}` : ''}，这不是实时执行状态。`);
    onPanelChange(null);
    setConnection(null);
    revealNode(agent.id);
  }, [active, traceFocus, graph.nodes, graph.edges, graph.select, graph.setNotice, onPanelChange, revealNode]);

  const add = (kind: 'agent' | 'tool', type: string, position?: XYPosition) => {
    const area = visibleArea();
    if (!area) return;
    const { bounds } = area;
    const center = flow.screenToFlowPosition({
      x: bounds.left + (area.left + area.right) / 2 - 100,
      y: bounds.top + bounds.height / 2 - 50,
    });
    if (!position) {
      while (graph.nodes.some((node) => Math.abs(node.position.x - center.x) < 24 && Math.abs(node.position.y - center.y) < 24)) {
        center.x += 28;
        center.y += 28;
      }
    }
    const id = graph.addNode(kind, type, position || center);
    onPanelChange(null);
    setConnection(null);
    if (id) revealNode(id);
  };

  const onSelectEdge = useCallback((id: string) => {
    graph.select({ kind: 'edge', id });
    onPanelChange(null);
    setConnection(null);
  }, [graph.select, onPanelChange]);
  const interaction = useMemo(() => ({ rules: graph.rules, reconnecting, onSelectEdge }), [graph.rules, reconnecting, onSelectEdge]);

  const onConnect = (raw: Connection) => {
    const value = graph.rules.normalize(raw);
    const options = graph.rules.options(value);
    const valid = options.filter((k) => !k.reason);
    if (!valid.length) { graph.setNotice(options[0]?.reason || '没有可用的连线关系。'); return; }
    if (valid.length === 1) {
      graph.connect(value, valid[0].kind);
      onPanelChange(null);
    } else {
      graph.select(null);
      onPanelChange(null);
      const bounds = root.current?.getBoundingClientRect();
      const target = graph.rules.nodeById.get(value.target);
      const point = flow.flowToScreenPosition(target?.position || { x: 0, y: 0 });
      setConnection({ value, anchor: { x: point.x - (bounds?.left || 0), y: point.y - (bounds?.top || 0) } });
    }
  };

  const isValidConnection: NonNullable<ReactFlowProps<GraphNode, GraphEdge>['isValidConnection']> = (raw) => {
    const value = { source: raw.source, target: raw.target, sourceHandle: raw.sourceHandle ?? null, targetHandle: raw.targetHandle ?? null };
    const existing = reconnectRef.current;
    if (existing) {
      const next = graph.rules.normalizeEdit(value, existing);
      return next.source === existing.source && next.target === existing.target
        || !graph.rules.error(next, existing.data?.kind || 'message', existing.id);
    }
    return graph.rules.options(value).some((option) => !option.reason);
  };

  const onConnectEnd: NonNullable<ReactFlowProps<GraphNode, GraphEdge>['onConnectEnd']> = (event, state) => {
    if (reconnectRef.current) return;
    const point = 'changedTouches' in event ? event.changedTouches[0] : event;
    const bounds = root.current?.getBoundingClientRect();
    if (point && bounds) setConnection((previous) => previous && {
      ...previous, anchor: { x: point.clientX - bounds.left, y: point.clientY - bounds.top },
    });
    if (!state.isValid && state.fromNode && state.toNode) {
      const value = graph.rules.normalize({
        source: state.fromNode.id, target: state.toNode.id,
        sourceHandle: state.fromHandle?.id ?? null, targetHandle: state.toHandle?.id ?? null,
      });
      const options = graph.rules.options(value);
      if (!options.some((option) => !option.reason)) {
        graph.setNotice(options[0]?.reason || '没有可用的关系。');
        const existing = graph.edges.find((edge) => edge.source === value.source && edge.target === value.target);
        if (existing) onSelectEdge(existing.id);
      }
    }
  };

  return <GraphInteractionContext.Provider value={interaction}><div className="mas-workbench" ref={root} tabIndex={-1} onKeyDown={(event) => {
    if (event.key === 'Escape') { event.stopPropagation(); closePanels(); }
    else if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable="true"]')) {
      event.stopPropagation();
    }
  }}>
    <GraphCanvas nodes={nodes} edges={edges} active={active}
      onNodesChange={graph.onNodesChange} onEdgesChange={graph.onEdgesChange} onConnect={onConnect}
      isValidConnection={isValidConnection} onConnectStart={() => setConnection(null)} onConnectEnd={onConnectEnd}
      onMoveStart={() => setConnection(null)}
      onReconnectStart={(_, edge) => { reconnectRef.current = edge; setReconnecting(edge); setConnection(null); }}
      onReconnect={(edge, value) => { graph.updateEdge(edge.id, value); }}
      onReconnectEnd={(_, edge, __, state) => {
        if (!state.isValid && state.toNode) graph.setNotice('未改接到合法对象，原连线已保留。');
        reconnectRef.current = null; setReconnecting(null);
      }}
      onNodeClick={(_, node) => { graph.select({ kind: 'node', id: node.id }); onPanelChange(null); setConnection(null); revealNode(node.id); }}
      onEdgeClick={(_, edge) => onSelectEdge(edge.id)}
      onPaneClick={() => { if (panel !== 'settings') closePanels(); else { graph.select(null); setConnection(null); } }}
      onAdd={add} onOpenLibrary={openLibrary} onError={graph.setNotice} />
    {traceInfo && <div className="mas-trace-location" role="status"><span>{traceInfo}</span>
      <Button size="sm" variant="ghost" aria-label="关闭历史定位说明" onClick={() => setTraceInfo('')}><X size={14} /></Button>
    </div>}
    {(graph.notice || !executable.ok) && <div className="mas-graph-notice" role="alert">
      <AlertCircle size={15} /><span>{graph.notice || executable.reason}</span>
      {!graph.notice && executable.nodeId && <Button size="sm" variant="ghost" onClick={() => {
        if (executable.nodeId) {
          graph.select({ kind: 'node', id: executable.nodeId }); onPanelChange(null); revealNode(executable.nodeId);
        }
      }}>定位</Button>}
      {graph.notice && <Button size="sm" variant="ghost" aria-label="关闭提示" onClick={() => graph.setNotice('')}><X size={14} /></Button>}
    </div>}
    {libraryOpen && <GraphPalette palette={palette} nodes={graph.nodes} tab={libraryTab} onTabChange={setLibraryTab} onAdd={add}
      onClose={() => onLibraryOpenChange(false)} onTemplate={(template) => {
        graph.applyTemplate(template); onPanelChange(null); setConnection(null);
        requestAnimationFrame(() => void flow.fitView({ padding: 0.25, maxZoom: 1 }));
      }} />}
    {settingsContent}
    {panel === null && graph.selected ? <NodeInspector selected={graph.selected} palette={palette} entryId={workflow.entry_agent || 'hub'}
        onPatch={graph.updateSelected} onEntry={graph.setEntry} onDelete={graph.deleteSelected} onClose={closePanels} />
      : panel === null && selectedEdge ? <EdgeInspector key={selectedEdge.id} edge={selectedEdge} nodes={graph.nodes}
        rules={graph.rules} topology={workflow.topology}
        onChange={(patch) => graph.updateEdge(selectedEdge.id, patch)} onDelete={graph.deleteSelected} onClose={closePanels} /> : null}
    {connection && panel === null && <ConnectionPicker connection={connection.value} anchor={connection.anchor}
      workspace={root} options={graph.rules.options(connection.value)}
      onClose={() => { setConnection(null); root.current?.focus(); }}
      onChoose={(kind) => {
        if (graph.connect(connection.value, kind)) { setConnection(null); root.current?.focus(); }
      }} />}
  </div></GraphInteractionContext.Provider>;
}

export default memo(function MasGraphEditor(props: Props) {
  return <ReactFlowProvider><GraphWorkbench {...props} /></ReactFlowProvider>;
});

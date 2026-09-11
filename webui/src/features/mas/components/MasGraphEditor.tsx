import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ReactFlowProvider, useReactFlow, type Connection, type XYPosition } from '@xyflow/react';
import { AlertCircle, X } from 'lucide-react';
import type { Palette, WorkflowSpec } from '../../../shared/api/types';
import type { Notice } from '../../../shared/components/InlineNotice';
import type { EditorPanel, GraphNode, GraphEdge } from '../types';
import { EDGE_LABELS, executableInfo } from '../model/workflowGraph';
import { useGraphEditor } from '../model/useGraphEditor';
import { Button } from '../../../shared/ui/button';
import { GraphPalette, type LibraryTab } from './GraphPalette';
import { GraphCanvas } from './GraphCanvas';
import { NodeInspector } from './NodeInspector';
import { EdgeInspector } from './EdgeInspector';
import { WorkflowSettings } from './WorkflowSettings';

type Props = {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  active: boolean;
  panel: EditorPanel;
  onPanelChange: (panel: EditorPanel) => void;
  libraryOpen: boolean;
  onLibraryOpenChange: (open: boolean) => void;
  rlDirty: boolean;
  rlNotice: Notice | null;
  rlSettings: ReactNode;
};

function GraphWorkbench({ workflow, palette, onChange, active, panel, onPanelChange, libraryOpen, onLibraryOpenChange, rlDirty, rlNotice, rlSettings }: Props) {
  const graph = useGraphEditor(workflow, onChange);
  const flow = useReactFlow<GraphNode, GraphEdge>();
  const root = useRef<HTMLDivElement>(null);
  const [libraryTab, setLibraryTab] = useState<LibraryTab>('agent');
  const [connection, setConnection] = useState<Connection | null>(null);
  const executable = useMemo(() => executableInfo(workflow), [workflow]);
  const nodes = useMemo(() => graph.nodes.map((n) =>
    n.id === executable.nodeId ? { ...n, data: { ...n.data, issue: executable.reason } } : n), [graph.nodes, executable]);

  const closePanels = useCallback(() => {
    onPanelChange(null);
    graph.select(null);
    setConnection(null);
  }, [onPanelChange, graph.select]);

  useEffect(() => {
    if (panel) { graph.select(null); setConnection(null); }
  }, [panel, graph.select]);

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

  const kindsFor = (value: Connection, replacing?: string) => {
    const target = graph.nodes.find((n) => n.id === value.target);
    const kinds = target?.type === 'tool' ? ['tool_call'] : (palette.edge_kinds || ['message', 'route', 'feedback']).filter((k) => k !== 'tool_call');
    if (replacing) {
      const current = graph.edges.find((e) => e.id === replacing)?.data?.kind;
      if (current && !kinds.includes(current)) kinds.push(current);
    }
    return kinds.map((kind) => ({ kind, reason: graph.connectionError(value, kind, replacing) }));
  };
  const connectionKinds = connection ? kindsFor(connection) : [];
  const selectedEdge = graph.selectedEdge;

  const onConnect = (value: Connection) => {
    const options = kindsFor(value);
    const valid = options.filter((k) => !k.reason);
    if (!valid.length) { graph.setNotice(options[0]?.reason || '没有可用的连线关系。'); return; }
    if (valid.length === 1) {
      graph.connect(value, valid[0].kind);
      onPanelChange(null);
    } else {
      graph.select(null);
      onPanelChange(null);
      setConnection(value);
    }
  };

  return <div className="mas-workbench" ref={root} onKeyDown={(event) => {
    if (event.key === 'Escape') { event.stopPropagation(); closePanels(); }
    else if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable="true"]')) {
      event.stopPropagation();
    }
  }}>
    <GraphCanvas nodes={nodes} edges={graph.edges} active={active}
      onNodesChange={graph.onNodesChange} onEdgesChange={graph.onEdgesChange} onConnect={onConnect}
      onNodeClick={(_, node) => { graph.select({ kind: 'node', id: node.id }); onPanelChange(null); setConnection(null); revealNode(node.id); }}
      onEdgeClick={(_, edge) => { graph.select({ kind: 'edge', id: edge.id }); onPanelChange(null); setConnection(null); }}
      onPaneClick={() => { if (panel !== 'settings') closePanels(); else { graph.select(null); setConnection(null); } }}
      onAdd={add} onOpenLibrary={openLibrary} onError={graph.setNotice} />
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
    {panel === 'settings' ? <WorkflowSettings workflow={workflow} rlDirty={rlDirty} rlNotice={rlNotice} rlSettings={rlSettings} onClose={closePanels} />
      : panel === null && graph.selected ? <NodeInspector selected={graph.selected} palette={palette} entryId={workflow.entry_agent || 'hub'}
        onPatch={graph.updateSelected} onEntry={graph.setEntry} onDelete={graph.deleteSelected} onClose={closePanels} />
      : panel === null && selectedEdge ? <EdgeInspector edge={selectedEdge}
        kinds={kindsFor({ ...selectedEdge, sourceHandle: null, targetHandle: null }, selectedEdge.id)}
        onChange={(kind) => graph.updateEdge(selectedEdge.id, kind)} onDelete={graph.deleteSelected} onClose={closePanels} /> : null}
    {connection && panel === null && <div className="mas-connection-picker" role="dialog" aria-label="选择连线关系">
      <div className="mas-panel-heading"><h2>选择连线关系</h2><Button size="sm" variant="ghost" aria-label="取消连接" onClick={() => setConnection(null)}><X size={15} /></Button></div>
      <p><code>{connection.source}</code> → <code>{connection.target}</code></p>
      <div className="mas-connection-options">{connectionKinds.map(({ kind, reason }) =>
        <Button key={kind} size="sm" disabled={Boolean(reason)} title={reason || undefined}
          onClick={() => { graph.connect(connection, kind); setConnection(null); }}>{EDGE_LABELS[kind] || kind}</Button>)}</div>
    </div>}
  </div>;
}

export default memo(function MasGraphEditor(props: Props) {
  return <ReactFlowProvider><GraphWorkbench {...props} /></ReactFlowProvider>;
});

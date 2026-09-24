import { memo, useEffect, useRef, useState, type DragEvent } from 'react';
import {
  ReactFlow, Background, BackgroundVariant, ConnectionMode, MiniMap, Panel,
  useNodesInitialized, useReactFlow, useViewport, type ReactFlowProps, type XYPosition,
} from '@xyflow/react';
import { FlaskConical, GitBranch, Map, Maximize, Minus, Plus, Workflow } from 'lucide-react';
import type { CanvasMode, GraphNode, GraphEdge, GraphNodePreset } from '../types';
import { Button } from '../../../shared/ui/button';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import RouterNode from './RouterNode';
import { NODE_TRANSFER } from './GraphPalette';
import { WorkflowEdge } from './WorkflowEdge';
import { ConnectionPreview } from './ConnectionPreview';
import { CanvasToolButton, CanvasToolbar, CanvasToolbarDivider, CanvasToolbarGroup } from './CanvasToolbar';

const nodeTypes = { agent: AgentNode, tool: ToolNode, router: RouterNode };
const edgeTypes = { workflow: WorkflowEdge };
type Props = Pick<ReactFlowProps<GraphNode, GraphEdge>,
  'nodes' | 'edges' | 'onNodesChange' | 'onEdgesChange' | 'onConnect' | 'onNodeClick' | 'onEdgeClick' | 'onPaneClick'
  | 'isValidConnection' | 'onConnectStart' | 'onConnectEnd' | 'onReconnect' | 'onReconnectStart' | 'onReconnectEnd' | 'onMoveStart'> & {
  active: boolean;
  onAdd: (preset: GraphNodePreset, position: XYPosition) => void;
  onOpenLibrary: (templates?: boolean) => void;
  onError: (message: string) => void;
  mode: CanvasMode;
  onModeChange: (mode: CanvasMode) => void;
  onDebug: () => void;
  debugOpen: boolean;
};

export const GraphCanvas = memo(function GraphCanvas({
  active, mode, onModeChange, onDebug, debugOpen, onAdd, onOpenLibrary, onError, ...flowProps
}: Props) {
  const flow = useReactFlow<GraphNode, GraphEdge>();
  const { zoom } = useViewport();
  const initialized = useNodesInitialized();
  const fitted = useRef(false);
  const [minimap, setMinimap] = useState(false);

  useEffect(() => {
    if (!active || !initialized || fitted.current) return;
    const frame = requestAnimationFrame(() => {
      void flow.fitView({ padding: 0.25, maxZoom: 1 });
      fitted.current = true;
    });
    return () => cancelAnimationFrame(frame);
  }, [active, initialized, flow]);

  const onDrop = (event: DragEvent) => {
    if (mode === 'sampling') return;
    event.preventDefault();
    const raw = event.dataTransfer.getData(NODE_TRANSFER);
    if (!raw) return;
    let payload: unknown;
    try { payload = JSON.parse(raw); }
    catch { onError('无法读取拖入的节点，请从节点库重新添加。'); return; }
    if (!payload || typeof payload !== 'object' || !('nodeType' in payload) || !('id' in payload)
      || !['agent', 'tool', 'router'].includes(String(payload.nodeType)) || typeof payload.id !== 'string') {
      onError('不支持此节点数据，请从节点库添加。');
      return;
    }
    onAdd(payload as GraphNodePreset, flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  return <div className="mas-canvas" aria-label={mode === 'sampling' ? 'Sampling 编排画布' : 'Workflow 画布'} onDrop={onDrop}
    onDragOver={(event) => { if (mode === 'workflow') { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; } }}>
    <ReactFlow<GraphNode, GraphEdge> {...flowProps} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
      connectionMode={ConnectionMode.Loose} connectionLineComponent={ConnectionPreview} reconnectRadius={12}
      minZoom={0.2} maxZoom={2} deleteKeyCode={active && mode === 'workflow' ? ['Backspace', 'Delete'] : null}
      nodesDraggable nodesConnectable={mode === 'workflow'} edgesReconnectable={mode === 'workflow'}
      onlyRenderVisibleElements
      ariaLabelConfig={{ 'minimap.ariaLabel': '画布小地图' }}>
      <Background id="mas-dots" variant={BackgroundVariant.Dots} gap={20} size={1.5}
        color="#b7c1cf" bgColor="var(--canvas)" />
      <Panel position="bottom-left" className="mas-canvas-controls">
        <CanvasToolbar className="canvas-toolbar--viewport">
          <CanvasToolbarGroup>
            <CanvasToolButton label="缩小画布" onClick={() => void flow.zoomOut()}><Minus size={18} /></CanvasToolButton>
            <span className="mas-zoom-value">{Math.round(zoom * 100)}%</span>
            <CanvasToolButton label="放大画布" onClick={() => void flow.zoomIn()}><Plus size={18} /></CanvasToolButton>
          </CanvasToolbarGroup>
          <CanvasToolbarDivider />
          <CanvasToolbarGroup>
            <CanvasToolButton label="适应画布" onClick={() => void flow.fitView({ padding: 0.25, maxZoom: 1 })}><Maximize size={18} /></CanvasToolButton>
            <CanvasToolButton label={minimap ? '关闭小地图' : '打开小地图'} aria-pressed={minimap}
              onClick={() => setMinimap(!minimap)}><Map size={18} /></CanvasToolButton>
          </CanvasToolbarGroup>
        </CanvasToolbar>
      </Panel>
      <Panel position="bottom-center" className="mas-canvas-primary-toolbar">
        <CanvasToolbar>
          <CanvasToolbarGroup>
            <CanvasToolButton label={mode === 'workflow' ? '添加节点' : '切换到 Workflow 后添加节点'}
              disabled={mode !== 'workflow'} onClick={() => onOpenLibrary()}><Plus size={18} /></CanvasToolButton>
          </CanvasToolbarGroup>
          <CanvasToolbarDivider />
          <CanvasToolbarGroup>
            <CanvasToolButton label="Workflow 编排" aria-pressed={mode === 'workflow'}
              onClick={() => onModeChange('workflow')}><Workflow size={18} /></CanvasToolButton>
            <CanvasToolButton label="Sampling 编排" aria-pressed={mode === 'sampling'}
              onClick={() => onModeChange('sampling')}><GitBranch size={18} /></CanvasToolButton>
          </CanvasToolbarGroup>
          <CanvasToolbarDivider />
          <CanvasToolbarGroup>
            <CanvasToolButton label={mode === 'workflow' ? '单题调试' : '切换到 Workflow 后进行单题调试'}
              disabled={mode !== 'workflow'} aria-pressed={debugOpen}
              onClick={onDebug}><FlaskConical size={18} /></CanvasToolButton>
          </CanvasToolbarGroup>
        </CanvasToolbar>
      </Panel>
      {minimap && <MiniMap position="bottom-left" className="mas-minimap" pannable zoomable />}
      {!flowProps.nodes?.length && <Panel position="top-center" className="mas-canvas-empty">
        <span className="mas-empty-icon"><Workflow size={26} /></span>
        <h2>从一个 Agent 开始</h2><p>添加节点、连接能力，搭建你的工作流。</p>
        <div><Button size="sm" variant="primary" onClick={() => onOpenLibrary()}><Plus size={14} />添加 Agent</Button>
          <Button size="sm" onClick={() => onOpenLibrary(true)}>从模板开始</Button></div>
      </Panel>}
    </ReactFlow>
  </div>;
});

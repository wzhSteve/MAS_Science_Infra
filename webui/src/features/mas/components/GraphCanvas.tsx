import { memo, useEffect, useRef, useState, type DragEvent } from 'react';
import {
  ReactFlow, Background, BackgroundVariant, MiniMap, Panel,
  useNodesInitialized, useReactFlow, useViewport, type ReactFlowProps, type XYPosition,
} from '@xyflow/react';
import { Maximize, Minus, Plus, Map, Workflow } from 'lucide-react';
import type { GraphNode, GraphEdge } from '../types';
import { Button } from '../../../shared/ui/button';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import { NODE_TRANSFER } from './GraphPalette';

const nodeTypes = { agent: AgentNode, tool: ToolNode };
type Props = Pick<ReactFlowProps<GraphNode, GraphEdge>,
  'nodes' | 'edges' | 'onNodesChange' | 'onEdgesChange' | 'onConnect' | 'onNodeClick' | 'onEdgeClick' | 'onPaneClick'> & {
  active: boolean;
  onAdd: (kind: 'agent' | 'tool', type: string, position: XYPosition) => void;
  onOpenLibrary: (templates?: boolean) => void;
  onError: (message: string) => void;
};

export const GraphCanvas = memo(function GraphCanvas({ active, onAdd, onOpenLibrary, onError, ...flowProps }: Props) {
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
    event.preventDefault();
    const raw = event.dataTransfer.getData(NODE_TRANSFER);
    if (!raw) return;
    let payload: unknown;
    try { payload = JSON.parse(raw); }
    catch { onError('无法读取拖入的节点，请从节点库重新添加。'); return; }
    if (!payload || typeof payload !== 'object' || !('kind' in payload) || !('type' in payload)
      || (payload.kind !== 'agent' && payload.kind !== 'tool') || typeof payload.type !== 'string') {
      onError('不支持此节点数据，请从节点库添加。');
      return;
    }
    onAdd(payload.kind, payload.type, flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  return <div className="mas-canvas" aria-label="Workflow 画布" onDrop={onDrop}
    onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; }}>
    <ReactFlow<GraphNode, GraphEdge> {...flowProps} nodeTypes={nodeTypes}
      minZoom={0.2} maxZoom={2} deleteKeyCode={active ? ['Backspace', 'Delete'] : null} onlyRenderVisibleElements
      ariaLabelConfig={{ 'minimap.ariaLabel': '画布小地图' }}>
      <Background id="mas-dots" variant={BackgroundVariant.Dots} gap={20} size={1.5}
        color="#b7c1cf" bgColor="var(--canvas)" />
      <Panel position="bottom-left" className="mas-canvas-controls">
        <Button size="sm" variant="ghost" aria-label="缩小画布" title="缩小" onClick={() => void flow.zoomOut()}><Minus size={15} /></Button>
        <span className="mas-zoom-value">{Math.round(zoom * 100)}%</span>
        <Button size="sm" variant="ghost" aria-label="放大画布" title="放大" onClick={() => void flow.zoomIn()}><Plus size={15} /></Button>
        <span className="mas-control-divider" />
        <Button size="sm" variant="ghost" aria-label="适应画布" title="适应画布" onClick={() => void flow.fitView({ padding: 0.25, maxZoom: 1 })}><Maximize size={15} /></Button>
        <Button size="sm" variant="ghost" aria-label="画布小地图" title="小地图" aria-pressed={minimap} onClick={() => setMinimap(!minimap)}><Map size={15} /></Button>
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

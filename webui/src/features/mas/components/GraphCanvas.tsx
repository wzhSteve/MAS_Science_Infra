import { memo, useCallback, useRef, type DragEvent } from 'react';
import {
  ReactFlow, Background, Controls, MiniMap,
  type Node, type ReactFlowInstance, type ReactFlowProps, type XYPosition,
} from '@xyflow/react';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import { NODE_TRANSFER, PIN_TRANSFER } from './GraphPalette';

const nodeTypes = { agent: AgentNode, tool: ToolNode };

type Props = Pick<ReactFlowProps, 'nodes' | 'edges' | 'onNodesChange' | 'onEdgesChange' | 'onConnect'
  | 'onNodeClick' | 'onPaneClick' | 'onNodeDragStop'> & {
    selectedId?: string;
    onAdd: (kind: 'agent' | 'tool', id: string, role: string, position?: XYPosition) => void;
    onEntry: (id: string) => void;
  };

export const GraphCanvas = memo(function GraphCanvas({ onAdd, onEntry, selectedId, ...flowProps }: Props) {
  const instance = useRef<ReactFlowInstance | null>(null);
  const hoverId = useRef<string | null>(null);
  const current = useRef({ nodes: flowProps.nodes, selectedId, onAdd, onEntry });
  current.current = { nodes: flowProps.nodes, selectedId, onAdd, onEntry };
  const onInit = useCallback((value: ReactFlowInstance) => { instance.current = value; }, []);
  const onNodeMouseEnter = useCallback((_: unknown, node: Node) => { hoverId.current = node.id; }, []);
  const onNodeMouseLeave = useCallback(() => { hoverId.current = null; }, []);
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);
  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault();
    const { nodes, selectedId: selected, onAdd: add, onEntry: entry } = current.current;
    if (event.dataTransfer.getData(PIN_TRANSFER)) {
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('.react-flow__node');
      const ids = [target?.getAttribute('data-id'), hoverId.current, selected];
      const hit = ids.map((id) => nodes?.find((node) => node.id === id && node.type !== 'tool')).find(Boolean);
      if (hit) entry(hit.id);
      return;
    }
    const raw = event.dataTransfer.getData(NODE_TRANSFER);
    if (!raw) return;
    try {
      const payload: unknown = JSON.parse(raw);
      if (!payload || typeof payload !== 'object') return;
      const value = payload as Record<string, unknown>;
      if ((value.kind !== 'agent' && value.kind !== 'tool') || typeof value.id !== 'string' || typeof value.role !== 'string') return;
      const position = instance.current?.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      add(value.kind, value.id, value.role, position);
    } catch {
      // A foreign drag payload must not interrupt the canvas.
    }
  }, []);

  return <div className="mas-canvas" aria-label="Workflow 画布" onDragOver={onDragOver} onDrop={onDrop}>
    <ReactFlow {...flowProps} nodeTypes={nodeTypes} onInit={onInit}
      onNodeMouseEnter={onNodeMouseEnter} onNodeMouseLeave={onNodeMouseLeave} fitView>
      <Background />
      <Controls />
      <MiniMap />
    </ReactFlow>
  </div>;
});

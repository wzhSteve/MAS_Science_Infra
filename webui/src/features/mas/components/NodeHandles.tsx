import { memo } from 'react';
import { Handle, Position, useConnection } from '@xyflow/react';
import { useGraphInteraction } from './GraphInteractionContext';

const handles = [
  { id: 'top', position: Position.Top, label: '上方' },
  { id: 'right', position: Position.Right, label: '右侧' },
  { id: 'bottom', position: Position.Bottom, label: '下方' },
  { id: 'left', position: Position.Left, label: '左侧' },
] as const;

export const NodeHandles = memo(function NodeHandles({ id }: { id: string }) {
  const { rules, reconnecting } = useGraphInteraction();
  const from = useConnection((connection) => connection.fromNode?.id ?? null);
  const candidate = from && from !== id && !reconnecting;
  const allowed = candidate && rules.options({ source: from, target: id, sourceHandle: null, targetHandle: null }).some((option) => !option.reason);
  const state = candidate ? allowed ? ' is-available' : ' is-unavailable' : '';
  return <>{handles.map(({ id: handleId, position, label }) =>
    <Handle key={handleId} id={handleId} type="source" position={position}
      className={`mas-handle${state}`} aria-label={`${id}，${label}连接点`} title={`${label}连接点`}>
      <span className="mas-handle-dot" />
    </Handle>)}</>;
});

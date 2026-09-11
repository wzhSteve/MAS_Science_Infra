import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

function ToolNode({ data, selected }: NodeProps) {
  const d = data as { label?: string };
  const cls = ['mas-node', 'mas-node--tool', selected ? 'is-selected' : ''].filter(Boolean).join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div className="mono font-semibold">{d.label || 'tool'}</div>
      <div className="mas-node__role">tool</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(ToolNode);

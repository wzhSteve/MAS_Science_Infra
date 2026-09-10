import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

function ToolNode({ data, selected }: NodeProps) {
  const d = data as { label?: string };
  const cls = ['node-tool', selected ? 'selected' : ''].filter(Boolean).join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div style={{ fontWeight: 600 }}>{d.label || 'tool'}</div>
      <div className="kind">tool</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(ToolNode);

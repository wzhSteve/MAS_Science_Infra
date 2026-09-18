import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

function ToolNode({ data, selected }: NodeProps) {
  const d = data as { label?: string; branchEnabled?: boolean; branchGate?: string };
  const cls = ['node-tool', selected ? 'selected' : '', d.branchEnabled ? 'branch-on' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div style={{ fontWeight: 600 }}>{d.label || 'tool'}</div>
      <div className="kind">tool</div>
      {d.branchEnabled ? (
        <div className="chip ok" style={{ marginTop: 4, fontSize: 11 }}>
          branch:{d.branchGate || 'on'}
        </div>
      ) : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(ToolNode);

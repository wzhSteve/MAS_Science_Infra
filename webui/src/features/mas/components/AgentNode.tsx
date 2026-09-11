import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '../types';

function AgentNode({ data, selected }: NodeProps) {
  const d = data as GraphNodeData;
  const cls = ['mas-node', 'mas-node--agent', selected ? 'is-selected' : '', d.entry ? 'is-entry' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div className="flex items-start justify-between gap-2">
        <div className="mono font-semibold">{d.label || 'agent'}</div>
        {d.entry ? <span className="mas-node__entry">入口</span> : null}
      </div>
      <div className="mas-node__role">{d.role}</div>
      {d.tools && d.tools.length ? <div className="mas-node__meta">{d.tools.join(', ')}</div> : null}
      {d.verify ? <div className="mas-node__meta">verify → {d.verify}</div> : null}
      <div className="mas-node__meta">{d.trainable === false ? '不训练' : '可训练'}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(AgentNode);

import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Bot, Flag } from 'lucide-react';
import type { GraphNode } from '../types';

function AgentNode({ data: d, selected }: NodeProps<GraphNode>) {
  const cls = ['mas-node', 'mas-node--agent', selected ? 'is-selected' : '', d.entry ? 'is-entry' : '', d.issue ? 'has-issue' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Left} />
      <div className="mas-node__heading">
        <span className="mas-node__icon"><Bot size={18} /></span>
        <div><div className="mas-node__name">{d.label || 'Agent'}</div><div className="mas-node__role">{d.role}</div></div>
      </div>
      <div className="mas-node__footer">
        {d.entry ? <span className="mas-node__entry"><Flag size={10} />入口</span> : <span>Agent</span>}
        <span>{d.tools?.length ? `${d.tools.length} 个工具` : d.trainable === false ? '不参与训练' : '参与训练'}</span>
      </div>
      {d.issue && <div className="mas-node__issue">{d.issue}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export default memo(AgentNode);

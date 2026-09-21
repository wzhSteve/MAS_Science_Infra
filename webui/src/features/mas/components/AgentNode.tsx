import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { Bot, Flag } from 'lucide-react';
import type { GraphNode } from '../types';
import { NodeHandles } from './NodeHandles';

function AgentNode({ id, data: d, selected }: NodeProps<GraphNode>) {
  const cls = ['mas-node', 'mas-node--agent', selected ? 'is-selected' : '', d.entry ? 'is-entry' : '', d.issue ? 'has-issue' : '', d.related ? 'is-related' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <NodeHandles id={id} />
      <div className="mas-node__heading">
        <span className="mas-node__icon"><Bot size={18} /></span>
        <div><div className="mas-node__name">{d.label || 'Agent'}</div><div className="mas-node__role">{d.role}</div></div>
      </div>
      <div className="mas-node__footer">
        {d.entry ? <span className="mas-node__entry"><Flag size={10} />入口</span> : <span>Agent</span>}
        <span>{d.tools?.length ? `${d.tools.length} 个工具` : d.trainable === false ? '不参与训练' : '参与训练'}</span>
      </div>
      {d.issue && <div className="mas-node__issue">{d.issue}</div>}
    </div>
  );
}

export default memo(AgentNode);

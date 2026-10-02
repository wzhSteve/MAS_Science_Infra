import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { Boxes } from 'lucide-react';
import type { GraphNode } from '../types';
import { NodeHandles } from './NodeHandles';

function PoolNode({ id, data, selected }: NodeProps<GraphNode>) {
  const members = data.members || [];
  const cls = [
    'mas-node', 'mas-node--pool', selected ? 'is-selected' : '',
    data.issue ? 'has-issue' : '', data.related ? 'is-related' : '',
  ].filter(Boolean).join(' ');
  return <div className={cls}>
    <NodeHandles id={id} />
    <div className="mas-node__heading">
      <span className="mas-node__icon"><Boxes size={18} /></span>
      <div><div className="mas-node__name">{data.label || 'tool-agent pool'}</div>
        <div className="mas-node__role">Router 下游</div></div>
    </div>
    <ul className="mas-pool-members">
      {members.length ? members.map((member) => <li key={member}>{(data.memberLabels || {})[member] || member}</li>) : <li className="is-empty">点击添加 tool-agent</li>}
    </ul>
    <div className="mas-node__footer"><span>Pool</span><span>{members.length} 个成员</span></div>
    {data.issue && <div className="mas-node__issue">{data.issue}</div>}
  </div>;
}

export default memo(PoolNode);

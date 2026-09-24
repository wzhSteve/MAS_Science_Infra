import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { GitBranch, Route } from 'lucide-react';
import type { GraphNode } from '../types';
import { NodeHandles } from './NodeHandles';
import { SamplingNodeStatus } from './SamplingNodeStatus';

function RouterNode({ id, data, selected }: NodeProps<GraphNode>) {
  const cls = [
    'mas-node', 'mas-node--router', selected ? 'is-selected' : '',
    data.issue ? 'has-issue' : '', data.related ? 'is-related' : '', data.canvasMode === 'sampling' ? 'is-sampling' : '',
  ].filter(Boolean).join(' ');
  return <div className={cls}>
    <NodeHandles id={id} />
    <div className="mas-node__heading">
      <span className="mas-node__icon"><Route size={18} /></span>
      <div><div className="mas-node__name">{data.label || id}</div>
        <div className="mas-node__role">{data.strategy || 'llm_choice'}</div></div>
    </div>
    <div className="mas-node__footer">
      <span>Router</span><span>{data.branchCount ? <span className="mas-node__branch"><GitBranch size={10} />{data.branchCount}</span>
        : `${data.candidates?.length || 0} 个候选`}</span>
    </div>
    {data.canvasMode === 'sampling' && <SamplingNodeStatus state={data.samplingState} label={data.samplingLabel} />}
    {data.issue && <div className="mas-node__issue">{data.issue}</div>}
  </div>;
}

export default memo(RouterNode);

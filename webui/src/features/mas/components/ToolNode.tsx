import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { GitBranch, Wrench } from 'lucide-react';
import type { GraphNode } from '../types';
import { NodeHandles } from './NodeHandles';

function ToolNode({ id, data: d, selected }: NodeProps<GraphNode>) {
  const cls = ['mas-node', 'mas-node--tool', selected ? 'is-selected' : '', d.issue ? 'has-issue' : '', d.related ? 'is-related' : '', d.canvasMode === 'sampling' ? 'is-sampling-context' : ''].filter(Boolean).join(' ');
  return (
    <div className={cls}>
      <NodeHandles id={id} />
      <div className="mas-node__heading"><span className="mas-node__icon"><Wrench size={17} /></span>
        <div><div className="mas-node__name">{d.label || 'Tool'}</div><div className="mas-node__role">内置工具 · 按需调用</div></div>
      </div>
      {d.canvasMode !== 'sampling' && d.branchCount ? <div className="mas-node__footer"><span>Branch Site</span>
        <span className="mas-node__branch"><GitBranch size={10} />{d.branchCount}</span></div> : null}
      {d.issue && <div className="mas-node__issue">{d.issue}</div>}
    </div>
  );
}

export default memo(ToolNode);

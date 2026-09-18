import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

/** agent-framework A5: diamond routing node (RouterSpec, schema 0.3). */
function RouterNode({ data, selected }: NodeProps) {
  const d = data as {
    label?: string;
    strategy?: string;
    candidates?: string[];
    n_candidates?: number;
    branchEnabled?: boolean;
    branchGate?: string;
  };
  const cls = ['node-router', selected ? 'selected' : '', d.branchEnabled ? 'branch-on' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div style={{ fontWeight: 600 }}>{d.label || 'router'}</div>
      <div className="kind">router</div>
      <div className="chip" style={{ marginTop: 4, fontSize: 11 }}>
        {d.strategy || 'llm_choice'} · {d.n_candidates ?? (d.candidates || []).length} cands
      </div>
      {d.branchEnabled ? (
        <div className="chip ok" style={{ marginTop: 4, fontSize: 11 }}>
          branch:{d.branchGate || 'on'}
        </div>
      ) : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(RouterNode);

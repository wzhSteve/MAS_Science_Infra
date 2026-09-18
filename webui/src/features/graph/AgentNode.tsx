import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

function AgentNode({ data, selected }: NodeProps) {
  const d = data as {
    label?: string;
    kind?: string;
    role?: string;
    skills?: string[];
    tools?: string[];
    verify?: string;
    entry?: boolean;
    trainable?: boolean;
    branchEnabled?: boolean;
    branchGate?: string;
  };
  const cls = ['node-agent', selected ? 'selected' : '', d.entry ? 'entry' : '', d.branchEnabled ? 'branch-on' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <Handle type="target" position={Position.Top} />
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ fontWeight: 650 }}>{d.label || 'agent'}</div>
        <span style={{ display: 'flex', gap: 4 }}>
          {d.kind && d.kind !== 'blank' ? (
            <span className={d.kind === 'tool' ? 'chip warn' : 'chip'} style={{ fontSize: 10 }}>
              {d.kind}
            </span>
          ) : null}
          {d.entry ? <span className="badge">入口</span> : null}
        </span>
      </div>
      <div className="role">{d.role}</div>
      {d.tools && d.tools.length ? <div className="meta">{d.tools.join(', ')}</div> : null}
      {d.verify ? <div className="meta">verify→{d.verify}</div> : null}
      {d.trainable === false ? <div className="meta">不训练</div> : null}
      {d.branchEnabled ? (
        <div className="chip ok" style={{ marginTop: 4, fontSize: 11 }}>
          branch:{d.branchGate || 'on'}
        </div>
      ) : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(AgentNode);

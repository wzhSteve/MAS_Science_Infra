import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';
import { Bot, Flag, GitBranch, Wrench } from 'lucide-react';
import type { GraphNode } from '../types';
import { NodeHandles } from './NodeHandles';

function AgentNode({ id, data: d, selected }: NodeProps<GraphNode>) {
  const toolAgent = d.kind === 'tool';
  const tools = d.tools?.length || 0;
  const modelName = d.effective_model_name || d.model_name || d.model || '未配置模型';
  const modelInvalid = d.model_available === false || (d.trainable !== false && d.model_trainable === false);
  const cls = ['mas-node', 'mas-node--agent', toolAgent ? 'mas-node--tool-agent' : '', selected ? 'is-selected' : '', d.entry ? 'is-entry' : '', d.issue ? 'has-issue' : '', d.related ? 'is-related' : '', d.canvasMode === 'sampling' ? 'is-sampling-context' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={cls}>
      <NodeHandles id={id} />
      <div className="mas-node__heading">
        <span className="mas-node__icon">{toolAgent ? <Wrench size={18} /> : <Bot size={18} />}</span>
        <div><div className="mas-node__name">{d.label || 'Agent'}</div>
          <div className="mas-node__role">{toolAgent ? '智能工具' : d.kind === 'blank' ? '自定义 Agent' : `${d.kind || 'Agent'} · ${d.role}`}</div></div>
      </div>
      <div className={`mas-node__model${modelInvalid ? ' is-invalid' : ''}`} title={modelName}>
        <span>{modelName}</span>
        <small>{d.model_inherited ? '继承' : d.model_source === 'api' ? 'API' : d.model_source === 'local' ? '本地' : '模型'}</small>
      </div>
      <div className="mas-node__footer">
        {d.entry ? <span className="mas-node__entry"><Flag size={10} />入口</span> : <span>Agent</span>}
        <span className="mas-node__tools"><Wrench size={10} />{tools}</span>
        <span>{d.canvasMode === 'sampling' && d.branchCount
          ? <span className="mas-node__branch"><GitBranch size={10} />{d.branchCount}</span>
          : <span className={`mas-node__training${d.trainable === false ? '' : ' is-trainable'}${modelInvalid ? ' is-invalid' : ''}`}>
              {d.trainable === false ? '已冻结' : modelInvalid ? '模型不可训练' : '参与训练'}
            </span>}</span>
      </div>
      {d.issue && <div className="mas-node__issue">{d.issue}</div>}
    </div>
  );
}

export default memo(AgentNode);

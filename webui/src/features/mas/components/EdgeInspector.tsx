import { useState } from 'react';
import { ArrowDown, ArrowDownUp, Bot, Check, Search, Trash2, Wrench, X } from 'lucide-react';
import type { EdgePatch, GraphEdge, GraphNode } from '../types';
import type { EdgeRules } from '../model/edgeRules';
import { edgeConnection, reverseConnection } from '../model/edgeRules';
import { edgeDefinition } from '../model/edgeDefinitions';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { RelationTypePicker } from './RelationTypePicker';

type Side = 'source' | 'target';

export function EdgeInspector({ edge, nodes, rules, topology, onChange, onClose, onDelete }: {
  edge: GraphEdge;
  nodes: GraphNode[];
  rules: EdgeRules;
  topology: string;
  onChange: (patch: EdgePatch) => boolean;
  onClose: () => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState<Side | null>(null);
  const [query, setQuery] = useState('');
  const kind = edge.data?.kind || 'message';
  const definition = edgeDefinition(kind);
  const Icon = definition.icon;
  const tool = kind === 'tool_call';
  const value = edgeConnection(edge);
  const issue = rules.error(value, kind, edge.id);
  const reverse = reverseConnection(value);
  const reverseError = tool ? '' : rules.error(reverse, kind, edge.id);
  const source = rules.nodeById.get(edge.source);
  const hubSubset = nodes.filter((node) => node.type === 'agent').every((node) => node.id === 'hub' || node.id === 'verifier');

  const entityCard = (side: Side) => {
    const id = edge[side];
    const node = rules.nodeById.get(id);
    const EntityIcon = node?.type === 'tool' ? Wrench : Bot;
    const label = tool ? side === 'source' ? '使用工具的 Agent' : '提供的 Tool' : side === 'source' ? '发送方' : '接收方';
    return <div className={`mas-entity-card${editing === side ? ' is-editing' : ''}`}>
      <span className="mas-entity-icon"><EntityIcon size={18} aria-hidden="true" /></span>
      <div className="mas-entity-copy"><span>{label}</span><strong>{node?.data.label || id}</strong>
        <small>{node?.data.role || (node ? node.type : '实体不存在')}</small></div>
      <Button size="sm" variant="ghost" aria-label={`更换${label}`} aria-expanded={editing === side}
        onClick={() => { setEditing(editing === side ? null : side); setQuery(''); }}>更换</Button>
    </div>;
  };

  const candidates = editing ? nodes.filter((node) => {
    const expected = tool && editing === 'target' ? 'tool' : 'agent';
    return node.type === expected && `${node.data.label || ''} ${node.id} ${node.data.role || ''}`.toLowerCase().includes(query.trim().toLowerCase());
  }) : [];

  return <aside className="mas-panel mas-inspector mas-edge-inspector" aria-label="连线属性">
    <div className="mas-panel-heading">
      <div><span className="mas-panel-caption">{tool ? '工具调用关系' : 'Agent 协作'}</span>
        <h2><Icon size={17} style={{ color: definition.color }} />{definition.title}</h2></div>
      <Button size="sm" variant="ghost" aria-label="关闭连线属性" onClick={onClose}><X size={16} /></Button>
    </div>
    <div className="mas-panel-body">
      {issue && <InlineNotice tone="danger">{issue}</InlineNotice>}
      <section className="mas-edge-section">
        <h3>{tool ? '工具提供给谁' : '关系双方'}</h3>
        {entityCard('source')}
        <div className="mas-entity-direction"><ArrowDown size={17} aria-hidden="true" />
          {tool ? <span>提供能力 · 按需使用</span> : <Button size="sm" variant="ghost" disabled={Boolean(reverseError)}
            onClick={() => onChange(reverse)}><ArrowDownUp size={13} />反转方向</Button>}</div>
        {entityCard('target')}
        {!tool && reverseError && <p className="field-hint">无法反转：{reverseError}</p>}
        {editing && <div className="mas-entity-picker" onKeyDown={(event) => {
          if (event.key === 'Escape') { event.stopPropagation(); setEditing(null); }
        }}>
          <div className="mas-entity-search"><Search size={14} /><Input autoFocus value={query} placeholder="搜索关联实体"
            aria-label="搜索关联实体" onChange={(e) => setQuery(e.target.value)} />
            <Button size="sm" variant="ghost" aria-label="关闭实体选择" onClick={() => setEditing(null)}><X size={14} /></Button></div>
          <div className="mas-entity-candidates">
            {candidates.map((node) => {
              const current = node.id === edge[editing];
              const reason = current ? '' : rules.error({ ...value, [editing]: node.id }, kind, edge.id);
              return <button type="button" key={node.id} disabled={Boolean(reason)} onClick={() => {
                if (current || onChange({ [editing]: node.id })) setEditing(null);
              }}>
                <span><strong>{node.data.label || node.id}</strong><small>{reason || node.data.role || node.type}</small></span>
                {current && <Check size={14} />}
              </button>;
            })}
            {!candidates.length && <p className="mas-panel-empty">没有匹配的实体</p>}
          </div>
        </div>}
      </section>
      {tool ? <section className="mas-edge-section mas-tool-binding-note">
        <span className="mas-binding-symbol"><Wrench size={18} /></span>
        <div><h3>由 Agent 自行决定调用</h3><p>这条线只提供工具能力，不保证每次运行都调用，也不是固定执行步骤。</p></div>
      </section> : <section className="mas-edge-section">
        <h3>协作方式</h3>
        <RelationTypePicker value={kind} options={rules.options(value, edge.id)} onChange={(next) => onChange({ kind: next })} />
      </section>}
      <section className="mas-edge-section mas-edge-explanation">
        <h3>关系说明</h3><p>{definition.description}</p>
        {tool && edge.source === 'hub' && <p>兼容行为：hub 没有显式绑定时会使用全局工具。移除最后一条绑定不等于禁止 hub 使用工具。</p>}
        {!tool && (topology !== 'graph' || hubSubset) && <p>当前拓扑使用 hub 兼容执行路径，不保证按通用图逐条调度协作关系。</p>}
        {!tool && ['verifier', 'critic'].includes(source?.data.role || '') && kind !== 'feedback'
          && <p>验证角色使用专用反馈路径，这条关系不代表无条件继续执行下游。</p>}
      </section>
    </div>
    <div className="mas-panel-footer mas-edge-footer">
      <p>修改随 Workflow 一起保存</p>
      <Button size="sm" variant="ghost" className="mas-delete-edge" onClick={onDelete}><Trash2 size={14} />
        {tool ? '移除工具绑定' : '删除连线'}</Button>
      {tool && <small>保留工具节点及其他 Agent 的绑定。</small>}
    </div>
  </aside>;
}

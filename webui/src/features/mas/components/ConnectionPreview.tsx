import { getBezierPath, type ConnectionLineComponentProps } from '@xyflow/react';
import type { GraphNode } from '../types';
import { useGraphInteraction } from './GraphInteractionContext';
import { edgeConnection } from '../model/edgeRules';

export function ConnectionPreview({ fromNode, fromHandle, toNode, toHandle, fromX, fromY, toX, toY, fromPosition, toPosition, connectionStatus }: ConnectionLineComponentProps<GraphNode>) {
  const { rules, reconnecting } = useGraphInteraction();
  const [path] = getBezierPath({ sourceX: fromX, sourceY: fromY, targetX: toX, targetY: toY, sourcePosition: fromPosition, targetPosition: toPosition });
  let hint = '拖到另一个实体的连接点';
  let error = '';
  if (toNode) {
    const candidate = { source: fromNode.id, target: toNode.id, sourceHandle: fromHandle.id ?? null, targetHandle: toHandle?.id ?? null };
    if (reconnecting) {
      const current = edgeConnection(reconnecting);
      const value = fromNode.id === reconnecting.target
        ? { ...current, source: toNode.id, sourceHandle: toHandle?.id ?? null }
        : { ...current, target: toNode.id, targetHandle: toHandle?.id ?? null };
      const next = rules.normalizeEdit(value, reconnecting);
      const geometryOnly = next.source === reconnecting.source && next.target === reconnecting.target;
      error = geometryOnly ? '' : rules.error(next, reconnecting.data?.kind || 'message', reconnecting.id);
      hint = geometryOnly ? '只调整出线位置，不改变业务关系' : '松开以更换关联实体';
    } else {
      const options = rules.options(candidate);
      error = options.some((option) => !option.reason) ? '' : options[0]?.reason || '没有可用的关系';
      hint = fromNode.type === 'tool' || toNode.type === 'tool' ? '提供工具能力 · Agent 按需调用' : '松开后选择协作方式';
    }
  }
  const invalid = connectionStatus === 'invalid' || Boolean(error);
  return <g className={`mas-connection-preview${invalid ? ' is-invalid' : ''}`}>
    <path d={path} fill="none" strokeWidth={1.5} strokeDasharray="5 4" />
    <circle cx={toX} cy={toY} r={4} />
    <foreignObject x={toX + 14} y={toY + 12} width={245} height={80} pointerEvents="none">
      <div className="mas-connection-hint">{error || hint}</div>
    </foreignObject>
  </g>;
}

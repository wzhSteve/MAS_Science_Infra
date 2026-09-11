import { memo, useId, type CSSProperties } from 'react';
import { BaseEdge, EdgeLabelRenderer, Position, getBezierPath, type EdgeProps } from '@xyflow/react';
import { CircleAlert } from 'lucide-react';
import type { GraphEdge } from '../types';
import { edgeDefinition } from '../model/edgeDefinitions';
import { useGraphInteraction } from './GraphInteractionContext';

const vectors: Record<Position, [number, number]> = {
  [Position.Top]: [0, -1], [Position.Right]: [1, 0], [Position.Bottom]: [0, 1], [Position.Left]: [-1, 0],
};

export const WorkflowEdge = memo(function WorkflowEdge(props: EdgeProps<GraphEdge>) {
  const { id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, data } = props;
  const { onSelectEdge } = useGraphInteraction();
  const markerId = `mas-arrow-${useId().replace(/:/g, '')}`;
  const definition = edgeDefinition(data?.kind || 'message');
  const Icon = data?.issue ? CircleAlert : definition.icon;
  let [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const lane = data?.lane || 0;
  if (lane) {
    const dx = targetX - sourceX;
    const dy = targetY - sourceY;
    const length = Math.hypot(dx, dy) || 1;
    const direction = source < target ? 1 : -1;
    const ox = -dy / length * lane * direction;
    const oy = dx / length * lane * direction;
    const distance = Math.max(60, length / 3);
    const [sx, sy] = vectors[sourcePosition];
    const [tx, ty] = vectors[targetPosition];
    const c1 = { x: sourceX + sx * distance + ox, y: sourceY + sy * distance + oy };
    const c2 = { x: targetX + tx * distance + ox, y: targetY + ty * distance + oy };
    path = `M ${sourceX},${sourceY} C ${c1.x},${c1.y} ${c2.x},${c2.y} ${targetX},${targetY}`;
    labelX = (sourceX + 3 * c1.x + 3 * c2.x + targetX) / 8;
    labelY = (sourceY + 3 * c1.y + 3 * c2.y + targetY) / 8;
  }
  const color = data?.issue ? 'var(--danger)' : definition.color;
  const style = { '--edge-color': color } as CSSProperties;
  return <>
    <g className={`mas-flow-edge${selected ? ' is-selected' : ''}${data?.issue ? ' has-issue' : ''}`} style={style}>
      {definition.directional && <defs><marker id={markerId} viewBox="0 0 10 10" markerWidth={7} markerHeight={7}
        refX={9} refY={5} orient="auto-start-reverse" markerUnits="userSpaceOnUse">
        <path d="M 1 1 L 9 5 L 1 9 Z" fill={color} />
      </marker></defs>}
      <path d={path} className="mas-edge-halo" fill="none" pointerEvents="none" />
      <BaseEdge id={id} path={path} interactionWidth={20}
        markerEnd={definition.directional ? `url(#${markerId})` : undefined}
        style={{ stroke: color, strokeWidth: definition.width, strokeDasharray: definition.dash }} />
    </g>
    <EdgeLabelRenderer>
      <button type="button" className={`mas-edge-label nodrag nopan${selected ? ' is-selected' : ''}${data?.issue ? ' has-issue' : ''}`}
        style={{ ...style, transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
        onClick={() => onSelectEdge(id)} aria-label={`${source} 到 ${target}，${definition.title}，编辑连线`}
        title={data?.issue || definition.description}>
        <Icon size={12} aria-hidden="true" /><span>{definition.label}</span>
      </button>
    </EdgeLabelRenderer>
  </>;
});

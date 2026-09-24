import { memo, useId, type CSSProperties } from 'react';
import { BaseEdge, Position, getBezierPath, type EdgeProps } from '@xyflow/react';
import type { GraphEdge } from '../types';
import { edgeDefinition } from '../model/edgeDefinitions';

const vectors: Record<Position, [number, number]> = {
  [Position.Top]: [0, -1], [Position.Right]: [1, 0], [Position.Bottom]: [0, 1], [Position.Left]: [-1, 0],
};

export const WorkflowEdge = memo(function WorkflowEdge(props: EdgeProps<GraphEdge>) {
  const { id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, data } = props;
  const markerId = `mas-arrow-${useId().replace(/:/g, '')}`;
  const definition = edgeDefinition(data?.kind || 'message');
  let [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
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
  }
  const color = data?.issue ? 'var(--danger)' : definition.color;
  const style = { '--edge-color': color, '--edge-width': `${definition.width}px` } as CSSProperties;
  return <g className={`mas-flow-edge${selected ? ' is-selected' : ''}${data?.issue ? ' has-issue' : ''}${data?.canvasMode === 'sampling' ? ` is-sampling is-${data.samplingState || 'unavailable'}` : ''}`} style={style}>
      {definition.directional && <defs><marker id={markerId} viewBox="0 0 12 10" markerWidth={12} markerHeight={10}
        refX={9} refY={5} orient="auto-start-reverse" markerUnits="userSpaceOnUse">
        <path d="M 3 1.5 L 9 5 L 3 8.5" fill="none" stroke={color}
          strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" />
      </marker></defs>}
      <path d={path} className="mas-edge-underlay" fill="none" pointerEvents="none" />
      <path d={path} className="mas-edge-halo" fill="none" pointerEvents="none" />
      <BaseEdge id={id} path={path} interactionWidth={20} className="mas-edge-line"
        markerEnd={definition.directional ? `url(#${markerId})` : undefined}
        style={{ stroke: color, strokeWidth: definition.width, strokeDasharray: definition.dash }} />
    </g>;
});

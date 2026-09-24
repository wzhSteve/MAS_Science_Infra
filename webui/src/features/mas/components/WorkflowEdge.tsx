import { memo, useId, type CSSProperties } from 'react';
import { BaseEdge, EdgeLabelRenderer, Position, getBezierPath, type EdgeProps } from '@xyflow/react';
import { Check, CornerUpLeft, Plus } from 'lucide-react';
import type { GraphEdge } from '../types';
import { edgeDefinition } from '../model/edgeDefinitions';
import { Tooltip } from '../../../shared/ui/tooltip';
import { useGraphInteraction } from './GraphInteractionContext';

const vectors: Record<Position, [number, number]> = {
  [Position.Top]: [0, -1], [Position.Right]: [1, 0], [Position.Bottom]: [0, 1], [Position.Left]: [-1, 0],
};

function cubicPoint(start: number, controlA: number, controlB: number, end: number, t: number) {
  const inverse = 1 - t;
  return inverse ** 3 * start
    + 3 * inverse ** 2 * t * controlA
    + 3 * inverse * t ** 2 * controlB
    + t ** 3 * end;
}

export const WorkflowEdge = memo(function WorkflowEdge(props: EdgeProps<GraphEdge>) {
  const { id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, data } = props;
  const interaction = useGraphInteraction();
  const markerId = `mas-arrow-${useId().replace(/:/g, '')}`;
  const definition = edgeDefinition(data?.kind || 'message');
  let [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  let markerX = labelX + (targetX - labelX) * 0.42;
  let markerY = labelY + (targetY - labelY) * 0.42;
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
    markerX = cubicPoint(sourceX, c1.x, c2.x, targetX, 0.72);
    markerY = cubicPoint(sourceY, c1.y, c2.y, targetY, 0.72);
  }
  const color = data?.issue ? 'var(--danger)' : definition.color;
  const style = { '--edge-color': color, '--edge-width': `${definition.width}px` } as CSSProperties;
  const opportunity = data?.samplingOpportunity;
  const sampling = data?.canvasMode === 'sampling';
  return <>
    <g className={`mas-flow-edge${selected ? ' is-selected' : ''}${data?.issue ? ' has-issue' : ''}${sampling ? ` is-sampling is-${data.samplingState || 'context'}${data.samplingSelected ? ' is-window-selected' : ''}${data.samplingHovered ? ' is-window-hovered' : ''}` : ''}`} style={style}>
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
    </g>
    {sampling && opportunity && <EdgeLabelRenderer>
      <div className="sampling-window-marker-host nodrag nopan" style={{
        transform: `translate(-50%, -50%) translate(${markerX}px, ${markerY}px)`,
      }}>
        <Tooltip content={<div className="sampling-window-tooltip"><strong>{opportunity.label}</strong>
          <dl>
            <div><dt>保留</dt><dd>此前对话和工具结果</dd></div>
            <div><dt>重新生成</dt><dd>{opportunity.anchor.agent_id || 'Agent'} 的后续回答</dd></div>
            <div><dt>状态</dt><dd>{opportunity.configured && opportunity.enabled ? '已启用' : '可启用'}</dd></div>
          </dl>
          <small>设计配置，不表示本次运行已发生分支</small>
        </div>}>
          <button type="button"
            className={`sampling-window-marker is-${data.samplingState || 'available'}${data.samplingSelected ? ' is-selected' : ''}`}
            aria-label={opportunity.label} aria-pressed={data.samplingSelected}
            onMouseEnter={() => interaction.onHoverSamplingOpportunity(opportunity.id)}
            onMouseLeave={() => interaction.onHoverSamplingOpportunity(null)}
            onClick={(event) => {
              event.stopPropagation();
              interaction.onSelectSamplingOpportunity(opportunity.id);
            }}>
            {opportunity.configured && opportunity.enabled ? <Check size={12} /> : <Plus size={12} />}
            <span>{opportunity.configured && opportunity.enabled ? '已启用分支' : '结果返回后分支'}</span>
            <CornerUpLeft className="sampling-window-return-icon" size={11} aria-hidden="true" />
          </button>
        </Tooltip>
      </div>
    </EdgeLabelRenderer>}
  </>;
});

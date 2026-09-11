import { memo, type DragEvent } from 'react';
import type { Palette } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';

export const NODE_TRANSFER = 'application/science-node';
export const PIN_TRANSFER = 'application/science-pin';

export const GraphPalette = memo(function GraphPalette({ palette, nodeCount, onAdd }: {
  palette: Palette;
  nodeCount: number;
  onAdd: (kind: 'agent' | 'tool', id: string, role: string) => void;
}) {
  const drag = (event: DragEvent, kind: 'agent' | 'tool', id: string, role: string) => {
    event.dataTransfer.setData(NODE_TRANSFER, JSON.stringify({ kind, id, role }));
    event.dataTransfer.effectAllowed = 'move';
  };
  return <div className="mas-palette">
    <div className="flex flex-wrap items-center gap-1.5" aria-label="添加 Agent">
      <span className="mas-palette__label">Agent</span>
      {(palette.roles || ['hub', 'planner', 'executor', 'verifier']).map((role) => {
        const id = role === 'hub' ? 'hub' : `${role}_${nodeCount + 1}`;
        return <Button key={role} size="sm" variant="ghost" draggable
          onDragStart={(event) => drag(event, 'agent', id, role)} onClick={() => onAdd('agent', id, role)}>
          + {role}
        </Button>;
      })}
    </div>
    <div className="flex flex-wrap items-center gap-1.5" aria-label="添加 Tool">
      <span className="mas-palette__label">Tool</span>
      {(palette.tools || []).map((tool) =>
        <Button key={tool} size="sm" variant="ghost" draggable
          onDragStart={(event) => drag(event, 'tool', tool, 'tool')}
          onClick={() => onAdd('tool', tool, 'tool')}>+ {tool}</Button>)}
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <Button size="sm" draggable onDragStart={(event) =>
        event.dataTransfer.setData(PIN_TRANSFER, JSON.stringify({ pin: 'entry' }))}>采集入口</Button>
      <span className="field-hint">点击或拖入节点；拖动入口到 Agent 设置 Episode 起点。</span>
    </div>
  </div>;
});

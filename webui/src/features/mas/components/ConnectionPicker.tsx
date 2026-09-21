import { useLayoutEffect, useRef, type RefObject } from 'react';
import { X, ArrowRight } from 'lucide-react';
import type { Connection, XYPosition } from '@xyflow/react';
import type { EdgeKind } from '../types';
import type { RelationOption } from '../model/edgeRules';
import { Button } from '../../../shared/ui/button';
import { RelationTypePicker } from './RelationTypePicker';

export function ConnectionPicker({ connection, anchor, workspace, options, onChoose, onClose }: {
  connection: Connection;
  anchor: XYPosition;
  workspace: RefObject<HTMLDivElement>;
  options: RelationOption[];
  onChoose: (kind: EdgeKind) => void;
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const popup = root.current;
    const container = workspace.current;
    if (!popup || !container) return;
    const place = () => {
      const bounds = container.getBoundingClientRect();
      const toolbox = container.querySelector('.mas-library')?.getBoundingClientRect();
      const leftLimit = toolbox && bounds.right - toolbox.right > popup.offsetWidth + 36 ? toolbox.right - bounds.left + 12 : 12;
      const left = Math.max(leftLimit, Math.min(anchor.x + 12, bounds.width - popup.offsetWidth - 12));
      const top = Math.max(12, Math.min(anchor.y + 12, bounds.height - popup.offsetHeight - 12));
      popup.style.left = `${left}px`;
      popup.style.top = `${top}px`;
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(container);
    observer.observe(popup);
    return () => observer.disconnect();
  }, [anchor, workspace]);
  useLayoutEffect(() => {
    root.current?.querySelector<HTMLInputElement>('input:not(:disabled)')?.focus();
  }, []);
  return <div ref={root} className="mas-connection-picker" role="dialog" aria-label="选择协作方式">
    <div className="mas-panel-heading"><h2>选择协作方式</h2>
      <Button size="sm" variant="ghost" aria-label="取消连接" onClick={onClose}><X size={15} /></Button></div>
    <div className="mas-connection-parties"><span>{connection.source}</span><ArrowRight size={14} /><span>{connection.target}</span></div>
    <div className="mas-connection-options"><RelationTypePicker options={options} onChange={onChoose} /></div>
    <p className="mas-panel-footnote">选择后建立关系 · Esc 取消</p>
  </div>;
}

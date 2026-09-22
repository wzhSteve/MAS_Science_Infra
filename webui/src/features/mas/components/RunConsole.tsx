import { memo, useEffect, useRef, useState, type CSSProperties } from 'react';
import { ChevronDown, ChevronUp, Maximize2, Minimize2, Terminal } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import { TrainingConsole, TRAIN_STATE } from '../../training/components/TrainingConsole';
import { useTraining, useTrainingViewRequest } from '../../../app/providers/RuntimeProvider';

const ConsoleSummary = memo(function ConsoleSummary() {
  const { data } = useTraining();
  return <span className="mas-console-summary-text">{data?.running
    ? `当前训练 · ${TRAIN_STATE[data.state] || '运行中'}`
    : '无活动训练'}</span>;
});

export const RunConsole = memo(function RunConsole({ experimentId, active, requestedOpen, requestedRunId }: {
  experimentId: string;
  active: boolean;
  requestedOpen?: boolean;
  requestedRunId?: string;
}) {
  const root = useRef<HTMLElement>(null);
  const drag = useRef<{ y: number; height: number; parentHeight: number } | null>(null);
  const [height, setHeight] = useState(48);
  const [maximized, setMaximized] = useState(false);
  const [open, setOpen] = useState(Boolean(requestedOpen));
  const viewRequest = useTrainingViewRequest();
  const lastRequest = useRef(viewRequest);
  useEffect(() => {
    if (requestedOpen || viewRequest !== lastRequest.current) setOpen(true);
    lastRequest.current = viewRequest;
  }, [requestedOpen, requestedRunId, viewRequest]);
  const clampHeight = (value: number) => Math.min(70, Math.max(25, value));
  const controls = <div className="console-window-controls">
    <Button size="sm" variant="ghost" onClick={() => setMaximized(value => !value)} aria-label={maximized ? '恢复控制台高度' : '最大化控制台'}>
      {maximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
    </Button>
    <Button size="sm" variant="ghost" onClick={() => setOpen(false)}><ChevronDown size={14} />收起</Button>
  </div>;

  return <section ref={root} className={`mas-run-console${open ? ' is-open' : ''}${open && maximized ? ' is-maximized' : ''}`}
    style={{ '--console-height': `${height}%` } as CSSProperties} aria-label="训练控制台">
    {open && !maximized && <div className="mas-console-resize" role="separator" tabIndex={0} aria-orientation="horizontal"
      aria-label="调整控制台高度" aria-valuenow={Math.round(height)} aria-valuemin={25} aria-valuemax={70}
      onPointerDown={event => {
        if (event.button !== 0) return;
        const parentHeight = root.current?.parentElement?.clientHeight;
        if (!parentHeight) return;
        drag.current = { y: event.clientY, height, parentHeight };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        const start = drag.current;
        if (start) setHeight(clampHeight(start.height + (start.y - event.clientY) / start.parentHeight * 100));
      }}
      onPointerUp={() => { drag.current = null; }}
      onLostPointerCapture={() => { drag.current = null; }}
      onKeyDown={event => {
        if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        setHeight(clampHeight(event.key === 'Home' ? 25 : event.key === 'End' ? 70 : height + (event.key === 'ArrowUp' ? 5 : -5)));
      }} />}
    {!open && <div className="mas-console-header">
      <button type="button" className="mas-console-toggle" aria-expanded={false} aria-controls="mas-console-content" onClick={() => setOpen(true)}>
        <ChevronUp size={14} /><Terminal size={14} /><strong>控制台</strong>
      </button>
      <span className="mas-console-summary" role="status"><ConsoleSummary /></span>
      <Button size="sm" variant="ghost" onClick={() => setOpen(true)}>展开</Button>
    </div>}
    <div id="mas-console-content" className="run-dock-layout" hidden={!open}>
      <TrainingConsole experimentId={experimentId} requestedRunId={requestedRunId} active={active && open} controls={controls} />
    </div>
  </section>;
});

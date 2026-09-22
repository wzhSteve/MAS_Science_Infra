import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { FlaskConical, Maximize2, Minimize2, X } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import { DownloadText } from '../../../shared/components/DownloadText';
import { ModelReadiness } from './ModelReadiness';
import { RolloutConfig, RolloutResults } from './RolloutRun';
import { useMasRolloutRun } from '../model/useMasRolloutRun';
import type { useMasDraft } from '../model/useMasDraft';
import type { ModelReadinessState } from '../model/useModelReadiness';
import type { TrajectoryLocation } from '../model/trajectoryView';

type Props = {
  experimentId: string;
  draft: ReturnType<typeof useMasDraft>;
  active: boolean;
  open: boolean;
  mode: 'live' | 'mock';
  executable: { ok: boolean; reason: string };
  readiness: ModelReadinessState;
  onClose: () => void;
  onLive: () => void;
  onConfigureModel: () => void;
  onLocate: (target: TrajectoryLocation) => void;
};

const DebugSession = memo(function DebugSession({ experimentId, draft, active, mode, executable, readiness, onConfigureModel, onLocate }: Omit<Props, 'open' | 'onClose' | 'onLive'>) {
  const rollout = useMasRolloutRun(experimentId, draft.executeWithSavedWorkflow, mode);
  const [currentRequest, setCurrentRequest] = useState(0);
  const input = useRef<HTMLDivElement>(null);
  const results = useRef<HTMLDivElement>(null);
  const showResults = useCallback(() => {
    setCurrentRequest(value => value + 1);
    results.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, []);
  const showInput = useCallback(() => {
    input.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    input.current?.querySelector('textarea')?.focus({ preventScroll: true });
  }, []);
  return <div className="workflow-debug-scroll" hidden={!active}>
    <div ref={input}>
      {mode === 'live' && <ModelReadiness readiness={readiness} onConfigureModel={onConfigureModel} />}
      <RolloutConfig rollout={rollout} pending={Boolean(draft.pending)} executable={executable}
        readiness={readiness} onShowResults={showResults} />
    </div>
    <div ref={results} className="workflow-debug-results">
      <div className="workflow-debug-results-heading"><h3>答案与执行过程</h3></div>
      <RolloutResults rollout={rollout} workflowChanged={Boolean(rollout.attempt && rollout.attempt.workflow !== draft.workflow)}
        workflow={draft.workflow} active={active} currentRequest={currentRequest}
        onLocate={onLocate} onUseQuestion={showInput} />
      {rollout.logs.length > 0 && <DownloadText
        text={rollout.logs.map(entry => `${new Date(entry.at).toISOString()} [${entry.tone}] ${entry.message}`).join('\n')}
        filename={`${rollout.attempt?.summary?.run_id || mode}-operations.log`} label="下载本页操作日志" />}
    </div>
  </div>;
});

export const WorkflowDebug = memo(function WorkflowDebug(props: Props) {
  const [expanded, setExpanded] = useState(false);
  const [mockVisited, setMockVisited] = useState(props.mode === 'mock');
  useEffect(() => { if (props.mode === 'mock') setMockVisited(true); }, [props.mode]);
  return <aside id="workflow-debug" className={`workflow-debug${expanded ? ' is-expanded' : ''}`} hidden={!props.open}
    aria-label="单题调试" onKeyDown={event => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        event.preventDefault(); event.stopPropagation(); props.onClose();
      }
    }}>
    <header className="workflow-debug-heading">
      <div><span>WORKFLOW DEBUG</span><h2><FlaskConical size={16} />{props.mode === 'mock' ? '模拟演示' : '单题调试'}</h2></div>
      <Button size="sm" variant="ghost" aria-label={expanded ? '恢复调试面板宽度' : '扩大调试面板'} onClick={() => setExpanded(value => !value)}>
        {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
      </Button>
      <Button size="sm" variant="ghost" aria-label="关闭单题调试" onClick={props.onClose}><X size={16} /></Button>
    </header>
    {props.mode === 'mock' && <div className="debug-mode-banner">Mock · 不调用真实模型<Button size="sm" variant="ghost" onClick={props.onLive}>返回真实调试</Button></div>}
    <DebugSession {...props} mode="live" active={props.active && props.open && props.mode === 'live'} />
    {(mockVisited || props.mode === 'mock') && <DebugSession {...props} mode="mock" active={props.active && props.open && props.mode === 'mock'} />}
  </aside>;
});

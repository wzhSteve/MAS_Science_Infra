import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Dialog } from 'radix-ui';
import { Terminal } from 'lucide-react';
import { runtimeApi } from '../../runtime/api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { useAction } from '../../../shared/hooks/useAction';
import { useRuntimeCommands, useTraining } from '../../../app/providers/RuntimeProvider';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { request } from '../../../shared/api/http';
import { runUrl, useRunLog } from '../model/useRunLog';
import { DownloadText } from '../../../shared/components/DownloadText';

export const TRAIN_STATE: Record<string, string> = {
  preparing: '准备中', starting: '启动中', running: '运行中', stopping: '停止中',
  succeeded: '已完成', failed: '失败', cancelled: '已取消', interrupted: '托管状态中断', idle: '尚未训练',
};
const LINE_HEIGHT = 20;

function RunClock({ started, ended, active }: { started?: number; ended?: number | null; active: boolean }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!started || ended || !active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [started, ended, active]);
  const seconds = Math.max(0, Math.floor(((ended ? ended * 1000 : now) - (started || now / 1000) * 1000) / 1000));
  return <span>{Math.floor(seconds / 3600).toString().padStart(2, '0')}:{Math.floor(seconds / 60 % 60).toString().padStart(2, '0')}:{(seconds % 60).toString().padStart(2, '0')}</span>;
}

export function Snapshot({ experimentId, runId }: { experimentId: string; runId: string }) {
  const [open, setOpen] = useState(false);
  const load = useCallback((signal: AbortSignal) =>
    request<Record<string, unknown>>(runUrl(experimentId, runId, 'snapshot'), { signal }), [experimentId, runId]);
  const data = usePollingResource(`snapshot:${runId}`, load, undefined, open);
  const text = useMemo(() => data.data ? JSON.stringify(data.data, null, 2) : '', [data.data]);
  return <Dialog.Root open={open} onOpenChange={setOpen}>
    <Dialog.Trigger asChild><Button size="sm" variant="ghost">本次配置</Button></Dialog.Trigger>
    <Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content training-snapshot">
      <Dialog.Title className="dialog-title">运行快照 · {runId}</Dialog.Title>
      <Dialog.Description className="dialog-description">只读启动配置，不是实验当前草稿。凭据字段已脱敏。</Dialog.Description>
      {data.error && <InlineNotice tone="danger">{data.error}</InlineNotice>}
      {data.loading && <p>读取中…</p>}
      {text && <><pre>{text}</pre><DownloadText text={text} filename={`${runId}-snapshot.json`} label="下载快照" /></>}
      <Dialog.Close asChild><Button>关闭</Button></Dialog.Close>
    </Dialog.Content></Dialog.Portal>
  </Dialog.Root>;
}

const RunOutput = memo(function RunOutput({ experimentId, runId, active, picker, controls }: {
  experimentId: string; runId: string; active: boolean; picker: ReactNode; controls: ReactNode;
}) {
  const log = useRunLog(experimentId, runId, active);
  const { stopTrain } = useRuntimeCommands();
  const action = useAction();
  const [follow, setFollow] = useState(true);
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [match, setMatch] = useState(0);
  const viewport = useRef<HTMLDivElement>(null);
  const previousFirst = useRef<number>();
  const [scroll, setScroll] = useState({ top: 0, height: 400 });
  useEffect(() => {
    const timer = setTimeout(() => { setSearch(query.toLocaleLowerCase()); setMatch(0); }, 200);
    return () => clearTimeout(timer);
  }, [query]);
  const hits = useMemo(() => search ? log.lines.flatMap((line, index) =>
    line.text.toLocaleLowerCase().includes(search) ? [index] : []) : [], [log.lines, search]);
  useLayoutEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setScroll(current => ({ ...current, height: element.clientHeight })));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useLayoutEffect(() => {
    const element = viewport.current;
    const firstId = log.lines[0]?.id;
    if (element) {
      if (follow) element.scrollTop = element.scrollHeight;
      else if (firstId !== undefined && previousFirst.current !== undefined && firstId > previousFirst.current) {
        element.scrollTop = Math.max(0, element.scrollTop - (firstId - previousFirst.current) * LINE_HEIGHT);
      }
    }
    previousFirst.current = firstId;
  }, [log.lines, follow]);
  const first = Math.max(0, Math.min(log.lines.length - 1, Math.floor(scroll.top / LINE_HEIGHT) - 10));
  const end = Math.min(log.lines.length, first + Math.ceil(scroll.height / LINE_HEIGHT) + 20);
  const state = log.state;
  const info = state?.meta;
  const jump = (index: number) => {
    if (!hits.length) return;
    const next = (index + hits.length) % hits.length;
    setMatch(next); setFollow(false);
    viewport.current?.scrollTo({ top: hits[next] * LINE_HEIGHT });
  };
  return <div className="training-output">
    <div className="training-run-heading">
      <Terminal size={14} /><strong>控制台</strong>{picker}
      <strong>{TRAIN_STATE[state?.state || ''] || '读取状态…'}</strong>
      {info?.algo != null && <span>{String(info.algo).toUpperCase()}</span>}
      {info?.cuda_visible_devices != null && <span>GPU {String(info.cuda_visible_devices)}</span>}
      {info?.n_runners != null && <span>Runner {String(info.n_runners)}</span>}
      {info?.group_n != null && <span>n={String(info.group_n)}</span>}
      <RunClock started={state?.started_at} ended={state?.ended_at} active={active && Boolean(state?.running)} />
      <span className="console-connection" title={log.connection} aria-label={log.connection} />
      <span className="console-heading-spacer" />
      <Snapshot experimentId={experimentId} runId={runId} />
      {state?.running && <Button size="sm" variant="danger"
        disabled={state.state === 'stopping' && state.failure_stage !== 'stop' || action.pending !== null}
        onClick={() => void action.run('stop', async () => { await stopTrain(runId); })}>
        {state.failure_stage === 'stop' ? '重试停止' : state.state === 'stopping' ? '停止中…' : '停止训练'}
      </Button>}
      {controls}
    </div>
    {state?.message && <InlineNotice tone={state.state === 'failed' ? 'danger' : 'warning'}>
      {state.failure_stage ? `${state.failure_stage}：` : ''}{state.message}{state.returncode != null ? `（退出码 ${state.returncode}）` : ''}
    </InlineNotice>}
    {state?.state === 'interrupted' && <InlineNotice tone="warning">Control 无法确认原进程状态，不代表服务器进程已经停止。</InlineNotice>}
    {log.error && <InlineNotice tone="warning">{log.error}<Button size="sm" onClick={log.reconnect}>重试日志</Button></InlineNotice>}
    {action.notice && <InlineNotice tone={action.notice.tone}>{action.notice.message}</InlineNotice>}
    <div className="training-log-tools">
      <Input aria-label="搜索当前已加载日志" placeholder="搜索当前窗口…" value={query} onChange={event => setQuery(event.target.value)} />
      {search && <><span>{hits.length ? `${match % hits.length + 1}/${hits.length}` : '0 个命中'}</span>
        <Button size="sm" disabled={!hits.length} onClick={() => jump(match - 1)}>上一个</Button>
        <Button size="sm" disabled={!hits.length} onClick={() => jump(match + 1)}>下一个</Button></>}
      <Button size="sm" aria-pressed={follow} onClick={() => setFollow(value => !value)}>{follow ? '暂停跟随' : '跟随末尾'}</Button>
      <Button size="sm" onClick={() => void action.run('copy', async () => {
        const selected = window.getSelection()?.toString();
        await navigator.clipboard.writeText(selected || log.lines.map(line => line.text).join('\n'));
        return selected ? '已复制选中文字' : '已复制当前日志窗口';
      })}>复制</Button>
      <a href={runUrl(experimentId, runId, 'log/download')} download>下载完整日志</a>
    </div>
    {log.trimmed && <p className="training-window-hint">显示有界日志窗口；更早内容或超长行请下载完整日志。暂停跟随不会暂停训练。</p>}
    <div className="training-log-viewport" ref={viewport} tabIndex={0} aria-label="训练输出"
      onScroll={event => {
        const element = event.currentTarget;
        setScroll({ top: element.scrollTop, height: element.clientHeight });
        if (element.scrollHeight - element.scrollTop - element.clientHeight > 40) setFollow(false);
      }}>
      {!log.lines.length && <p className="training-log-empty">等待训练输出…</p>}
      <div style={{ height: log.lines.length * LINE_HEIGHT, position: 'relative' }}>
        <div style={{ position: 'absolute', top: first * LINE_HEIGHT, left: 0, minWidth: '100%' }}>
          {log.lines.slice(first, end).map(line => <div key={line.id} className={`training-log-line${search && line.text.toLocaleLowerCase().includes(search) ? ' is-match' : ''}`}>
            <span className="training-line-number" aria-hidden="true">{line.id + 1}</span>{line.text || ' '}
          </div>)}
        </div>
      </div>
    </div>
  </div>;
});

export const TrainingConsole = memo(function TrainingConsole({ experimentId, requestedRunId, active, controls }: {
  experimentId: string; requestedRunId?: string; active: boolean; controls: ReactNode;
}) {
  const activity = useTraining();
  const { viewTraining } = useRuntimeCommands();
  const load = useCallback((signal: AbortSignal) => runtimeApi.trainingRuns(experimentId, signal), [experimentId]);
  const history = usePollingResource(`training-history:${experimentId}:${activity.data?.runId}:${activity.data?.state}`, load, undefined, active);
  const [selected, setSelected] = useState<string | undefined>(requestedRunId);
  useEffect(() => { if (requestedRunId) setSelected(requestedRunId); }, [requestedRunId]);
  useEffect(() => {
    if (!selected && history.data) setSelected(activity.data?.runId || history.data.runs[0]?.run_id);
  }, [history.data, selected, activity.data?.runId]);
  const runId = requestedRunId || selected;
  const picker = <Select className="console-run-picker" aria-label="当前或最近训练" value={runId || ''} onChange={event => {
        setSelected(event.target.value); viewTraining(event.target.value);
      }}>
        {!runId && <option value="">尚无训练运行</option>}
        {runId && !history.data?.runs.some(item => item.run_id === runId) && <option value={runId}>{runId}</option>}
        {history.data?.runs.map(item => <option key={item.run_id} value={item.run_id}>
          {item.run_id === activity.data?.runId && item.running ? '当前运行' : '历史运行'} · {item.run_id}
        </option>)}
      </Select>;
  return <div className="training-console">
    {history.error && <InlineNotice tone="danger">运行列表读取失败：{history.error}<Button size="sm" onClick={() => void history.refresh()}>重试</Button></InlineNotice>}
    {runId ? <RunOutput key={runId} experimentId={experimentId} runId={runId} active={active} picker={picker} controls={controls} />
      : <><div className="training-run-heading"><Terminal size={14} /><strong>控制台</strong>{picker}<span className="console-heading-spacer" />{controls}</div>
        <div className="training-console-empty"><span className="training-console-empty-mark" aria-hidden="true">&gt;_</span><strong>{history.loading ? '正在读取运行…' : '暂无训练日志'}</strong></div></>}
  </div>;
});

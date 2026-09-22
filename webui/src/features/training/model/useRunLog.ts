import { useEffect, useRef, useState } from 'react';
import { ApiError, errorMessage, experimentQuery, request } from '../../../shared/api/http';
import type { RunSummary } from '../../../shared/api/types';

interface LogBlock {
  offset: number; next_offset: number; size: number; text: string;
  generation: string; reset: boolean; terminal: boolean; state?: RunSummary;
}
export interface LogLine { id: number; text: string }
const MAX_LINES = 10000;
const MAX_CHARS = 1024 * 1024;

export function runUrl(experimentId: string, runId: string, endpoint: string) {
  return `/api/rl/runs/${encodeURIComponent(runId)}/${endpoint}?${experimentQuery(experimentId)}`;
}

export function useRunLog(experimentId: string, runId: string, active: boolean) {
  const [foreground, setForeground] = useState(!document.hidden);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [state, setState] = useState<RunSummary | null>(null);
  const [connection, setConnection] = useState('正在连接');
  const [error, setError] = useState('');
  const [trimmed, setTrimmed] = useState(false);
  const [retryVersion, setRetryVersion] = useState(0);
  const buffer = useRef({ offset: undefined as number | undefined, generation: '', lines: [] as LogLine[],
    chars: 0, nextId: 0, partial: false, clipped: false, trimmed: false });
  useEffect(() => {
    const changed = () => setForeground(!document.hidden);
    document.addEventListener('visibilitychange', changed);
    return () => document.removeEventListener('visibilitychange', changed);
  }, []);
  useEffect(() => {
    if (!active || !foreground) { setConnection('已暂停读取'); return; }
    const data = buffer.current;
    const controller = new AbortController();
    let source: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let flush: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    let fallbackSince = 0;
    const publish = () => {
      flush = undefined;
      setLines([...data.lines]);
      setTrimmed(data.trimmed);
    };
    const consume = (block: LogBlock) => {
      if (block.state) setState(block.state);
      if (block.reset) {
        data.lines = []; data.chars = 0; data.partial = false; data.clipped = false;
        data.trimmed = true;
        setError('日志文件已重置，当前显示新文件的末尾。完整日志可下载。');
      }
      if (!block.reset && data.offset !== undefined && block.next_offset <= data.offset) return;
      data.trimmed ||= block.offset > 0 && data.offset === undefined;
      data.offset = block.next_offset;
      data.generation = block.generation;
      if (!block.text) return;
      const parts = block.text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r/g, '\n').split('\n');
      parts.forEach((part, index) => {
        if (index === parts.length - 1 && !part) { data.partial = false; data.clipped = false; return; }
        const previous = data.partial ? data.lines.pop() : undefined;
        const text = (previous?.text || '') + (data.clipped ? '' : part);
        if (previous) data.chars -= previous.text.length;
        const clipped = text.length > 8192;
        const shown = clipped ? `${text.slice(0, 8192)} …[长行已截断，请下载原日志]` : text;
        data.lines.push({ id: previous?.id ?? data.nextId++, text: shown });
        data.chars += shown.length;
        data.partial = index === parts.length - 1;
        data.clipped = data.partial && (data.clipped || clipped);
        if (!data.partial) data.clipped = false;
      });
      let remove = 0;
      while (data.lines.length - remove > MAX_LINES || data.chars > MAX_CHARS) {
        data.chars -= data.lines[remove++].text.length;
        data.trimmed = true;
      }
      if (remove) data.lines = data.lines.slice(remove);
      if (!flush) flush = setTimeout(publish, 150);
    };
    const url = (endpoint: string) => runUrl(experimentId, runId, endpoint)
      + (data.offset === undefined ? '' : `&offset=${data.offset}&generation=${encodeURIComponent(data.generation)}`);
    const poll = async () => {
      try {
        const block = await request<LogBlock>(url('log'), { signal: controller.signal });
        if (controller.signal.aborted) return;
        consume(block);
        setConnection('增量补读');
        if (block.terminal && block.next_offset >= block.size) { setConnection('历史日志'); return; }
        retry = setTimeout(Date.now() - fallbackSince > 30000 ? connect : () => void poll(),
          block.next_offset < block.size ? 150 : 3000);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(errorMessage(reason)); setConnection('读取失败，正在重试');
        if (reason instanceof ApiError && reason.status && reason.status >= 400 && reason.status < 500 && reason.status !== 429) {
          setConnection('日志不可用'); return;
        }
        retry = setTimeout(connect, 5000);
      }
    };
    const connect = () => {
      if (controller.signal.aborted) return;
      setConnection('正在连接');
      source = new EventSource(url('events'));
      source.addEventListener('state', event => setState(JSON.parse((event as MessageEvent<string>).data) as RunSummary));
      source.addEventListener('stdout', event => {
        const block = JSON.parse((event as MessageEvent<string>).data) as LogBlock;
        consume(block);
        failures = 0;
        if (!block.reset) setError('');
        setConnection(block.terminal ? '读取历史' : '实时');
      });
      source.addEventListener('complete', () => {
        source?.close(); publish(); setConnection('历史日志');
      });
      source.onerror = () => {
        source?.close();
        setConnection('重连中');
        if (++failures >= 3) { fallbackSince = Date.now(); void poll(); }
        else retry = setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      controller.abort(); source?.close(); clearTimeout(retry); clearTimeout(flush);
      publish();
    };
  }, [experimentId, runId, active, foreground, retryVersion]);
  return { lines, state, connection, error, trimmed, reconnect: () => setRetryVersion(value => value + 1) };
}

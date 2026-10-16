import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Bot, Eraser, Paperclip, Send, Sparkles, Square, X } from 'lucide-react';
import type { WorkflowSpec } from '../../shared/api/types';
import { Button } from '../../shared/ui/button';
import { Select } from '../../shared/ui/select';
import { assistantApi, workflowSummary, type UserProject } from './api';
import { useAssistantCanvas } from './context';
import { MarkdownMessage } from './MarkdownMessage';

const WELCOME = [
  '我可以读仓库、改 user_space/ 里的用户代码，并把适配结果应用到画布。也可以直接用文字问我：',
  '- 训练资源：「训练 agentflow 要几张卡？显存怎么分？」',
  '- 训练诊断：「训练崩了，帮我看看日志」',
  '- 实验配置：「切换到 ARPO 算法」或「把采样站点改成 planner + verifier」',
].join('\n');

const LEGACY_WELCOME_PREFIXES = ['我可以读仓库', '我只能改 user_space/'];

function isWelcome(item: ChatItem) {
  return item.role === 'assistant' && LEGACY_WELCOME_PREFIXES.some((prefix) => item.content.startsWith(prefix));
}

type ChatItem =
  | { id: string; role: 'user' | 'assistant'; content: string; streaming?: boolean }
  | { id: string; role: 'tool'; name: string; status: 'running' | 'done' | 'error'; detail?: string };

function newId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function storageKey(expId: string | null) {
  return `science-assistant-chat:${expId || 'global'}`;
}

function loadItems(expId: string | null): ChatItem[] {
  try {
    const raw = window.localStorage.getItem(storageKey(expId));
    if (!raw) return [{ id: 'welcome', role: 'assistant', content: WELCOME }];
    const parsed = JSON.parse(raw) as ChatItem[];
    if (!Array.isArray(parsed) || !parsed.length) return [{ id: 'welcome', role: 'assistant', content: WELCOME }];
    // 旧欢迎语不含技能提示：替换为最新版本，让老用户也能看到技能示例。
    if (parsed.length === 1 && isWelcome(parsed[0])) {
      return [{ id: 'welcome', role: 'assistant', content: WELCOME }];
    }
    return parsed;
  } catch {
    return [{ id: 'welcome', role: 'assistant', content: WELCOME }];
  }
}

function historyForApi(items: ChatItem[]) {
  return items
    .filter((item): item is Extract<ChatItem, { role: 'user' | 'assistant' }> => item.role === 'user' || item.role === 'assistant')
    .filter((item) => item.content && !isWelcome(item))
    .slice(-24)
    .map((item) => ({ role: item.role, content: item.content }));
}

// tool_start 事件的 args 可能是对象，也可能是 JSON 字符串；解析出技能名（load_skill 用）。
function skillNameFromArgs(args: unknown): string {
  let parsed: unknown = args;
  if (typeof args === 'string') {
    try {
      parsed = JSON.parse(args);
    } catch {
      return '';
    }
  }
  if (parsed && typeof parsed === 'object') {
    const name = (parsed as Record<string, unknown>).name;
    if (typeof name === 'string' && name.trim()) return name.trim();
  }
  return '';
}

// 工具条文案：技能类工具（list_skills/load_skill/query_*）显示「技能」措辞，与状态栏一致。
function toolSummary(item: Extract<ChatItem, { role: 'tool' }>): string {
  const isSkill = item.name === 'load_skill' || item.name === 'list_skills';
  const label = isSkill ? `技能 ${item.name}` : item.name;
  if (item.status === 'running') return `正在使用 ${label}`;
  if (item.status === 'error') return `${label} 失败`;
  return `已完成 ${label}`;
}

export function AssistantDock() {
  const canvas = useAssistantCanvas();
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [projects, setProjects] = useState<UserProject[]>([]);
  const [projectId, setProjectId] = useState('');
  const [items, setItems] = useState<ChatItem[]>(() => loadItems(null));
  const fileRef = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const expKey = canvas.expId || 'global';

  useEffect(() => {
    setItems(loadItems(canvas.expId));
  }, [expKey]);

  useEffect(() => {
    window.localStorage.setItem(
      storageKey(canvas.expId),
      JSON.stringify(items.filter((item) => item.role === 'tool' || !item.streaming)),
    );
  }, [items, expKey]);

  const reloadProjects = useCallback(async () => {
    const data = await assistantApi.listProjects(canvas.expId || undefined);
    setProjects(data.projects || []);
    setProjectId((current) => current || data.projects?.[0]?.id || '');
    canvas.reloadPalette();
  }, [canvas]);

  useEffect(() => {
    if (open) void reloadProjects().catch(() => undefined);
  }, [open, reloadProjects]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [items, open, status]);

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setStatus('');
    setItems((prev) => prev.map((item) => (item.role === 'assistant' && item.streaming ? { ...item, streaming: false } : item)));
  };

  const send = async () => {
    const text = message.trim();
    if (!text || busy) return;
    setMessage('');
    const userItem: ChatItem = { id: newId(), role: 'user', content: text };
    const assistantId = newId();
    const pending: ChatItem[] = [...items, userItem];
    setItems([...pending, { id: assistantId, role: 'assistant', content: '', streaming: true }]);
    setBusy(true);
    setStatus('正在读工作区…');
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await assistantApi.chat({
        message: text,
        experiment_id: canvas.expId || undefined,
        project_id: projectId || undefined,
        history: historyForApi(pending),
        workflow_summary: workflowSummary(canvas.workflow),
        selected_node_id: canvas.selectedNodeId || undefined,
        current_workflow: canvas.workflow || undefined,
      }, (event) => {
        if (event.event === 'status') {
          const phase = String(event.data.phase || '');
          setStatus(phase === 'calling_tools' ? '正在调用工具…' : '正在思考…');
        } else if (event.event === 'token') {
          const piece = String(event.data.text || '');
          if (!piece) return;
          setItems((prev) => prev.map((item) => (
            item.id === assistantId && item.role === 'assistant'
              ? { ...item, content: item.content + piece, streaming: true }
              : item
          )));
        } else if (event.event === 'tool_start') {
          const name = String(event.data.name || 'tool');
          setItems((prev) => [...prev, {
            id: `tool-${name}-${newId()}`,
            role: 'tool',
            name,
            status: 'running',
            detail: JSON.stringify(event.data.args || {}),
          }]);
          setStatus(
            name === 'load_skill'
              ? (() => {
                  const skillName = skillNameFromArgs(event.data.args);
                  return skillName ? `正在使用技能 ${skillName}…` : '正在使用技能…';
                })()
              : `正在 ${name}…`,
          );
        } else if (event.event === 'tool_end') {
          const name = String(event.data.name || 'tool');
          const ok = event.data.ok !== false;
          setItems((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) {
              const item = next[i];
              if (item.role === 'tool' && item.name === name && item.status === 'running') {
                next[i] = { ...item, status: ok ? 'done' : 'error', detail: String(event.data.result || item.detail || '') };
                break;
              }
            }
            return next;
          });
        } else if (event.event === 'error') {
          setItems((prev) => prev.map((item) => (
            item.id === assistantId && item.role === 'assistant'
              ? { ...item, content: item.content || String(event.data.message || '辅助对话失败'), streaming: false }
              : item
          )));
        } else if (event.event === 'done') {
          const workflow = event.data.workflow as WorkflowSpec | undefined;
          if (workflow) canvas.applyWorkflow(workflow);
          setItems((prev) => prev.map((item) => (
            item.id === assistantId && item.role === 'assistant'
              ? { ...item, content: item.content || (event.data.offline ? '（离线说明）' : ''), streaming: false }
              : item
          )));
        }
      }, controller.signal);
      await reloadProjects();
    } catch (error) {
      if ((error as { name?: string }).name === 'AbortError') return;
      setItems((prev) => prev.map((item) => (
        item.id === assistantId && item.role === 'assistant'
          ? { ...item, content: error instanceof Error ? error.message : String(error), streaming: false }
          : item
      )));
    } finally {
      abortRef.current = null;
      setBusy(false);
      setStatus('');
    }
  };

  const upload = async (file: File) => {
    const form = new FormData();
    form.append('title', file.name.replace(/\.[^.]+$/, '') || 'user-agent');
    form.append('mode', 'io_module');
    if (canvas.expId) form.append('experiment_id', canvas.expId);
    form.append('file', file);
    setBusy(true);
    try {
      const result = await assistantApi.upload(form);
      setProjectId(result.project.id);
      setItems((prev) => [...prev, {
        id: newId(),
        role: 'assistant',
        content: `已收入用户区项目 **${result.project.id}**（${result.project.mode}）。可以说「应用到画布」，或继续让我改 \`adapted/\`。`,
      }]);
      await reloadProjects();
    } catch (error) {
      setItems((prev) => [...prev, { id: newId(), role: 'assistant', content: error instanceof Error ? error.message : String(error) }]);
    } finally {
      setBusy(false);
    }
  };

  const apply = async () => {
    if (!projectId || !canvas.workflow) {
      setItems((prev) => [...prev, { id: newId(), role: 'assistant', content: canvas.workflow ? '请先选一个用户项目。' : '请先打开一个实验工作区。' }]);
      return;
    }
    setBusy(true);
    try {
      const result = await assistantApi.apply(projectId, canvas.workflow, false);
      canvas.applyWorkflow(result.workflow);
      const n = result.workflow?.agents?.length || 0;
      const wraps = (result.workflow?.agents || []).map((agent) => {
        const label = agent.meta?.wraps;
        return typeof label === 'string' ? label : agent.id;
      }).filter(Boolean);
      setItems((prev) => [...prev, {
        id: newId(),
        role: 'assistant',
        content: n
          ? `已按解析重构写入画布（${n} 个节点${wraps.length ? `：${wraps.join(' / ')}` : ''}）。请确认后再点顶栏「保存实验」。`
          : '已写入当前实验草稿。请到 MAS 画布确认，再点顶栏「保存实验」。',
      }]);
    } catch (error) {
      setItems((prev) => [...prev, { id: newId(), role: 'assistant', content: error instanceof Error ? error.message : String(error) }]);
    } finally {
      setBusy(false);
    }
  };

  const resetChat = () => {
    stop();
    setItems([{ id: 'welcome', role: 'assistant', content: WELCOME }]);
  };

  const onComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  const contextHint = useMemo(() => {
    const bits = [
      canvas.expId ? `实验 ${canvas.expId}` : null,
      projectId ? `项目 ${projectId}` : null,
      canvas.selectedNodeId ? `选中 ${canvas.selectedNodeId}` : null,
    ].filter(Boolean);
    return bits.join(' · ');
  }, [canvas.expId, canvas.selectedNodeId, projectId]);

  return <>
    <button type="button" className="assistant-fab" aria-expanded={open} aria-controls="assistant-dock"
      onClick={() => setOpen((value) => !value)}>
      <Sparkles size={18} />
      <span>AI 辅助</span>
    </button>
    {open && <section id="assistant-dock" className="assistant-dock" aria-label="用户区 AI 辅助">
      <header className="assistant-dock-head">
        <div>
          <strong><Bot size={15} /> 辅助对话</strong>
          <small>{contextHint || '只写 user_space/ · 管理区只读'}</small>
        </div>
        <div className="assistant-dock-head-actions">
          <Button variant="ghost" size="sm" onClick={resetChat} aria-label="新对话"><Eraser size={15} /></Button>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)} aria-label="关闭辅助"><X size={16} /></Button>
        </div>
      </header>
      <div className="assistant-dock-controls">
        <Select aria-label="用户项目" value={projectId} onChange={(event) => setProjectId(event.target.value)} className="assistant-project-select">
          <option value="">未选择项目</option>
          {projects.map((project) => <option key={project.id} value={project.id}>{project.title} · {project.mode}</option>)}
        </Select>
        <Button size="sm" onClick={() => fileRef.current?.click()} disabled={busy}><Paperclip size={14} />上传</Button>
        <Button size="sm" variant="primary" onClick={() => void apply()} disabled={busy || !projectId}>应用到画布</Button>
        <input ref={fileRef} type="file" hidden onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) void upload(file);
        }} />
      </div>
      <div className="assistant-dock-log" ref={scroller}>
        {items.map((item) => item.role === 'tool' ? (
          <details key={item.id} className={`assistant-tool is-${item.status}`} open={item.status === 'running'}>
            <summary>{toolSummary(item)}</summary>
            {item.detail ? <pre>{item.detail}</pre> : null}
          </details>
        ) : (
          <div key={item.id} className={`assistant-bubble is-${item.role}${item.role === 'assistant' && item.streaming ? ' is-streaming' : ''}`}>
            <span>{item.role === 'user' ? '你' : '辅助'}</span>
            {item.role === 'user' ? <p>{item.content}</p> : <MarkdownMessage content={item.content || (item.streaming ? '' : '（没有回复）')} />}
          </div>
        ))}
        {busy && status ? <div className="assistant-status" role="status">{status}</div> : null}
      </div>
      <form className="assistant-dock-composer" onSubmit={(event) => { event.preventDefault(); void send(); }}>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={onComposerKey}
          placeholder="问画布、读文件，或试试「训练 agentflow 要几张卡」… Enter 发送，Shift+Enter 换行"
          disabled={busy}
          aria-label="辅助输入"
          rows={2}
        />
        {busy ? (
          <Button type="button" variant="danger" onClick={stop} aria-label="停止生成"><Square size={14} /></Button>
        ) : (
          <Button type="submit" variant="primary" disabled={!message.trim()} aria-label="发送"><Send size={14} /></Button>
        )}
      </form>
    </section>}
  </>;
}

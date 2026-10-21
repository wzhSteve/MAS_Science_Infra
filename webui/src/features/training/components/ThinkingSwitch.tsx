import { memo, useCallback, useEffect, useState } from 'react';
import { request } from '../../../shared/api/http';
import { Checkbox } from '../../../shared/ui/checkbox';

type ThinkingSetting = { enable_thinking: boolean; applies: string };
const CHANGE = 'enable-thinking-change';

export const ThinkingSwitch = memo(function ThinkingSwitch() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    request<ThinkingSetting>('/api/runtime/thinking', { signal: controller.signal })
      .then(data => setEnabled(data.enable_thinking))
      .catch(reason => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : '读取失败');
      });
    const onChange = (event: Event) => setEnabled(Boolean((event as CustomEvent<boolean>).detail));
    window.addEventListener(CHANGE, onChange);
    return () => {
      controller.abort();
      window.removeEventListener(CHANGE, onChange);
    };
  }, []);

  const change = useCallback(async (next: boolean) => {
    setPending(true);
    setError('');
    try {
      const data = await request<ThinkingSetting>('/api/runtime/thinking', {
        method: 'PUT',
        body: JSON.stringify({ enable_thinking: next }),
      });
      setEnabled(data.enable_thinking);
      window.dispatchEvent(new CustomEvent(CHANGE, { detail: data.enable_thinking }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败');
    } finally {
      setPending(false);
    }
  }, []);

  const on = Boolean(enabled);
  return <label className="thinking-switch" title="写入 .env 的 enable_thinking。已经启动的训练保持原设置，下次启动生效。">
    <Checkbox
      checked={on}
      disabled={enabled === null || pending}
      aria-label="模型思考"
      onChange={event => void change(event.target.checked)}
    />
    <span>思考{enabled === null ? '…' : on ? '开' : '关'}</span>
    {error && <span className="thinking-switch-error">{error}</span>}
  </label>;
});

import { useCallback, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/http';
import type { Notice } from '../components/InlineNotice';
import { useNotify } from '../feedback/useNotify';

interface UseActionOptions {
  feedback?: 'toast' | 'inline' | 'silent';
  successFeedback?: 'toast' | 'inline' | 'silent';
  errorFeedback?: 'toast' | 'inline' | 'silent';
  errorTitle?: string;
}

export function useAction(options: UseActionOptions = {}) {
  const notify = useNotify();
  const successFeedback = options.successFeedback || options.feedback || 'toast';
  const errorFeedback = options.errorFeedback || options.feedback || 'toast';
  const [pending, setPending] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const lock = useRef(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

  const run = useCallback(async (name: string, action: () => Promise<string | void>): Promise<boolean> => {
    if (lock.current) return false;
    lock.current = true;
    setPending(name);
    setNotice(null);
    try {
      const message = await action();
      if (mounted.current && message) {
        if (successFeedback === 'toast') notify.success(message);
        else if (successFeedback === 'inline') setNotice({ message, tone: 'success' });
      }
      return true;
    } catch (error) {
      if (mounted.current) {
        const message = errorMessage(error);
        if (errorFeedback === 'toast') notify.error(message, { title: options.errorTitle || '操作失败' });
        else if (errorFeedback === 'inline') setNotice({ message, tone: 'danger' });
      }
      return false;
    } finally {
      lock.current = false;
      if (mounted.current) setPending(null);
    }
  }, [errorFeedback, notify, options.errorTitle, successFeedback]);
  return { pending, notice, setNotice, run };
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../api/http';
import type { Notice } from '../components/InlineNotice';

export function useAction() {
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
      if (mounted.current && message) setNotice({ message, tone: 'success' });
      return true;
    } catch (error) {
      if (mounted.current) setNotice({ message: errorMessage(error), tone: 'danger' });
      return false;
    } finally {
      lock.current = false;
      if (mounted.current) setPending(null);
    }
  }, []);
  return { pending, notice, setNotice, run };
}

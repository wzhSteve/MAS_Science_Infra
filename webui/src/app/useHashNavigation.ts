import { useCallback, useEffect, useRef, useState } from 'react';
import { HOME_HASH, parseRoute, routeHash, type AppRoute, type WorkspaceRoute } from './navigation';
import { useConfirmExperimentLeave } from './providers/NavigationGuard';

interface Entry { hash: string; index: number; route: AppRoute }
interface NavigationState { route: AppRoute; workspace: WorkspaceRoute | null }
interface Traversal { target: Entry; phase: 'restore' | 'confirm' | 'apply' }

function historyIndex(): number | null {
  const state: unknown = window.history.state;
  if (state && typeof state === 'object' && 'scienceNavigationIndex' in state
    && Number.isSafeInteger(state.scienceNavigationIndex)) return state.scienceNavigationIndex as number;
  return null;
}

function writeHistory(entry: Entry, replace: boolean) {
  const previous: unknown = window.history.state;
  const state = {
    ...(previous && typeof previous === 'object' && !Array.isArray(previous) ? previous : {}),
    scienceNavigationIndex: entry.index,
  };
  if (replace) window.history.replaceState(state, '', entry.hash);
  else window.history.pushState(state, '', entry.hash);
}

function initialEntry(): Entry {
  const route = parseRoute(window.location.hash);
  const hash = route.kind === 'not-found' ? window.location.hash : routeHash(route);
  return { hash: hash || HOME_HASH, index: historyIndex() ?? 0, route };
}

export function useHashNavigation() {
  const confirmLeave = useConfirmExperimentLeave();
  const [initial] = useState(initialEntry);
  const entry = useRef(initial);
  const workspace = useRef<WorkspaceRoute | null>(initial.route.kind === 'workspace' ? initial.route : null);
  const [state, setState] = useState<NavigationState>({ route: initial.route, workspace: workspace.current });
  const mounted = useRef(true);
  const navigating = useRef(false);
  const traversal = useRef<Traversal | null>(null);
  const changesExperiment = useCallback((route: AppRoute) =>
    route.kind === 'workspace' && workspace.current !== null && route.experimentId !== workspace.current.experimentId, []);
  const commit = useCallback((next: Entry) => {
    entry.current = next;
    if (next.route.kind === 'workspace') workspace.current = next.route;
    setState({ route: next.route, workspace: workspace.current });
  }, []);

  const navigate = useCallback(async (route: Exclude<AppRoute, { kind: 'not-found' }>) => {
    const hash = routeHash(route);
    if (navigating.current || traversal.current || hash === entry.current.hash) return;
    navigating.current = true;
    try {
      if (changesExperiment(route) && !(await confirmLeave())) return;
      if (!mounted.current) return;
      const next = { hash, route, index: entry.current.index + 1 };
      writeHistory(next, false);
      commit(next);
    } finally {
      navigating.current = false;
    }
  }, [changesExperiment, commit, confirmLeave]);

  useEffect(() => {
    mounted.current = true;
    writeHistory(entry.current, true);
    const onHashChange = () => {
      const hash = window.location.hash || HOME_HASH;
      let index = historyIndex();
      if (index === null || (index === entry.current.index && hash !== entry.current.hash)) {
        index = entry.current.index + 1;
        writeHistory({ hash, index, route: parseRoute(hash) }, true);
      }
      const next = { hash, index, route: parseRoute(hash) };
      const pending = traversal.current;
      if (pending?.phase === 'apply') {
        traversal.current = null;
        commit(next);
        return;
      }
      if (pending) {
        if (next.index !== entry.current.index) {
          window.history.go(entry.current.index - next.index);
          return;
        }
        if (pending.phase !== 'restore') return;
        pending.phase = 'confirm';
        void confirmLeave().then(allow => {
          if (!mounted.current || traversal.current !== pending) return;
          if (!allow) { traversal.current = null; return; }
          pending.phase = 'apply';
          window.history.go(pending.target.index - entry.current.index);
        });
        return;
      }
      if (hash === entry.current.hash) return;
      if (navigating.current || changesExperiment(next.route)) {
        // Restore the current history entry before asking, so Cancel also restores the URL.
        traversal.current = { target: next, phase: 'restore' };
        window.history.go(entry.current.index - next.index);
        return;
      }
      commit(next);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => {
      mounted.current = false;
      window.removeEventListener('hashchange', onHashChange);
    };
  }, [changesExperiment, commit, confirmLeave]);

  return { ...state, navigate };
}

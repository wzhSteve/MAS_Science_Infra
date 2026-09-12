import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { experimentApi } from '../api';
import { errorMessage } from '../../../shared/api/http';
import type { Bundle } from '../../../shared/api/types';

const PAGE_SIZE = 6;
const CACHE_SIZE = 24;
const CONCURRENCY = 3;

export interface ExperimentSummary {
  id: string;
  name: string;
  topology: string;
  agentCount: number;
  toolCount: number;
  executable: boolean;
}
export interface ExperimentCardState {
  summary?: ExperimentSummary;
  loading: boolean;
  error?: string;
}

function summarize(bundle: Bundle): ExperimentSummary {
  const workflow = bundle.workflow;
  const explicit = Boolean(workflow.agents?.length) || workflow.topology === 'graph';
  const ids = new Set(explicit ? (workflow.agents || []).map(agent => agent.id) : ['hub']);
  if (workflow.hub?.verify && ids.has('hub')) ids.add('verifier');
  return {
    id: bundle.id, name: bundle.meta.name || bundle.id, topology: workflow.topology,
    agentCount: ids.size, toolCount: new Set(workflow.tools || []).size, executable: bundle.executable.ok,
  };
}

export function useExperimentCatalog(active: boolean) {
  const [listing, setListing] = useState<{ ids: string[]; loading: boolean; error: string | null; version: number }>({
    ids: [], loading: true, error: null, version: 0,
  });
  const [filter, setFilter] = useState({ query: '', page: 0 });
  const [reload, setReload] = useState(0);
  const [retry, setRetry] = useState(0);
  const [cards, setCards] = useState<Record<string, ExperimentCardState>>({});
  const cache = useRef(new Map<string, { summary: ExperimentSummary; version: number }>());

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setListing(previous => ({ ...previous, loading: true, error: null }));
    void experimentApi.listExperiments(controller.signal).then(result => {
      if (controller.signal.aborted) return;
      if (!Array.isArray(result.experiments) || result.experiments.some(id => typeof id !== 'string')) {
        throw new Error('实验列表响应格式无效');
      }
      const ids = Array.from(new Set(result.experiments));
      setListing(previous => ({ ids, loading: false, error: null, version: previous.version + 1 }));
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setListing(previous => ({ ...previous, loading: false, error: errorMessage(reason) }));
    });
    return () => controller.abort();
  }, [active, reload]);

  const matches = useMemo(() => {
    const query = filter.query.trim().toLowerCase();
    return query ? listing.ids.filter(id => id.toLowerCase().includes(query)) : listing.ids;
  }, [listing.ids, filter.query]);
  const pageCount = Math.max(1, Math.ceil(matches.length / PAGE_SIZE));
  const page = Math.min(filter.page, pageCount - 1);
  const pageIds = useMemo(() => matches.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE), [matches, page]);

  useEffect(() => {
    if (!active || listing.loading || listing.version === 0) return;
    const controller = new AbortController();
    const initial: Record<string, ExperimentCardState> = {};
    const pending: string[] = [];
    for (const id of pageIds) {
      const cached = cache.current.get(id);
      const fresh = cached?.version === listing.version;
      initial[id] = { summary: cached?.summary, loading: !fresh };
      if (!fresh) pending.push(id);
    }
    setCards(initial);
    let nextIndex = 0;
    const worker = async () => {
      while (!controller.signal.aborted && nextIndex < pending.length) {
        const id = pending[nextIndex++];
        try {
          const bundle = await experimentApi.getExperiment(id, controller.signal);
          if (controller.signal.aborted) return;
          if (bundle.id !== id) throw new Error('返回的实验身份不匹配');
          const summary = summarize(bundle);
          cache.current.delete(id);
          cache.current.set(id, { summary, version: listing.version });
          while (cache.current.size > CACHE_SIZE) {
            const oldest = cache.current.keys().next().value;
            if (oldest !== undefined) cache.current.delete(oldest);
          }
          setCards(previous => ({ ...previous, [id]: { summary, loading: false } }));
        } catch (reason) {
          if (controller.signal.aborted) return;
          setCards(previous => ({
            ...previous, [id]: { ...previous[id], loading: false, error: errorMessage(reason) },
          }));
        }
      }
    };
    for (let index = 0; index < Math.min(CONCURRENCY, pending.length); index++) void worker();
    return () => controller.abort();
  }, [active, listing.loading, listing.version, pageIds, retry]);

  const refresh = useCallback(() => setReload(value => value + 1), []);
  const setQuery = useCallback((query: string) => setFilter({ query, page: 0 }), []);
  const setPage = useCallback((next: number) => setFilter(previous => ({ ...previous, page: Math.max(0, next) })), []);
  const retryCard = useCallback((id: string) => {
    cache.current.delete(id);
    setRetry(value => value + 1);
  }, []);
  const created = useCallback((bundle: Bundle) => {
    const summary = summarize(bundle);
    cache.current.set(bundle.id, { summary, version: -1 });
    setListing(previous => ({
      ...previous, ids: Array.from(new Set([...previous.ids, bundle.id])).sort(),
    }));
    refresh();
  }, [refresh]);

  return {
    ids: listing.ids, loading: listing.loading, error: listing.error, loaded: listing.version > 0,
    cards, pageIds, page, pageCount, total: listing.ids.length, matchedCount: matches.length,
    query: filter.query, setQuery, setPage, refresh, retryCard, created,
  };
}

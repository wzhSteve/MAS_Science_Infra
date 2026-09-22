import { Boxes, FlaskConical, Cpu, Activity, Network, ScanLine, History } from 'lucide-react';
import type { SettingsSection } from '../features/settings/model/sections';

export type ResourceCategory = 'models' | 'datasets';
export interface ResourceRoute {
  kind: 'resources';
  category: ResourceCategory;
  experimentId?: string;
}

export const NAVIGATION = [
  { id: 'experiment', label: 'Experiment', description: '实验配置', icon: FlaskConical },
  { id: 'llm', label: 'LLM', description: '模型连接', icon: Cpu },
  { id: 'mas', label: 'MAS', description: '工作流', icon: Network },
  { id: 'rl', label: 'RL', description: '训练配置', icon: Boxes },
  { id: 'harness', label: 'Harness', description: '诊断', icon: ScanLine },
  { id: 'monitor', label: 'Monitor', description: '监控', icon: Activity },
  { id: 'records', label: 'Runs', description: '训练记录', icon: History },
] as const;
export type PanelId = (typeof NAVIGATION)[number]['id'];

export function isCanvasPanel(panel: PanelId) {
  return panel === 'mas' || panel === 'llm' || panel === 'rl' || panel === 'harness';
}

export interface WorkspaceRoute {
  kind: 'workspace';
  experimentId: string;
  panel: PanelId;
  settings?: SettingsSection;
  selectedResource?: string;
  resourcePurpose?: 'inference' | 'training';
  console?: 'training';
  runId?: string;
}
export type AppRoute = { kind: 'home' } | WorkspaceRoute | ResourceRoute | { kind: 'not-found' };

export const HOME_HASH = '#/experiments';

export function workspaceRoute(experimentId: string, panel: PanelId = 'mas', settings?: SettingsSection): WorkspaceRoute {
  return { kind: 'workspace', experimentId, panel, ...(settings ? { settings } : {}) };
}

export function routeHash(route: Exclude<AppRoute, { kind: 'not-found' }>): string {
  if (route.kind === 'home') return HOME_HASH;
  if (route.kind === 'resources') return `#/resources/${route.category}${route.experimentId ? `?experiment=${encodeURIComponent(route.experimentId)}` : ''}`;
  const base = `${HOME_HASH}/${encodeURIComponent(route.experimentId)}`;
  const params = new URLSearchParams();
  if (route.selectedResource) params.set('select', route.selectedResource);
  if (route.resourcePurpose) params.set('purpose', route.resourcePurpose);
  if (route.console) params.set('console', route.console);
  if (route.runId) params.set('run', route.runId);
  const suffix = params.size ? `?${params}` : '';
  if (route.settings) return `${base}/workspace/settings/${route.settings}${suffix}`;
  return (route.panel === 'mas' ? `${base}/workspace` : `${base}/panels/${route.panel}`) + suffix;
}

function isPanel(value: string): value is PanelId {
  return NAVIGATION.some(panel => panel.id === value);
}

export function parseRoute(hash: string): AppRoute {
  const [rawPath, query = ''] = hash.replace(/^#/, '').split('?');
  const path = rawPath.replace(/\/+$/, '');
  if (path === '/resources/models' || path === '/resources/datasets') {
    const id = new URLSearchParams(query).get('experiment') || undefined;
    if (id && !/^[A-Za-z0-9_-]{1,64}$/.test(id)) return { kind: 'not-found' };
    return { kind: 'resources', category: path.endsWith('/models') ? 'models' : 'datasets', experimentId: id };
  }
  if (!path || path === '/' || path === '/experiments') return { kind: 'home' };
  const legacy = path.replace(/^\//, '');
  if (isPanel(legacy)) return workspaceRoute('demo', legacy);
  const parts = path.split('/');
  if (parts[0] !== '' || parts[1] !== 'experiments') return { kind: 'not-found' };
  let experimentId: string;
  try { experimentId = decodeURIComponent(parts[2] || ''); }
  catch { return { kind: 'not-found' }; }
  if (!experimentId || /[/\\\u0000-\u001f]/.test(experimentId) || experimentId === '.' || experimentId === '..') {
    return { kind: 'not-found' };
  }
  const params = new URLSearchParams(query);
  const consoleState = params.get('console') === 'training' ? {
    console: 'training' as const,
    runId: params.get('run') || undefined,
  } : {};
  if (consoleState.runId && !/^[a-f0-9]{12}$/.test(consoleState.runId)) return { kind: 'not-found' };
  if (parts.length === 4 && parts[3] === 'workspace') return { ...workspaceRoute(experimentId), ...consoleState };
  if (parts.length === 6 && parts[3] === 'workspace' && parts[4] === 'settings') {
    const section = parts[5];
    if (section === 'model' || section === 'data' || section === 'training' || section === 'environment' || section === 'diagnostics' || section === 'inference') {
      const selectedResource = new URLSearchParams(query).get('select') || undefined;
      if (selectedResource && !/^[A-Za-z0-9_-]{1,128}$/.test(selectedResource)) return { kind: 'not-found' };
      return { ...workspaceRoute(experimentId, 'mas', section), ...consoleState,
        resourcePurpose: params.get('purpose') === 'training' ? 'training' : params.get('purpose') === 'inference' ? 'inference' : undefined,
        ...(selectedResource ? { selectedResource } : {}) };
    }
  }
  if (parts.length === 5 && parts[3] === 'panels' && isPanel(parts[4])) return { ...workspaceRoute(experimentId, parts[4]), ...consoleState };
  return { kind: 'not-found' };
}

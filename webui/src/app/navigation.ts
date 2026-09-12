import { Boxes, FlaskConical, Cpu, Activity, Network, ScanLine } from 'lucide-react';

export const NAVIGATION = [
  { id: 'experiment', label: 'Experiment', description: '实验配置', icon: FlaskConical },
  { id: 'llm', label: 'LLM', description: '模型连接', icon: Cpu },
  { id: 'mas', label: 'MAS', description: '工作流', icon: Network },
  { id: 'rl', label: 'RL', description: '训练配置', icon: Boxes },
  { id: 'harness', label: 'Harness', description: '诊断', icon: ScanLine },
  { id: 'monitor', label: 'Monitor', description: '监控', icon: Activity },
] as const;
export type PanelId = (typeof NAVIGATION)[number]['id'];

export function isCanvasPanel(panel: PanelId) {
  return panel === 'mas' || panel === 'llm' || panel === 'rl' || panel === 'harness';
}

export interface WorkspaceRoute {
  kind: 'workspace';
  experimentId: string;
  panel: PanelId;
}
export type AppRoute = { kind: 'home' } | WorkspaceRoute | { kind: 'not-found' };

export const HOME_HASH = '#/experiments';

export function workspaceRoute(experimentId: string, panel: PanelId = 'mas'): WorkspaceRoute {
  return { kind: 'workspace', experimentId, panel };
}

export function routeHash(route: Exclude<AppRoute, { kind: 'not-found' }>): string {
  if (route.kind === 'home') return HOME_HASH;
  const base = `${HOME_HASH}/${encodeURIComponent(route.experimentId)}`;
  return route.panel === 'mas' ? `${base}/workspace` : `${base}/panels/${route.panel}`;
}

function isPanel(value: string): value is PanelId {
  return NAVIGATION.some(panel => panel.id === value);
}

export function parseRoute(hash: string): AppRoute {
  const path = hash.replace(/^#/, '').replace(/\/+$/, '');
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
  if (parts.length === 4 && parts[3] === 'workspace') return workspaceRoute(experimentId);
  if (parts.length === 5 && parts[3] === 'panels' && isPanel(parts[4])) return workspaceRoute(experimentId, parts[4]);
  return { kind: 'not-found' };
}

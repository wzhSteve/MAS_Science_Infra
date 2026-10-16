import { ExternalLink } from 'lucide-react';
import { useAgl } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { Tooltip } from '../../shared/ui/tooltip';

export function AglMetricsLink({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const { data, error } = useAgl();
  const online = Boolean(data?.ok && !error);
  const built = Boolean(data?.dashboard_built);
  const href = (() => {
    const base = data?.ui_path || '/agl/metrics';
    if (typeof window === 'undefined') return base;
    const match = window.location.hash.match(/#\/experiments\/([^/?#]+)/);
    if (!match) return base;
    const url = new URL(base, window.location.origin);
    url.searchParams.set('experiment', decodeURIComponent(match[1]));
    return `${url.pathname}${url.search}`;
  })();
  if (!online && !built) {
    return <Tooltip content={error || '需要已构建的 AGL Dashboard，或训练进程拉起 LightningStore'}>
      <span tabIndex={0}><Button size={size} disabled>Metrics 未就绪</Button></span>
    </Tooltip>;
  }
  const historical = !online;
  return <a
    className={`ui-button ui-button--secondary ui-button--${size}`}
    href={href}
    target="_blank"
    rel="noreferrer"
    title={historical ? 'LightningStore 未在线，打开磁盘上的历史指标' : '打开 Agent-lightning Metrics'}
  >
    {historical ? 'AGL Metrics（历史）' : 'AGL Metrics'}
    <ExternalLink size={13} aria-hidden="true" />
  </a>;
}

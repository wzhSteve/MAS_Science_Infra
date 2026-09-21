import { ExternalLink } from 'lucide-react';
import { useAgl } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { Tooltip } from '../../shared/ui/tooltip';

export function MetricsLink() {
  const { data, error } = useAgl();
  return data?.ok && !error ? <a className="ui-button ui-button--secondary ui-button--sm" href="/agl/metrics" target="_blank" rel="noreferrer">
    AGL Metrics<ExternalLink size={13} aria-hidden="true" />
  </a> : <Tooltip content={error || '训练进程拉起 LightningStore 后可用'}>
    <span tabIndex={0}><Button size="sm" disabled>Metrics 未就绪</Button></span>
  </Tooltip>;
}

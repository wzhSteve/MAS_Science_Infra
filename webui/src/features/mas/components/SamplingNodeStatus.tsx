import { CircleDot, GitBranch, TriangleAlert, XCircle } from 'lucide-react';
import type { SamplingCanvasState } from '../types';

const content = {
  available: { icon: CircleDot, label: '可配置' },
  configured: { icon: GitBranch, label: '已配置' },
  compatibility: { icon: TriangleAlert, label: '兼容位置' },
  unavailable: { icon: XCircle, label: '暂不支持' },
} satisfies Record<SamplingCanvasState, { icon: typeof CircleDot; label: string }>;

export function SamplingNodeStatus({ state, label }: { state?: SamplingCanvasState; label?: string }) {
  if (!state) return null;
  const item = content[state];
  const Icon = item.icon;
  return <div className={`mas-node__sampling is-${state}`} title={label}>
    <Icon size={11} /><span>{item.label}</span>
  </div>;
}

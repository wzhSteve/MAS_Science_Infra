import { memo } from 'react';
import type { TrainProgress } from '../../../shared/api/types';

export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor(whole / 60) % 60;
  const remain = whole % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remain).padStart(2, '0')}`;
}

export const TrainEta = memo(function TrainEta({ progress, running }: {
  progress?: TrainProgress | null;
  running: boolean;
}) {
  if (!running && !progress) return <span className="train-eta">当前没有活动训练</span>;
  if (!progress) return <span className="train-eta">等待第一步</span>;
  const step = `第 ${progress.step}/${progress.total} 步`;
  if (progress.eta_seconds == null) return <span className="train-eta">{step} · 等待第一步</span>;
  if (!running) return <span className="train-eta">{step}</span>;
  return <span className="train-eta">{step} · 预计剩余 {formatClock(progress.eta_seconds)}</span>;
});

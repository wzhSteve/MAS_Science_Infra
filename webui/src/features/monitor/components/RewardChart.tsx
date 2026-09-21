import { lazy, memo, Suspense } from 'react';
import { EmptyState } from '../../../shared/components/EmptyState';
import { LoadingState } from '../../../shared/components/LoadingState';

const RewardChartPlot = lazy(() => import('./RewardChartPlot'));

export interface ChartPoint {
  i: number;
  reward: number;
}

export interface RewardChartProps {
  data: ChartPoint[];
  title: string;
  color: 'var(--chart-collect)' | 'var(--chart-train)';
}

export const RewardChart = memo(function RewardChart({ data, title, color }: RewardChartProps) {
  if (!data.length) return <EmptyState title="暂无数据点。" />;

  return (
    <figure className="m-0 h-[220px] w-full min-w-0" aria-label={title}>
      <figcaption className="sr-only">{title}，横轴 index，纵轴 reward（0–1）。</figcaption>
      <Suspense fallback={<LoadingState label={`正在加载 ${title} 图表…`} />}>
        <RewardChartPlot data={data} color={color} title={title} />
      </Suspense>
    </figure>
  );
});

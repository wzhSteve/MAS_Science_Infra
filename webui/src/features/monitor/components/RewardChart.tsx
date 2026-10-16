import { lazy, memo, Suspense } from 'react';
import { EmptyState } from '../../../shared/components/EmptyState';
import { LoadingState } from '../../../shared/components/LoadingState';

const RewardChartPlot = lazy(() => import('./RewardChartPlot'));

export interface ChartPoint {
  i: number;
  reward?: number;
  val?: number;
  pg_loss?: number;
  entropy_loss?: number;
  kl_loss?: number;
}

export interface ChartSeries {
  key: keyof Omit<ChartPoint, 'i'>;
  name: string;
  color: string;
}

export interface RewardChartProps {
  data: ChartPoint[];
  title: string;
  series: ChartSeries[];
}

export const RewardChart = memo(function RewardChart({ data, title, series }: RewardChartProps) {
  if (!data.length || !series.length) return <EmptyState title="暂无数据点。" />;

  return (
    <figure className="m-0 h-[220px] w-full min-w-0" aria-label={title}>
      <figcaption className="sr-only">{title}，横轴 step，纵轴按数据缩放。</figcaption>
      <Suspense fallback={<LoadingState label={`正在加载 ${title} 图表…`} />}>
        <RewardChartPlot data={data} series={series} title={title} />
      </Suspense>
    </figure>
  );
});

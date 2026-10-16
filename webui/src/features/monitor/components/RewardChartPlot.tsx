import { memo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RewardChartProps } from './RewardChart';

const TICK_STYLE = { fill: 'var(--text-secondary)', fontSize: 12 };
const TOOLTIP_STYLE = {
  backgroundColor: 'var(--surface)',
  borderColor: 'var(--border-subtle)',
  color: 'var(--text-secondary)',
  borderRadius: 6,
};

function formatTick(value: number): string {
  if (!Number.isFinite(value)) return '';
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 10000)) return value.toExponential(1);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function paddedDomain(values: number[]): [number, number] {
  if (!values.length) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    const pad = Math.abs(min) > 1e-6 ? Math.abs(min) * 0.15 : 0.1;
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

export default memo(function RewardChartPlot({ data, series, title }: RewardChartProps) {
  const values = data.flatMap(row => series.map(item => row[item.key]).filter((value): value is number => typeof value === 'number'));
  const [yMin, yMax] = paddedDomain(values);
  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <LineChart data={data} accessibilityLayer aria-label={title}>
        <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="3 3" />
        <XAxis dataKey="i" stroke="var(--border-subtle)" tick={TICK_STYLE} />
        <YAxis domain={[yMin, yMax]} stroke="var(--border-subtle)" tick={TICK_STYLE} width={64} tickFormatter={formatTick} />
        <Tooltip contentStyle={TOOLTIP_STYLE} isAnimationActive={false} />
        <Legend />
        {series.map(item => (
          <Line
            key={item.key}
            type="monotone"
            dataKey={item.key}
            name={item.name}
            stroke={item.color}
            dot={{ r: 3 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
});

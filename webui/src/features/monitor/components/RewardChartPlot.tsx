import { memo } from 'react';
import {
  CartesianGrid,
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

export default memo(function RewardChartPlot({ data, color, title }: RewardChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <LineChart data={data} accessibilityLayer aria-label={title}>
        <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="3 3" />
        <XAxis dataKey="i" stroke="var(--border-subtle)" tick={TICK_STYLE} />
        <YAxis domain={[0, 1]} stroke="var(--border-subtle)" tick={TICK_STYLE} />
        <Tooltip contentStyle={TOOLTIP_STYLE} isAnimationActive={false} />
        <Line type="monotone" dataKey="reward" stroke={color} dot isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
});

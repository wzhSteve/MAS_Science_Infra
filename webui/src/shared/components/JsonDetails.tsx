import { memo, useMemo } from 'react';

export const JsonDetails = memo(function JsonDetails({ value, label = '查看完整数据', open = false }: {
  value: unknown; label?: string; open?: boolean;
}) {
  const text = useMemo(() => JSON.stringify(value, null, 2), [value]);
  return <details className="json-details" open={open || undefined}><summary>{label}</summary><pre>{text}</pre></details>;
});

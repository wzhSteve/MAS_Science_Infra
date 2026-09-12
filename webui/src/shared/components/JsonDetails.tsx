import { memo, useMemo } from 'react';
import { InlineNotice } from './InlineNotice';
import { LazyDetails } from './LazyDetails';
import { TextContent } from './TextContent';

const JsonContent = memo(function JsonContent({ value, label }: { value: unknown; label: string }) {
  const result = useMemo(() => {
    try {
      return { text: JSON.stringify(value, null, 2) ?? '', error: false };
    } catch {
      return { text: '', error: true };
    }
  }, [value]);
  return result.error ? <InlineNotice tone="warning">无法将此数据序列化为 JSON。</InlineNotice>
    : <TextContent text={result.text} label={label} />;
});

export const JsonDetails = memo(function JsonDetails({ value, label = '查看完整数据', open = false }: {
  value: unknown; label?: string; open?: boolean;
}) {
  return <LazyDetails className="json-details" summary={label} open={open}>
    <JsonContent value={value} label={label} />
  </LazyDetails>;
});

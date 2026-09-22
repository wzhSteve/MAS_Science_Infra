import { memo } from 'react';
import type { RlConfig } from '../../../shared/api/types';
import type { HydraField } from '../model/hydraFields';
import { FormField } from '../../../shared/components/FormField';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';

export const RlField = memo(function RlField({ field, value = '', onPatch, compact = false }: {
  field: HydraField; value?: string | number; onPatch: (update: (current: RlConfig) => RlConfig) => void;
  compact?: boolean;
}) {
  const update = (value: string) => onPatch(current => field.update(current, value));
  return <FormField label={<span title={field.description}>{field.label}</span>} hint={compact ? undefined : field.description}>
    {field.path === 'data.truncation' ? <Select value={value} onChange={event => update(event.target.value)}>
      <option value="">继承默认值</option>
      {!['', 'error', 'left', 'right'].includes(String(value)) && <option value={value}>{value}</option>}
      <option value="error">error</option><option value="left">left</option><option value="right">right</option>
    </Select> : <Input type={field.type} step={field.type === 'number' ? 'any' : undefined}
      value={value} placeholder="继承默认值" onChange={event => update(event.target.value)} />}
  </FormField>;
});

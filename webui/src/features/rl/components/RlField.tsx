import { memo } from 'react';
import type { RlConfig } from '../../../shared/api/types';
import { HYDRA_FIELDS, type HydraField } from '../model/hydraFields';
import { FormField } from '../../../shared/components/FormField';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';

export const RlField = memo(function RlField({ field, value = '', onPatch, compact = false }: {
  field: HydraField; value?: string | number | boolean; onPatch: (update: (current: RlConfig) => RlConfig) => void;
  compact?: boolean;
}) {
  const update = (value: string) => onPatch(current => field.update(current, value));
  const hint = typeof field.hint === 'function' ? undefined : field.hint;
  return <FormField label={<span title={field.description}>{field.label}</span>} hint={compact ? hint || undefined : [field.description, hint].filter(Boolean).join(' ') || undefined}>
    {field.type === 'toggle'
      ? <input type="checkbox"
          aria-label={field.label}
          checked={value === true}
          title={field.description}
          onChange={event => update(String(event.target.checked))} />
      : field.type === 'select'
        ? <Select value={String(value ?? '')} onChange={event => update(event.target.value)}>
            <option value="">继承默认值</option>
            {(field.options ?? []).map(option => <option key={option} value={option}>{option}</option>)}
          </Select>
        : <Input type={field.type === 'number' ? 'number' : 'text'} step={field.type === 'number' ? 'any' : undefined}
            value={String(value ?? '')} placeholder="继承默认值" onChange={event => update(event.target.value)} />}
  </FormField>;
});

/** Filter helper: apply the field's declared visibility (RLDesign §4.2 联动 7). */
export function visibleFields(rl: RlConfig, items: typeof HYDRA_FIELDS = HYDRA_FIELDS) {
  return items.filter(field => !field.visible || field.visible(rl));
}

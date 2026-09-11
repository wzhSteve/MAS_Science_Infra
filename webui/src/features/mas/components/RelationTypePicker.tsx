import { useId } from 'react';
import { Check } from 'lucide-react';
import type { EdgeKind } from '../types';
import type { RelationOption } from '../model/edgeRules';
import { edgeDefinition } from '../model/edgeDefinitions';

export function RelationTypePicker({ options, value, onChange }: {
  options: RelationOption[];
  value?: string;
  onChange: (kind: EdgeKind) => void;
}) {
  const name = useId();
  return <fieldset className="mas-relation-options">
    <legend className="sr-only">协作方式</legend>
    {options.map(({ kind, reason }) => {
      const definition = edgeDefinition(kind);
      const Icon = definition.icon;
      const selected = value === kind;
      return <label key={kind} className={`mas-relation-option${selected ? ' is-selected' : ''}${reason ? ' is-disabled' : ''}`}>
        <input type="radio" name={name} value={kind} checked={selected} disabled={Boolean(reason)}
          aria-describedby={`${name}-${kind}`} onChange={() => onChange(kind)} />
        <span className="mas-relation-icon" style={{ color: definition.color }}><Icon size={17} aria-hidden="true" /></span>
        <span className="mas-relation-copy"><strong>{definition.title}</strong>
          <span id={`${name}-${kind}`}>{reason || definition.description}</span></span>
        {selected && <Check size={14} className="mas-relation-check" aria-hidden="true" />}
      </label>;
    })}
  </fieldset>;
}

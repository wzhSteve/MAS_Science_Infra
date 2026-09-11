import { memo, useMemo, useState } from 'react';
import type { Hypothesis, TrainSignal } from '../../../shared/api/types';
import { Select } from '../../../shared/ui/select';
import { DataTable } from '../../../shared/components/DataTable';
import { EmptyState } from '../../../shared/components/EmptyState';
import { FormField } from '../../../shared/components/FormField';
import { JsonDetails } from '../../../shared/components/JsonDetails';
import { Section } from '../../../shared/components/Section';

interface HarnessDiagnosticsProps {
  hypotheses: Hypothesis[];
  trainSignal?: TrainSignal;
}

export const HarnessDiagnostics = memo(function HarnessDiagnostics({ hypotheses, trainSignal }: HarnessDiagnosticsProps) {
  const [plugin, setPlugin] = useState('');
  const plugins = useMemo(() => [...new Set(hypotheses.map((hypothesis) => hypothesis.plugin))], [hypotheses]);
  const filtered = useMemo(
    () => plugin ? hypotheses.filter((hypothesis) => hypothesis.plugin === plugin) : hypotheses,
    [hypotheses, plugin],
  );

  return (
    <Section
      title="Harness"
      actions={
        <FormField label="plugin 筛选" className="w-52 max-w-full">
          <Select value={plugin} onChange={(event) => setPlugin(event.target.value)}>
            <option value="">(all)</option>
            {plugin && !plugins.includes(plugin) ? <option value={plugin}>{plugin}</option> : null}
            {plugins.map((name) => <option key={name} value={name}>{name}</option>)}
          </Select>
        </FormField>
      }
    >
      <DataTable aria-label="Harness hypotheses">
        <thead><tr><th scope="col">plugin</th><th scope="col">message</th></tr></thead>
        <tbody>
          {filtered.map((hypothesis, index) => (
            <tr key={`${hypothesis.plugin}-${hypothesis.event_id ?? index}`}>
              <td className="mono break-all">{hypothesis.plugin}</td>
              <td className="whitespace-pre-wrap break-words">{hypothesis.message}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      {!filtered.length ? <EmptyState title="(no hypotheses)" /> : null}
      {trainSignal ? (
        <div className="mt-3 grid gap-2">
          <JsonDetails label="TrainSignal advantage" value={trainSignal.advantage} open />
          <JsonDetails label="TrainSignal loss" value={trainSignal.loss} open />
        </div>
      ) : null}
    </Section>
  );
});

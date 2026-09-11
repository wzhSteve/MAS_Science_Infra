import { memo, useState } from 'react';
import type { CollectBody, CollectRow } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { Section } from '../../../shared/components/Section';
import { DataTable } from '../../../shared/components/DataTable';

type CollectAction = (name: string, body: CollectBody) => Promise<void>;

export const WorkflowActions = memo(function WorkflowActions({ pending, executable, onSave, onCollect, algo, onAlgoChange }: {
  pending: string | null;
  executable: boolean;
  onSave: () => Promise<void>;
  onCollect: CollectAction;
  algo: string;
  onAlgoChange: (value: string) => void;
}) {
  const [n, setN] = useState(1);
  return <ActionBar className="mas-actions">
    <Button variant="primary" onClick={onSave} disabled={Boolean(pending)} loading={pending === 'save'}>
      保存 workflow.yaml
    </Button>
    <FormField label="采集条数"><Input type="number" min={1} value={n} className="w-20"
      onChange={(event) => setN(Number(event.target.value))} /></FormField>
    <FormField label="algo"><Input value={algo} className="w-24" onChange={(event) => onAlgoChange(event.target.value)} /></FormField>
    <Button disabled={Boolean(pending) || !executable} loading={pending === 'mock'}
      onClick={() => onCollect('mock', { mock: true, n, algo })}>Collect (mock)</Button>
    <Button disabled={Boolean(pending) || !executable} loading={pending === 'live'}
      onClick={() => onCollect('live', { mock: false, n, algo })}>Collect (live)</Button>
    <span className="field-hint">采集前保存当前 workflow</span>
  </ActionBar>;
});

export const ParquetCollect = memo(function ParquetCollect({ pending, executable, algo, onCollect, onPreview, rows }: {
  pending: string | null;
  executable: boolean;
  algo: string;
  onCollect: CollectAction;
  onPreview: (body: { parquet: string; data_n: number; source: string }) => Promise<void>;
  rows: CollectRow[];
}) {
  const [parquet, setParquet] = useState('data/val.parquet');
  const [dataN, setDataN] = useState(5);
  const [source, setSource] = useState('gsm8k');
  return <Section title="live-api-data" description={<>parquet → TirAgent · 对齐 <code>./run.sh live-api-data</code>：LLM → tool → answer → reward。</>}>
    <div className="form-grid">
      <FormField label="parquet"><Input value={parquet} onChange={(event) => setParquet(event.target.value)} className="mono" /></FormField>
      <FormField label="data_n"><Input type="number" min={1} value={dataN} onChange={(event) => setDataN(Number(event.target.value))} /></FormField>
      <FormField label="source"><Input value={source} onChange={(event) => setSource(event.target.value)} /></FormField>
    </div>
    <ActionBar className="mt-4">
      <Button variant="primary" disabled={Boolean(pending) || !executable} loading={pending === 'parquet'}
        onClick={() => onCollect('parquet', { mock: false, n: 1, algo, parquet, data_n: dataN, source, sequential: true })}>
        Collect from parquet (live)
      </Button>
      <Button disabled={Boolean(pending)} loading={pending === 'preview'}
        onClick={() => onPreview({ parquet, data_n: dataN, source })}>仅预览抽样</Button>
    </ActionBar>
    {rows.length > 0 && <div className="mt-4">
      <DataTable aria-label="采集结果">
        <thead><tr><th>id</th><th>answer</th><th>reward</th><th>tool</th></tr></thead>
        <tbody>{rows.map((row, index) => <tr key={`${row.id}-${index}`}>
          <td className="mono">{row.id}</td>
          <td>{String(row.answer ?? '')}</td>
          <td className="mono">{row.reward ?? ''}</td>
          <td>{row.tool ? '✓' : '✗'}</td>
        </tr>)}</tbody>
      </DataTable>
    </div>}
  </Section>;
});

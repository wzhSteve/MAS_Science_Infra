import { memo } from 'react';
import type { Palette, WorkflowSpec } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';
import { Select } from '../../../shared/ui/select';
import { FormField } from '../../../shared/components/FormField';
import { StatusBadge } from '../../../shared/components/StatusBadge';

export const GraphToolbar = memo(function GraphToolbar({ palette, edgeKind, onEdgeKindChange, onTemplate, executable, entryId }: {
  palette: Palette;
  edgeKind: string;
  onEdgeKindChange: (kind: string) => void;
  onTemplate: (workflow: WorkflowSpec) => void;
  executable: { ok: boolean; reason: string };
  entryId: string;
}) {
  return <div className="mas-toolbar">
    <div className="flex flex-wrap items-center gap-2" aria-label="Workflow 模板">
      <span className="field-hint">模板</span>
      {(palette.templates || []).map((template) =>
        <Button key={template.id} size="sm" disabled={!template.workflow}
          onClick={() => template.workflow && onTemplate(template.workflow)}>
          {template.label}
        </Button>)}
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <FormField label="连线 kind" className="mas-edge-kind">
        <Select value={edgeKind} onChange={(event) => onEdgeKindChange(event.target.value)}>
          {(palette.edge_kinds || ['message', 'route', 'feedback', 'tool_call']).map((kind) =>
            <option key={kind} value={kind}>{kind}</option>)}
        </Select>
      </FormField>
      <StatusBadge tone={executable.ok ? 'success' : 'danger'}>
        {executable.ok ? 'executable' : 'not executable'} · {executable.reason}
      </StatusBadge>
      <StatusBadge tone="info">入口 · {entryId}</StatusBadge>
    </div>
  </div>;
});

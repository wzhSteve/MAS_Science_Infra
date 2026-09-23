import { memo, useState } from 'react';
import type { Bundle } from '../../../shared/api/types';
import { api } from '../../../api/client';
import { ModelBinding, type BindingView } from '../../resources/components/ModelBinding';
import { Section } from '../../../shared/components/Section';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';

export const InferenceSettings = memo(function InferenceSettings({ bundle, active, view, onReload, onManage, suggestedId, onSuggestionApplied, compact = false }: {
  bundle: Bundle; active: boolean; view: 'connection' | 'environment' | 'all'; onReload: () => void;
  onManage: () => void; suggestedId?: string; onSuggestionApplied?: () => void;
  compact?: boolean;
}) {
  const [binding, setBinding] = useState<BindingView>({ loaded: false, bound: false, resource: null, error: null });
  const action = useAction();
  const config = binding.bound ? binding.resource?.config : bundle.llm;
  const environment = view !== 'connection';
  useUnsavedChanges('bound-inference-operation', { label: '模型服务操作', resource: 'model-service', dirty: false, busy: action.pending !== null });
  return <div className="page-stack settings-embedded">
    <div hidden={view === 'environment'}>
      <ModelBinding experimentId={bundle.id} purpose="inference" active={active} onReload={onReload} onManage={onManage} saveInHeader
        compact={compact} fallbackName={bundle.llm.model}
        onState={setBinding} suggestedId={suggestedId} onSuggestionApplied={onSuggestionApplied} />
    </div>
    {environment && binding.error && <InlineNotice tone="danger">{binding.error}</InlineNotice>}
    {environment && binding.loaded && !binding.error && config?.kind === 'local' && <Section title="本地模型服务">
      <p className="field-hint">{binding.resource?.name || config.model || '本地模型'} · 端口 {config.port || 8000}</p>
      <div className="action-bar">
        <Button size="sm" disabled={action.pending !== null} loading={action.pending === 'start'} onClick={() => {
          void action.run('start', async () => {
            const result = await api.llmStart(bundle.id);
            onReload();
            return `本地服务启动请求已提交：${result.run_id}`;
          });
        }}>启动本地模型</Button>
        <Button size="sm" variant="danger" disabled={action.pending !== null} loading={action.pending === 'stop'} onClick={() => {
          void action.run('stop', async () => { await api.llmStop(bundle.id); return '已请求停止本地模型服务'; });
        }}>停止本地模型</Button>
      </div>
    </Section>}
  </div>;
});

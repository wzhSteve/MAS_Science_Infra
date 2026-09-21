import { memo, useCallback, useState, type ReactNode } from 'react';
import type { Bundle, MetaResponse } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { RlQuickSettings } from '../../rl/components/RlQuickSettings';
import { RlSettingsForm } from '../../rl/components/RlSettingsForm';
import { HYDRA_FIELDS } from '../../rl/model/hydraFields';
import { ModelBinding, type BindingView } from '../../resources/components/ModelBinding';
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';

export type TrainingSettingsProps = {
  expId: string;
  bundle: Bundle;
  meta: MetaResponse | null;
  onReload: () => void;
  section: 'training' | 'data' | 'environment';
  active: boolean;
  renderRuntime: (props: { onStart: () => void; pending: string | null; active: boolean }) => ReactNode;
  onManageModels: () => void;
  suggestedResourceId?: string;
  onSuggestionApplied?: () => void;
};

const DATA_FIELDS = HYDRA_FIELDS.filter((field) => field.group === 'Data');
const ENVIRONMENT_FIELDS = HYDRA_FIELDS.filter((field) => field.group === 'Rollout');
const PROFILE_LABELS: Record<string, string> = {
  fast: '单卡 / 快速验证',
  a800: 'A800 单卡',
  a800_2gpu: 'A800 双卡',
};

export const TrainingSettings = memo(function TrainingSettings({
  expId, bundle, meta, onReload, section, active, renderRuntime, onManageModels, suggestedResourceId, onSuggestionApplied,
}: TrainingSettingsProps) {
  const { draft, gpu } = useTrainingConfig();
  const { rl, patch, pending } = draft;
  const [mode, setMode] = useState<'quick' | 'full'>('quick');
  const [binding, setBinding] = useState<BindingView>({ loaded: false, bound: false, resource: null, error: null });
  const trainingVisible = active && section === 'training';
  const save = useCallback(() => { void draft.save(); }, [draft.save]);
  const canRecommend = Boolean(gpu.data?.gpus.length);

  return <div className="experiment-settings-training" hidden={!active}>
    {(draft.dirty || pending) && <div role="status" aria-live="polite">
      <StatusBadge tone={draft.dirty ? 'warning' : 'neutral'}>{pending ? '训练配置处理中…' : '训练配置未保存'}</StatusBadge>
    </div>}
    {draft.notice && <InlineNotice tone={draft.notice.tone}>{draft.notice.message}</InlineNotice>}
    {section !== 'data' && <div>
      <ModelBinding experimentId={expId} purpose="training" active={active} onReload={onReload}
        onManage={onManageModels} onState={setBinding} legacyDirty={draft.dirty || pending !== null} suggestedId={suggestedResourceId} onSuggestionApplied={onSuggestionApplied} />
    </div>}

    {section === 'training' && <div>
      <ActionBar>
        <div className="experiment-settings-view-switch" role="group" aria-label="训练参数模式">
          <Button size="sm" variant={mode === 'quick' ? 'primary' : 'ghost'} aria-pressed={mode === 'quick'} onClick={() => setMode('quick')}>基础设置</Button>
          <Button size="sm" variant={mode === 'full' ? 'primary' : 'ghost'} aria-pressed={mode === 'full'} onClick={() => setMode('full')}>高级设置</Button>
        </div>
      </ActionBar>
      <div hidden={mode !== 'quick'}>
        <Section title="训练方案">
          <RlQuickSettings rl={rl} meta={meta} sampling={bundle.workflow.sampling}
            onPatch={(next) => patch(next)} onSave={save} saving={pending !== null} />
          <ActionBar>
            <Button disabled={pending !== null || !canRecommend} loading={pending === 'recommend'} onClick={draft.recommend}>按当前机器推荐</Button>
            <Button disabled={pending !== null} loading={pending === 'prefill'} onClick={draft.prefill}>应用最近采样建议</Button>
          </ActionBar>
        </Section>
      </div>
      <div hidden={mode !== 'full'}>
        <RlSettingsForm rl={rl} meta={meta} dirty={draft.dirty} pending={pending} onPatch={patch}
          onSave={save} onRecommend={draft.recommend} onPrefill={draft.prefill} showSaveAction={false}
          modelBound sampling={bundle.workflow.sampling} advancedOnly />
      </div>
      {section === 'training' && renderRuntime({
        onStart: draft.start, pending: !binding.loaded || binding.error ? 'binding-unavailable' : pending, active: trainingVisible,
      })}
    </div>}

    {section === 'data' && <div>
      <Section title="训练数据">
        <div className="form-grid">
          {DATA_FIELDS.map((field) => <FormField key={field.path}
            label={<HelpLabel help={field.description}>{field.label}</HelpLabel>}>
            <Input type={field.type} step={field.type === 'number' ? 'any' : undefined}
              value={field.read(rl) ?? ''} onChange={(event) => patch(field.update(rl, event.target.value))} />
          </FormField>)}
        </div>
      </Section>
    </div>}

    {section === 'environment' && <div>
      <Section title="训练环境">
        <div className="form-grid">
          <FormField label={<HelpLabel help="根据服务器规模选择启动 Preset；实际设备以当前 GPU 选择为准。">运行档位</HelpLabel>}>
            <Select value={rl.profile || 'fast'} onChange={(event) => patch({ ...rl, profile: event.target.value })}>
              {(meta?.profiles || ['fast']).map((profile) =>
                <option key={profile} value={profile}>{PROFILE_LABELS[profile] || profile}</option>)}
            </Select>
          </FormField>
          <FormField label={<HelpLabel help="同时执行 Rollout 采集的 Runner 数，不等于 GPU 数量。">并行采集数</HelpLabel>}>
            <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => patch({ ...rl, n_runners: Number(event.target.value) })} />
          </FormField>
          {ENVIRONMENT_FIELDS.map((field) => <FormField key={field.path}
            label={<HelpLabel help={field.description}>{field.label}</HelpLabel>}>
            <Input type={field.type} step={field.type === 'number' ? 'any' : undefined}
              value={field.read(rl) ?? ''} onChange={(event) => patch(field.update(rl, event.target.value))} />
          </FormField>)}
        </div>
      </Section>
    </div>}

    {(section !== 'training' || mode === 'full') && <ActionBar>
      <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={save}>保存训练配置</Button>
    </ActionBar>}
  </div>;
});

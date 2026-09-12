import { memo, useCallback, useEffect, useState, type ReactNode } from 'react';
import type { Bundle, MetaResponse, RlConfig } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import GpuPicker from '../../gpu/GpuPicker';
import { RlQuickSettings } from '../../rl/components/RlQuickSettings';
import { RlSettingsForm } from '../../rl/components/RlSettingsForm';
import { HYDRA_FIELDS } from '../../rl/model/hydraFields';
import { useRlDraft } from '../../rl/model/useRlDraft';

export type TrainingSettingsProps = {
  expId: string;
  bundle: Bundle;
  meta: MetaResponse | null;
  onReload: () => void;
  onStartTrain: () => Promise<void>;
  section: 'training' | 'data' | 'environment';
  active: boolean;
  renderRuntime: (props: { onStart: () => void; pending: string | null; active: boolean }) => ReactNode;
};

const DATA_FIELDS = HYDRA_FIELDS.filter((field) => field.group === 'Data');

export const TrainingSettings = memo(function TrainingSettings({
  expId, bundle, meta, onReload, onStartTrain, section, active, renderRuntime,
}: TrainingSettingsProps) {
  const draft = useRlDraft(expId, bundle.rl, onReload, onStartTrain);
  const { rl, patch, pending } = draft;
  const [mode, setMode] = useState<'quick' | 'full'>('quick');
  const environmentVisible = active && section === 'environment';
  const trainingVisible = active && section === 'training';
  const [environmentVisited, setEnvironmentVisited] = useState(environmentVisible);
  useEffect(() => {
    if (environmentVisible) setEnvironmentVisited(true);
  }, [environmentVisible]);
  const patchQuick = useCallback((next: Partial<RlConfig>) => patch((current) => ({ ...current, ...next })), [patch]);
  const save = useCallback(() => { void draft.save(); }, [draft.save]);
  const selectGpus = useCallback((ids: number[]) => patch((current) => ({
    ...current, devices: { ...current.devices, ids },
    trainer: { ...current.trainer, n_gpus_per_node: ids.length },
  })), [patch]);

  return <div className="experiment-settings-training" hidden={!active}>
    <div className="experiment-settings-status" role="status" aria-live="polite">
      <StatusBadge tone={draft.dirty ? 'warning' : 'neutral'}>{draft.dirty ? '训练配置未保存' : '已载入训练配置'}</StatusBadge>
      {pending && <span>训练配置操作中…</span>}
    </div>
    <p className="field-hint">训练、数据与训练环境共用同一份 rl.yaml 草稿；保存会应用三个分组的全部修改。</p>
    {draft.notice && <InlineNotice tone={draft.notice.tone}>{draft.notice.message}</InlineNotice>}

    <div hidden={section !== 'training'}>
      <ActionBar>
        <div className="experiment-settings-view-switch" role="group" aria-label="训练参数模式">
          <Button size="sm" variant={mode === 'quick' ? 'primary' : 'ghost'} aria-pressed={mode === 'quick'} onClick={() => setMode('quick')}>常用</Button>
          <Button size="sm" variant={mode === 'full' ? 'primary' : 'ghost'} aria-pressed={mode === 'full'} onClick={() => setMode('full')}>全部参数</Button>
        </div>
      </ActionBar>
      <div hidden={mode !== 'quick'}>
        <Section title="常用训练参数">
          <RlQuickSettings rl={rl} onPatch={patchQuick} onSave={save} saving={pending !== null} />
          <ActionBar>
            <Button disabled={pending !== null} loading={pending === 'recommend'} onClick={draft.recommend}>按当前机器推荐</Button>
            <Button disabled={pending !== null} loading={pending === 'prefill'} onClick={draft.prefill}>从 TrainSignal 预填</Button>
          </ActionBar>
          <p className="field-hint">推荐与预填只更新当前草稿，需手动保存。</p>
        </Section>
      </div>
      <div hidden={mode !== 'full'}>
        <RlSettingsForm rl={rl} meta={meta} dirty={draft.dirty} pending={pending} onPatch={patch}
          onSave={save} onRecommend={draft.recommend} onPrefill={draft.prefill} initiallyAdvanced showSaveAction={false} />
      </div>
      {section === 'training' && renderRuntime({ onStart: draft.start, pending, active: trainingVisible })}
    </div>

    <div hidden={section !== 'data'}>
      <Section title="训练数据">
        <p className="section-description">以下为现有 Hydra Data 训练字段，仅配置训练批次，不启动独立数据集采集，也不选择或注册数据集。</p>
        <div className="form-grid">
          {DATA_FIELDS.map((field) => <FormField key={field.path} label={field.label} hint={field.path}>
            <Input type={field.type} step={field.type === 'number' ? 'any' : undefined}
              value={field.read(rl) ?? ''} onChange={(event) => patch(field.update(rl, event.target.value))} />
          </FormField>)}
        </div>
      </Section>
    </div>

    <div hidden={section !== 'environment'}>
      <Section title="训练环境">
        <p className="section-description">GPU 选择只修改当前训练草稿，不立即保存或启动服务。保存后应用于训练，不改变调试模型连接。</p>
        {(environmentVisible || environmentVisited) && <GpuPicker selected={rl.devices?.ids || [0]} onChange={selectGpus} />}
        <p className="field-hint">trainer.n_gpus_per_node = {(rl.devices?.ids || [0]).length}（由所选 GPU 数量同步）。Train 与本地 vLLM 互斥。</p>
        <div className="form-grid">
          <FormField label="profile">
            <Select value={rl.profile || 'fast'} onChange={(event) => patch({ ...rl, profile: event.target.value })}>
              {(meta?.profiles || ['fast']).map((profile) => <option key={profile} value={profile}>{profile}</option>)}
            </Select>
          </FormField>
          <FormField label="训练模型路径（model_path）">
            <Input value={rl.model_path || ''} onChange={(event) => patch({ ...rl, model_path: event.target.value })} />
          </FormField>
          <FormField label="并行采集进程（n_runners）">
            <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => patch({ ...rl, n_runners: Number(event.target.value) })} />
          </FormField>
        </div>
      </Section>
    </div>

    {(section !== 'training' || mode === 'full') && <ActionBar>
      <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={save}>保存训练配置</Button>
    </ActionBar>}
  </div>;
});

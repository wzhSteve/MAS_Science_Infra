import { memo, useId, useState } from 'react';
import type { MetaResponse, RlConfig } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { HYDRA_FIELDS } from '../model/hydraFields';
import { rolloutPatch } from '../model/rlPrefills';

type Props = {
  rl: RlConfig;
  meta: MetaResponse | null;
  dirty: boolean;
  pending: string | null;
  onPatch: (next: RlConfig) => void;
  onSave: () => void;
  onRecommend: () => void;
  onPrefill: () => void;
  initiallyAdvanced?: boolean;
  showSaveAction?: boolean;
  modelBound?: boolean;
};

const HYDRA_GROUPS = ['Actor', 'Rollout', 'Data', 'Trainer'];

export const RlSettingsForm = memo(function RlSettingsForm({ rl, meta, dirty, pending, onPatch, onSave, onRecommend, onPrefill, initiallyAdvanced = false, showSaveAction = true, modelBound = false }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(initiallyAdvanced);
  const advancedId = useId();
  return <Section title="训练参数" actions={<StatusBadge tone={dirty ? 'warning' : 'neutral'}>{dirty ? '未保存' : '已载入配置'}</StatusBadge>}>
    <p className="section-description">占用卡数 = 已选 GPU 数（{(rl.devices?.ids || [0]).length}）。Train 与本地 vLLM 互斥。每题采样 ≠ 画布上的采集入口。</p>
    <div className="form-grid">
      <FormField label="algo">
        <Select value={rl.algo || 'grpo'} onChange={(event) => onPatch({ ...rl, algo: event.target.value })}>
          {(meta?.algos || ['grpo']).map((algo) => <option key={algo} value={algo}>{algo}</option>)}
        </Select>
      </FormField>
      <FormField label="profile">
        <Select value={rl.profile || 'fast'} onChange={(event) => onPatch({ ...rl, profile: event.target.value })}>
          {(meta?.profiles || ['fast']).map((profile) => <option key={profile} value={profile}>{profile}</option>)}
        </Select>
      </FormField>
      <FormField label="每题采样条数（GRPO 组大小 / rollout.n）">
        <Input type="number" min={1} value={rl.rollout_per_gpu ?? rl.actor_rollout_ref?.rollout?.n ?? 2} onChange={(event) => onPatch({ ...rl, ...rolloutPatch(rl, Number(event.target.value)) })} />
      </FormField>
      <FormField label="并行采集进程（n_runners）">
        <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => onPatch({ ...rl, n_runners: Number(event.target.value) })} />
      </FormField>
      {!modelBound && <FormField label="model_path"><Input value={rl.model_path || ''} onChange={(event) => onPatch({ ...rl, model_path: event.target.value })} /></FormField>}
    </div>
    <ActionBar>
      <Button loading={pending === 'recommend'} disabled={pending !== null} onClick={onRecommend}>按当前机器推荐</Button>
      <Button loading={pending === 'prefill'} disabled={pending !== null} onClick={onPrefill}>从 TrainSignal 预填</Button>
      <Button variant="ghost" aria-expanded={showAdvanced} aria-controls={advancedId} onClick={() => setShowAdvanced((current) => !current)}>
        {showAdvanced ? '收起高级' : '高级 Hydra 字段'}
      </Button>
    </ActionBar>
    <p className="field-hint">推荐与预填只更新当前草稿，需手动保存。</p>
    {showAdvanced && <div id={advancedId} className="grid gap-5 border-t border-[var(--border-subtle)] pt-4">
      {HYDRA_GROUPS.map((group) => <fieldset key={group} className="m-0 min-w-0 border-0 p-0">
        <legend className="mb-3 text-sm font-semibold">{group}</legend>
        <div className="form-grid">
          {HYDRA_FIELDS.filter((field) => field.group === group).map((field) => <FormField label={field.label} hint={field.path} key={field.path}>
            <Input type={field.type} step={field.type === 'number' ? 'any' : undefined} value={field.read(rl) ?? ''} onChange={(event) => onPatch(field.update(rl, event.target.value))} />
          </FormField>)}
        </div>
      </fieldset>)}
    </div>}
    {showSaveAction && <ActionBar><Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={onSave}>保存超参</Button></ActionBar>}
  </Section>;
});

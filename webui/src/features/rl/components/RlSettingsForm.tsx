import { memo, useId, useState } from 'react';
import type { MetaResponse, RlConfig, SamplingSpec } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { HYDRA_FIELDS, HYDRA_GROUPS } from '../model/hydraFields';
import { algorithmFromSampling } from '../model/normalizeRlPayload';
import { rolloutPatch } from '../model/rlPrefills';

type Props = {
  rl: RlConfig;
  meta: MetaResponse | null;
  dirty: boolean;
  pending: string | null;
  onPatch: (next: RlConfig) => void;
  onSave: () => void;
  onRecommend: () => void;
  showSaveAction?: boolean;
  modelBound?: boolean;
  sampling?: SamplingSpec;
  advancedOnly?: boolean;
};

export const RlSettingsForm = memo(function RlSettingsForm({
  rl, meta, dirty, pending, onPatch, onSave, onRecommend,
  showSaveAction = true, modelBound = false, sampling, advancedOnly = false,
}: Props) {
  const [showAdvanced, setShowAdvanced] = useState(advancedOnly);
  const advancedId = useId();
  const samplingAlgorithm = algorithmFromSampling(sampling);
  const algorithm = samplingAlgorithm || rl.algo || 'grpo';
  const samplingGroup = sampling?.group_n;
  const rolloutCount = samplingGroup ?? rl.rollout_per_gpu ?? rl.actor_rollout_ref?.rollout?.n ?? 2;
  return <Section title={advancedOnly ? '高级训练设置' : '训练参数'}
    actions={<StatusBadge tone={dirty ? 'warning' : 'neutral'}>{dirty ? '未保存' : '已载入配置'}</StatusBadge>}>
    {!advancedOnly && <div className="form-grid">
      <FormField label={<HelpLabel help={samplingAlgorithm
        ? '当前值由画布中的采样策略决定。'
        : '选择训练使用的 Advantage 与 Branch 策略。'}>Algorithm</HelpLabel>}>
        <Select value={algorithm} disabled={Boolean(samplingAlgorithm)}
          onChange={(event) => onPatch({ ...rl, algo: event.target.value })}>
          {(meta?.algos || ['grpo']).map((algo) => <option key={algo} value={algo}>{algo.toUpperCase()}</option>)}
        </Select>
      </FormField>
      <FormField label={<HelpLabel help="根据机器规模选择训练启动 Preset。">运行档位</HelpLabel>}>
        <Select value={rl.profile || 'fast'} onChange={(event) => onPatch({ ...rl, profile: event.target.value })}>
          {(meta?.profiles || ['fast']).map((profile) => <option key={profile} value={profile}>{profile}</option>)}
        </Select>
      </FormField>
      <FormField label={<HelpLabel help={samplingGroup != null
        ? '当前值由画布中的采样策略决定。'
        : '每道题生成多少条候选 Rollout，用于组内比较。'}>group_n</HelpLabel>}>
        <Input type="number" min={1} value={rolloutCount} disabled={samplingGroup != null}
          onChange={(event) => onPatch({ ...rl, ...rolloutPatch(rl, Number(event.target.value)) })} />
      </FormField>
      <FormField label={<HelpLabel help="同时执行 Rollout 采集的 Runner 数。">并行采集数</HelpLabel>}>
        <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => onPatch({ ...rl, n_runners: Number(event.target.value) })} />
      </FormField>
      {!modelBound && <FormField label={<HelpLabel help="没有绑定训练模型资源时使用的本地模型目录。">模型路径</HelpLabel>}>
        <Input value={rl.model_path || ''} onChange={(event) => onPatch({ ...rl, model_path: event.target.value })} />
      </FormField>}
    </div>}
    {!advancedOnly && <ActionBar>
      <Button loading={pending === 'recommend'} disabled={pending !== null} onClick={onRecommend}>按当前机器推荐</Button>
      <Button variant="ghost" aria-expanded={showAdvanced} aria-controls={advancedId} onClick={() => setShowAdvanced((current) => !current)}>
        {showAdvanced ? '收起高级设置' : '高级设置'}
      </Button>
    </ActionBar>}
    {showAdvanced && <div id={advancedId} className="grid gap-5 border-t border-[var(--border-subtle)] pt-4">
      {HYDRA_GROUPS.map((group) => <fieldset key={group.id} className="m-0 min-w-0 border-0 p-0">
        <legend className="mb-3 text-sm font-semibold"><HelpLabel help={group.description}>{group.label}</HelpLabel></legend>
        <div className="form-grid">
          {HYDRA_FIELDS.filter((field) => field.group === group.id).map((field) => <FormField
            label={<HelpLabel help={field.description}>{field.label}</HelpLabel>} key={field.path}>
            <Input type={field.type} step={field.type === 'number' ? 'any' : undefined} value={field.read(rl) ?? ''} onChange={(event) => onPatch(field.update(rl, event.target.value))} />
          </FormField>)}
        </div>
      </fieldset>)}
    </div>}
    {showSaveAction && <ActionBar><Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={onSave}>保存超参</Button></ActionBar>}
  </Section>;
});

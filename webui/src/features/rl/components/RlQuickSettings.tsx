import { memo } from 'react';
import type { MetaResponse, RlConfig, SamplingSpec } from '../../../shared/api/types';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { algorithmFromSampling } from '../model/normalizeRlPayload';
import { rolloutPatch } from '../model/rlPrefills';

const ALGORITHM_HINTS: Record<string, string> = {
  grpo: '标准组相对策略优化。',
  arpo: '在不确定窗口进行分支采样。',
  aepo: '强调高信息增益轨迹。',
  igpo: '使用信息增益调整优势。',
  gigpo: '按组内信息增益归一化。',
  rae: '结合分支验证与失败回溯。',
};

export type RlQuickSettingsProps = {
  rl: RlConfig;
  meta: MetaResponse | null;
  sampling?: SamplingSpec;
  onPatch: (next: RlConfig) => void;
  onSave: () => void | Promise<void>;
  saving?: boolean;
};

export const RlQuickSettings = memo(function RlQuickSettings({
  rl, meta, sampling, onPatch, onSave, saving,
}: RlQuickSettingsProps) {
  const samplingAlgorithm = algorithmFromSampling(sampling);
  const algorithm = samplingAlgorithm || rl.algo || 'grpo';
  const samplingGroup = sampling?.group_n;
  const rolloutCount = samplingGroup ?? rl.rollout_per_gpu ?? rl.actor_rollout_ref?.rollout?.n ?? 2;
  return <div className="grid gap-4">
    <div className="form-grid">
      <FormField label={<HelpLabel help={samplingAlgorithm
        ? '当前值由画布中的采样策略决定。如需修改，请在采样设置中调整。'
        : ALGORITHM_HINTS[algorithm]}>训练算法</HelpLabel>}>
        <Select value={algorithm} disabled={Boolean(samplingAlgorithm)}
          onChange={(event) => onPatch({ ...rl, algo: event.target.value })}>
          {(meta?.algos || ['grpo']).map((algo) => <option key={algo} value={algo}>{algo.toUpperCase()}</option>)}
        </Select>
      </FormField>
      <FormField label={<HelpLabel help={samplingGroup != null
        ? '当前值由画布中的采样策略决定。'
        : '每道题生成多少条候选轨迹，用于组内比较。'}>每题候选数</HelpLabel>}>
        <Input type="number" min={1} value={rolloutCount} disabled={samplingGroup != null}
          onChange={(event) => onPatch({ ...rl, ...rolloutPatch(rl, Number(event.target.value)) })} />
      </FormField>
      <FormField label={<HelpLabel help="同时执行轨迹采集的任务数。提高后会增加吞吐，也会占用更多 CPU、显存和模型并发。">并行采集数</HelpLabel>}>
        <Input type="number" min={1} value={rl.n_runners ?? 1}
          onChange={(event) => onPatch({ ...rl, n_runners: Number(event.target.value) })} />
      </FormField>
      <FormField label={<HelpLabel help="完整遍历训练数据的次数。">训练轮次</HelpLabel>}>
        <Input type="number" min={1} value={rl.trainer?.total_epochs ?? 1}
          onChange={(event) => onPatch({
            ...rl, trainer: { ...rl.trainer, total_epochs: Number(event.target.value) },
          })} />
      </FormField>
      <FormField label={<HelpLabel help="控制模型每次参数更新的幅度。数值过大可能导致训练不稳定。">学习率</HelpLabel>}>
        <Input type="number" min={0} step="any" value={rl.actor_rollout_ref?.actor?.optim?.lr ?? 1e-6}
          onChange={(event) => onPatch({
            ...rl,
            actor_rollout_ref: {
              ...rl.actor_rollout_ref,
              actor: {
                ...rl.actor_rollout_ref?.actor,
                optim: { ...rl.actor_rollout_ref?.actor?.optim, lr: Number(event.target.value) },
              },
            },
          })} />
      </FormField>
    </div>
    <Button loading={saving} onClick={onSave}>保存训练方案</Button>
  </div>;
});

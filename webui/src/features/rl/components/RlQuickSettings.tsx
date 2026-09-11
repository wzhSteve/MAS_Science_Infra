import { memo } from 'react';
import type { RlConfig } from '../../../shared/api/types';
import { FormField } from '../../../shared/components/FormField';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { rolloutPatch } from '../model/rlPrefills';

export type RlQuickSettingsProps = {
  rl: RlConfig;
  onPatch: (patch: Partial<RlConfig>) => void;
  onSave: () => void | Promise<void>;
  saving?: boolean;
};

export const RlQuickSettings = memo(function RlQuickSettings({ rl, onPatch, onSave, saving }: RlQuickSettingsProps) {
  return <details open className="grid gap-3">
    <summary className="cursor-pointer text-sm font-medium">采样与并行</summary>
    <div className="grid gap-3 pt-3">
      <FormField label="每题采样条数（GRPO 组大小）">
        <Input type="number" min={1} value={rl.rollout_per_gpu ?? rl.actor_rollout_ref?.rollout?.n ?? 2} onChange={(event) => onPatch(rolloutPatch(rl, Number(event.target.value)))} />
      </FormField>
      <FormField label="并行采集进程">
        <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => onPatch({ n_runners: Number(event.target.value) })} />
      </FormField>
      <p className="field-hint">训练算法：{rl.algo || 'grpo'}，作用于参与训练的 Agent。</p>
      <Button loading={saving} onClick={onSave}>保存训练配置</Button>
    </div>
  </details>;
});

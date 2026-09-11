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
    <summary className="cursor-pointer text-sm font-medium">训练简参（写入 rl.yaml，不画成图节点）</summary>
    <div className="grid gap-3 pt-3">
      <FormField label="每题采样条数（GRPO 组大小）">
        <Input type="number" min={1} value={rl.rollout_per_gpu ?? rl.actor_rollout_ref?.rollout?.n ?? 2} onChange={(event) => onPatch(rolloutPatch(rl, Number(event.target.value)))} />
      </FormField>
      <FormField label="并行采集进程 n_runners">
        <Input type="number" min={1} value={rl.n_runners ?? 1} onChange={(event) => onPatch({ n_runners: Number(event.target.value) })} />
      </FormField>
      <p className="field-hint">算法={rl.algo || 'grpo'} · 可训 Agent → --active-agent</p>
      <Button loading={saving} onClick={onSave}>写入 rl.yaml</Button>
    </div>
  </details>;
});

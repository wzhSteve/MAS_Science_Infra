import type { Config, RlConfig } from '../../../shared/api/types';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { Select } from '../../../shared/ui/select';

const FORMAT_ALGOS = new Set(['arpo', 'appo']);

export function FormatRewardField({
  algo,
  rl,
  onPatch,
}: {
  algo: string;
  rl: RlConfig;
  onPatch: (next: RlConfig) => void;
}) {
  if (!FORMAT_ALGOS.has(algo)) return null;
  const reward = (rl.algorithm?.reward as Config | undefined) ?? {};
  const includeFormat = Boolean(reward.include_format);
  return (
    <FormField
      label={<HelpLabel help="默认只调用 gpt-4o 判断终局答案对错，与 GRPO / Flow-GRPO 相同。打开后在对错分上叠加 ARPO/APPO 的格式分（抽不出答案 -0.3，格式不完整各 -0.05，双工具 +0.1）。">Reward</HelpLabel>}
    >
      <Select
        value={includeFormat ? 'format' : 'answer'}
        onChange={(event) => {
          const include_format = event.target.value === 'format';
          onPatch({
            ...rl,
            algorithm: {
              ...rl.algorithm,
              reward: { ...reward, include_format },
            },
          });
        }}
      >
        <option value="answer">仅答案对错（gpt-4o）</option>
        <option value="format">答案对错 + 格式</option>
      </Select>
    </FormField>
  );
}

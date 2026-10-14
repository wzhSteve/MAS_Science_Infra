import { memo, useMemo } from 'react';
import type { RlConfig, SamplingSpec } from '../../../shared/api/types';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { algorithmFromSampling } from '../model/normalizeRlPayload';
import {
  PRESET_EXPANSION_BASE, PRESET_LABELS, PRESET_TAGLINES,
  collapseComposedToPreset, expandPreset, expandPresetIntoDraft,
} from '../model/rlPrefills';

/** RL6 (L1): 算法组合面板 —— RLDesign §4.3 三层信息架构的最小渲染容器。
 * model 层共享物（expandPreset 镜像 / ComposedSpec 类型 / 归一规则）驱动
 * 双视图（TrainingSettings 03 区 + RlSettingsForm），容器只做呈现。
 *
 * 用户裁决（RL6 执行 brief）：组合面先下架 rae —— 可编辑算子仅
 * zero_prefix / appo_credit / aepo_entropy，base 仅 grpo | igpo | gigpo。
 * rae 预设仍出现在第一层下拉中（v1 展示 + expandPreset('rae') 展开正确渲染），
 * 但组合编辑器不提供 rae_verdict / rae_full_tgt 的编辑入口；RL3 管线的算子
 * 注册表不动（后端可跑 rae 组合，UI 只是先不暴露）。 */
const EDITABLE_BASES = ['grpo', 'igpo', 'gigpo'] as const;

const EDITABLE_OPERATORS = [
  {
    id: 'zero_prefix',
    label: 'zero_prefix',
    description: '分支前缀不参与训练',
    risk: 'resume_boundary 之前 token 优势置 0；与 appo_credit 同开时前缀信号丢弃更彻底',
  },
  {
    id: 'appo_credit',
    label: 'appo_credit',
    description: '分支轨迹的结局折扣回写父轨迹',
    risk: '提升组内信号密度，但父轨迹优势被外部结局影响',
  },
  {
    id: 'aepo_entropy',
    label: 'aepo_entropy',
    description: '低熵位置的优势差分放大',
    risk: '基优势为负时符号可能翻转',
  },
] as const;

const OPERATOR_ORDER = ['zero_prefix', 'appo_credit', 'aepo_entropy'] as const;

type Props = {
  rl: RlConfig;
  sampling?: SamplingSpec;
  onPatch: (next: RlConfig | ((current: RlConfig) => RlConfig)) => void;
};

function draftComposed(rl: RlConfig): NonNullable<RlConfig['composed']> {
  return rl.composed ?? { base: 'grpo', operators: [], reward: { components: [] }, loss: { mode: 'vanilla' } };
}

function toggleOperator(composed: NonNullable<RlConfig['composed']>, opId: string, on: boolean): NonNullable<RlConfig['composed']> {
  const current = composed.operators ?? [];
  const active = new Set(current.map(op => op.id));
  if (on) active.add(opId);
  else active.delete(opId);
  // 可编辑算子按注册表顺序 zero_prefix → appo_credit → aepo_entropy 排列；
  // 不可编辑算子（rae 系，用户裁决下架）原样保留在后——expandPreset('rae')
  // 展开的 rae_verdict 不因切换其他开关而被静默清除（保底渲染红线）。
  const editable = OPERATOR_ORDER.filter(id => active.has(id));
  const preserved = current
    .map(op => op.id)
    .filter(id => !OPERATOR_ORDER.includes(id as 'zero_prefix') && active.has(id));
  const operators = [...editable, ...preserved].map(id => {
    const existing = current.find(op => op.id === id);
    return existing ? { ...existing } : { id };
  });
  return { ...composed, operators };
}

export const ComposedPanel = memo(function ComposedPanel({ rl, sampling, onPatch }: Props) {
  const presetKeys = useMemo(() => Object.keys(PRESET_EXPANSION_BASE), []);
  const composed = rl.composed;
  const activePreset = composed ? null : (rl.algo || algorithmFromSampling(sampling) || 'grpo');
  const samplingMode = sampling?.mode;
  const needsBranching = (composed?.operators ?? []).some(op => op.id === 'appo_credit');
  const branchingMismatch = needsBranching && samplingMode != null && ['grpo_n', 'grpo'].includes(String(samplingMode));

  return <div className="composed-panel" data-composed={composed ? 'true' : 'false'}>
    <FormField label={<HelpLabel help="v1 单选预设（A9 永久保留）。预设路径完全不变形，落 rl.yaml 的 algo/tir_algo。">预设</HelpLabel>}>
      <Select value={composed ? '' : String(activePreset)}
        onChange={(event) => onPatch(current => collapseComposedToPreset(current, event.target.value))}>
        {composed && <option value="">（已展开为组合 — 选择预设返回 v1 形态）</option>}
        {presetKeys.map(preset => <option key={preset} value={preset}>
          {PRESET_LABELS[preset] ?? preset} — {PRESET_TAGLINES[preset] ?? ''}
        </option>)}
      </Select>
    </FormField>

    {!composed && <>
      <p className="composed-panel-hint">
        当前为 v1 预设形态（algo={String(activePreset)}）。展开后可编辑 base 与算子栈；回预设卡片会覆盖未另存的组合修改。
      </p>
      <Button size="sm" onClick={() => onPatch(current => expandPresetIntoDraft(current, String(activePreset)))}>
        展开为组合
      </Button>
    </>}

    {composed && <>
      <div className="composed-panel-grid">
        <FormField label={<HelpLabel help="组内基线估计器；所有非 grpo 预设都以 grpo 阶段 0 组基线为起点（RLDesign §3.1.6）。">Base</HelpLabel>}>
          <Select value={composed.base ?? 'grpo'}
            onChange={(event) => onPatch(current => ({ ...current, composed: { ...draftComposed(current), base: event.target.value as 'grpo' | 'igpo' | 'gigpo' } }))}>
            {EDITABLE_BASES.map(base => <option key={base} value={base}>{base.toUpperCase()}</option>)}
          </Select>
        </FormField>
        <div className="composed-operators" aria-label="Advantage 算子栈">
          <span className="composed-operators-label">
            <HelpLabel help="有序算子栈——执行顺序 zero_prefix → appo_credit → aepo_entropy（注册表顺序）。勾选即加入栈。">算子栈</HelpLabel>
          </span>
          {EDITABLE_OPERATORS.map(op => {
            const enabled = (composed.operators ?? []).some(item => item.id === op.id);
            return <label key={op.id} className="composed-operator-row" title={op.risk}>
              <input type="checkbox" checked={enabled}
                onChange={(event) => onPatch(current => ({ ...current, composed: toggleOperator(draftComposed(current), op.id, event.target.checked) }))} />
              <span className="composed-operator-name">{op.label}</span>
              <span className="composed-operator-desc">{op.description}</span>
            </label>;
          })}
          <p className="composed-operators-note">
            rae_verdict / rae_full_tgt 暂未在组合面开放（本阶段下架，预设 rae 的展开仍如实显示）。
          </p>
        </div>
        <FormField label={<HelpLabel help="Loss 形态：vanilla clip 或 future_kl（FIPO）。future_kl 需 D2 双写连通后生效。">Loss mode</HelpLabel>}>
          <Select value={composed.loss?.mode ?? 'vanilla'}
            onChange={(event) => onPatch(current => ({
              ...current,
              composed: { ...draftComposed(current), loss: { ...draftComposed(current).loss, mode: event.target.value as 'vanilla' | 'future_kl' } },
            }))}>
            <option value="vanilla">vanilla</option>
            <option value="future_kl">future_kl（阶段标注：需 D2 连通后生效）</option>
          </Select>
        </FormField>
        <FormField label={<HelpLabel help="显式组合下不再由 sampling.mode 覆盖算法结构（规则一）；group_n 数值联动仍保留。">group_n 联动</HelpLabel>}>
          <Input value={String(sampling?.group_n ?? rl.rollout_per_gpu ?? '')} readOnly
            placeholder="由采样策略（workflow.sampling）驱动" />
        </FormField>
      </div>
      <Button size="sm" variant="ghost"
        onClick={() => onPatch(current => {
          const preset = (composed.operators ?? []).length === 0 && composed.base === 'grpo' ? 'grpo'
            : String(Object.keys(PRESET_EXPANSION_BASE).find(key => {
              const expanded = expandPreset(key);
              return expanded && expanded.base === composed.base
                && JSON.stringify(expanded.operators?.map(op => op.id) ?? []) === JSON.stringify((composed.operators ?? []).map(op => op.id));
            }) ?? 'grpo');
          return collapseComposedToPreset(current, preset);
        })}>
        返回预设形态
      </Button>
      {branchingMismatch && <InlineNotice tone="warning">
        采样模式仍为 {String(samplingMode)}——当前组合含 appo_credit，假设分支采样（rollout_role 列），请手动对齐采样模式。
      </InlineNotice>}
    </>}
  </div>;
});

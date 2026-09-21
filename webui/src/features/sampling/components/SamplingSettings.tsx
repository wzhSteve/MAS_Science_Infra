import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { GitBranch } from 'lucide-react';
import type { BranchSiteSpec, Palette, SamplingSpec, WorkflowSpec } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Button } from '../../../shared/ui/button';
import { Checkbox } from '../../../shared/ui/checkbox';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import {
  BRANCH_GATE_OPTIONS,
  SAMPLING_MODE_OPTIONS,
  candidateTypeLabel,
  deriveBranchCandidates,
  effectiveSites,
  findCandidateSite,
  gateOptionsFor,
  isBranchingMode,
  updateCandidateSite,
  type BranchCandidate,
} from '../model/branchSites';

const DEFAULT_SAMPLING: SamplingSpec = {
  mode: 'grpo_n',
  group_n: 1,
  beam_size: 1,
  initial_rollouts: 1,
  max_branch_depth: 2,
  expand_in_runner: true,
  barriers: ['after_tool'],
  sites: [],
};

const CandidateRow = memo(function CandidateRow({ candidate, site, selected, onSelect, onToggle }: {
  candidate: BranchCandidate;
  site?: BranchSiteSpec;
  selected: boolean;
  onSelect: (id: string) => void;
  onToggle: (candidate: BranchCandidate, enabled: boolean) => void;
}) {
  const enabled = Boolean(site && site.enabled !== false);
  return <button type="button"
    className={`branch-site-item${selected ? ' is-selected' : ''}`}
    onClick={() => onSelect(candidate.id)}>
    <Checkbox checked={enabled} aria-label={`${enabled ? '关闭' : '开启'} ${candidate.label}`}
      onClick={(event) => event.stopPropagation()}
      onChange={() => onToggle(candidate, !enabled)} />
    <span className="branch-site-icon"><GitBranch size={14} /></span>
    <span><strong>{candidate.label}</strong><small>{candidateTypeLabel(candidate.kind)}</small></span>
    <StatusBadge tone={enabled ? 'success' : 'neutral'}>{enabled ? '开启' : '关闭'}</StatusBadge>
  </button>;
});

export const SamplingSettings = memo(function SamplingSettings({ workflow, palette, onChange, onSave, saving }: {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  onSave: () => Promise<void>;
  saving: boolean;
}) {
  const sampling = useMemo(() => ({ ...DEFAULT_SAMPLING, ...(workflow.sampling || {}) }), [workflow.sampling]);
  const candidates = useMemo(() => deriveBranchCandidates(workflow),
    [workflow.agents, workflow.edges, workflow.hub, workflow.routers, workflow.tools]);
  const sites = useMemo(() => effectiveSites(sampling, candidates), [sampling, candidates]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = useMemo(() =>
    candidates.find((candidate) => candidate.id === selectedId) || candidates[0] || null,
  [candidates, selectedId]);
  const selectedSite = useMemo(() => selected ? findCandidateSite(sites, selected) : undefined,
    [sites, selected]);
  const branching = isBranchingMode(sampling.mode);
  const enabledCount = useMemo(() => sites.filter((site) => site.enabled !== false).length, [sites]);
  const modes = useMemo(() =>
    Array.from(new Set([...(palette.sampling_modes || []), ...SAMPLING_MODE_OPTIONS.map((item) => item.value)])),
  [palette.sampling_modes]);
  const gates = useMemo(() => {
    if (!selected) return [];
    const compatible = gateOptionsFor(sampling.mode, selected.kind);
    const current = selectedSite?.gate.type;
    return current && !compatible.includes(current) ? [...compatible, current] : compatible;
  }, [sampling.mode, selected, selectedSite?.gate.type]);
  const workflowRef = useRef(workflow);
  const samplingRef = useRef(sampling);
  const candidatesRef = useRef(candidates);
  workflowRef.current = workflow;
  samplingRef.current = sampling;
  candidatesRef.current = candidates;

  const patchSampling = useCallback((patch: Partial<SamplingSpec>) => {
    onChange({ ...workflowRef.current, sampling: { ...samplingRef.current, ...patch } });
  }, [onChange]);
  const patchSite = useCallback((candidate: BranchCandidate, patch: Partial<BranchSiteSpec>) => {
    patchSampling(updateCandidateSite(samplingRef.current, candidatesRef.current, candidate, patch));
  }, [patchSampling]);
  const toggleSite = useCallback((candidate: BranchCandidate, enabled: boolean) => {
    patchSite(candidate, { enabled });
  }, [patchSite]);
  const modeLabel = (mode: string) => SAMPLING_MODE_OPTIONS.find((item) => item.value === mode)?.label || mode;
  const gateLabel = (gate: string) => BRANCH_GATE_OPTIONS.find((item) => item.value === gate)?.label || gate;
  const gateParams = selectedSite?.gate.params || {};
  const officialArpoGate = gateParams.use_official_arpo_gate !== false;
  const patchGateParams = (patch: Record<string, unknown>) => {
    if (!selected || !selectedSite) return;
    patchSite(selected, { gate: { ...selectedSite.gate, params: { ...gateParams, ...patch } } });
  };
  const switchArpoGate = (official: boolean) => {
    if (!selected || !selectedSite) return;
    const params = { ...gateParams };
    if (official) {
      params.use_official_arpo_gate = true;
      params.branch_probability ??= params.alpha ?? 0.5;
      params.entropy_weight ??= params.gamma ?? 0.5;
      delete params.alpha;
      delete params.gamma;
      delete params.entropy_threshold;
      delete params.threshold;
    } else {
      params.use_official_arpo_gate = false;
      params.alpha ??= params.branch_probability ?? 0.5;
      params.gamma ??= params.entropy_weight ?? 0.2;
      params.entropy_threshold ??= params.threshold ?? 0.15;
      delete params.branch_probability;
      delete params.entropy_weight;
      delete params.threshold;
    }
    patchSite(selected, { gate: { ...selectedSite.gate, params } });
  };

  return <div className="sampling-settings">
    <Section title="Sampling Policy" actions={<StatusBadge tone={branching ? 'info' : 'neutral'}>
      {branching ? 'Branch Sampling' : 'Independent Rollouts'}
    </StatusBadge>}>
      <div className="form-grid">
        <FormField label={<HelpLabel help="选择 GRPO、ARPO、AEPO、APPO 或 RAE 的 Rollout 策略。">Sampling Mode</HelpLabel>}>
          <Select value={sampling.mode} onChange={(event) => patchSampling({ mode: event.target.value })}>
            {modes.map((mode) => <option key={mode} value={mode}>{modeLabel(mode)}</option>)}
          </Select>
        </FormField>
        <FormField label={<HelpLabel help="每道题最终需要生成的完整 Rollout 数，也是 GRPO Group Size。">group_n</HelpLabel>}>
          <Input type="number" min={1} value={sampling.group_n}
            onChange={(event) => patchSampling({ group_n: Number(event.target.value) })} />
        </FormField>
        {branching && <>
          <FormField label={<HelpLabel help="Branch 前先从裸 Prompt 运行的独立 Rollout 数。">initial_rollouts</HelpLabel>}>
            <Input type="number" min={1} value={sampling.initial_rollouts}
              onChange={(event) => patchSampling({ initial_rollouts: Number(event.target.value) })} />
          </FormField>
          <FormField label={<HelpLabel help="每个 Snapshot 最多生成的后续 Branch 数。">beam_size</HelpLabel>}>
            <Input type="number" min={1} value={sampling.beam_size}
              onChange={(event) => patchSampling({ beam_size: Number(event.target.value) })} />
          </FormField>
          <FormField label={<HelpLabel help="单条 Rollout Tree 允许的最大 Branch Depth。">max_branch_depth</HelpLabel>}>
            <Input type="number" min={1} value={sampling.max_branch_depth}
              onChange={(event) => patchSampling({ max_branch_depth: Number(event.target.value) })} />
          </FormField>
        </>}
      </div>
    </Section>

    {branching && <Section title="Branch Sites"
      description="Branch Site 定义保存 Snapshot 的执行位置；Gate 再决定到达该位置时是否创建 Branch。"
      actions={<StatusBadge tone={enabledCount ? 'success' : 'warning'}>
      {enabledCount} 个已开启
    </StatusBadge>}>
      <div className="branch-site-list">
        {candidates.map((candidate) => {
          const site = findCandidateSite(sites, candidate);
          return <CandidateRow key={candidate.id} candidate={candidate} site={site}
            selected={selected?.id === candidate.id} onSelect={setSelectedId} onToggle={toggleSite} />;
        })}
        {!candidates.length && <p className="field-hint">画布中还没有可以尝试其他路径的位置。</p>}
      </div>

      {selected && selectedSite && <div className="branch-site-editor">
        <div className="branch-site-editor-title">
          <strong>{selected.label}</strong>
          <span>{candidateTypeLabel(selected.kind)}</span>
        </div>
        <div className="form-grid">
          <FormField label={<HelpLabel help={BRANCH_GATE_OPTIONS.find((item) => item.value === selectedSite.gate.type)?.help
            || '决定到达 Branch Site 时是否创建 Branch。'}>Gate</HelpLabel>}>
            <Select value={selectedSite.gate.type} onChange={(event) =>
              patchSite(selected, { gate: { ...selectedSite.gate, type: event.target.value } })}>
              {gates.map((gate) => <option key={gate} value={gate}>{gateLabel(gate)}</option>)}
            </Select>
          </FormField>
          <FormField label={<HelpLabel help="同一个 Branch Site 在一次执行中第几次命中时应用 Gate。">Apply on</HelpLabel>}>
            <Select value={selectedSite.when || 'first'} onChange={(event) =>
              patchSite(selected, { when: event.target.value as BranchSiteSpec['when'] })}>
              <option value="first">First</option>
              <option value="every">Every</option>
              <option value="nth">Nth</option>
            </Select>
          </FormField>
          {selectedSite.when === 'nth' && <FormField label={<HelpLabel help="第几次命中 Branch Site 时应用 Gate。">nth</HelpLabel>}>
            <Input type="number" min={1} value={selectedSite.nth || 1}
              onChange={(event) => patchSite(selected, { nth: Number(event.target.value) })} />
          </FormField>}
          <FormField label={<HelpLabel help="当前 Branch Site 触发后最多生成的后续 Branch 数。">beam_size</HelpLabel>}>
            <Input type="number" min={1} value={selectedSite.fork?.beam_size ?? sampling.beam_size}
              onChange={(event) => patchSite(selected, {
                fork: { ...selectedSite.fork, beam_size: Number(event.target.value) },
              })} />
          </FormField>
          <FormField label={<HelpLabel help="Branch Rollout 使用的 Reward / Credit Assignment 方案。">Reward Scheme</HelpLabel>}>
            <Select value={selectedSite.reward?.scheme || 'scalar_grpo'} onChange={(event) =>
              patchSite(selected, { reward: { ...selectedSite.reward, scheme: event.target.value } })}>
              <option value="scalar_grpo">scalar_grpo</option>
              <option value="rae_adjudicate">rae_adjudicate</option>
            </Select>
          </FormField>
        </div>
        {selectedSite.gate.type === 'entropy_delta' && <div className="branch-gate-parameters">
          <div className="form-grid">
            <FormField label={<HelpLabel help="Official ARPO 使用随机接受概率；Threshold Gate 使用 p > entropy_threshold。">Gate Variant</HelpLabel>}>
              <Select value={officialArpoGate ? 'official' : 'threshold'}
                onChange={(event) => switchArpoGate(event.target.value === 'official')}>
                <option value="official">Official ARPO Gate</option>
                <option value="threshold">Legacy ΔH Threshold Gate</option>
              </Select>
            </FormField>
            {officialArpoGate ? <>
              <FormField label={<HelpLabel help="Official ARPO 的基础 Branch 接受阈值。值越大越容易 Branch。">branch_probability</HelpLabel>}>
                <Input type="number" min={0} max={1} step="0.05"
                  value={Number(gateParams.branch_probability ?? 0.5)}
                  onChange={(event) => patchGateParams({ branch_probability: Number(event.target.value) })} />
              </FormField>
              <FormField label={<HelpLabel help="Entropy ΔH 对 Branch 接受概率的影响强度。">entropy_weight</HelpLabel>}>
                <Input type="number" min={0} step="0.05"
                  value={Number(gateParams.entropy_weight ?? 0.5)}
                  onChange={(event) => patchGateParams({ entropy_weight: Number(event.target.value) })} />
              </FormField>
            </> : <>
              <FormField label={<HelpLabel help="Legacy Gate 中 p = alpha + gamma × ΔH 的基础概率。">alpha</HelpLabel>}>
                <Input type="number" min={0} max={1} step="0.05" value={Number(gateParams.alpha ?? 0.5)}
                  onChange={(event) => patchGateParams({ alpha: Number(event.target.value) })} />
              </FormField>
              <FormField label={<HelpLabel help="Legacy Gate 中 Entropy ΔH 的权重。">gamma</HelpLabel>}>
                <Input type="number" min={0} step="0.05" value={Number(gateParams.gamma ?? 0.2)}
                  onChange={(event) => patchGateParams({ gamma: Number(event.target.value) })} />
              </FormField>
              <FormField label={<HelpLabel help="Legacy Gate 仅在 p > entropy_threshold 时 Branch。">entropy_threshold</HelpLabel>}>
                <Input type="number" min={0} max={1} step="0.05" value={Number(gateParams.entropy_threshold ?? 0.15)}
                  onChange={(event) => patchGateParams({ entropy_threshold: Number(event.target.value) })} />
              </FormField>
            </>}
          </div>
        </div>}
        {selectedSite.gate.type === 'dual_entropy' && <div className="branch-gate-parameters">
          <div className="form-grid">
            <FormField label={<HelpLabel help="用于估计 Outcome Entropy H_B 的 Probe 数量。">probe_k</HelpLabel>}>
              <Input type="number" min={1} value={Number(gateParams.probe_k ?? 2)}
                onChange={(event) => patchGateParams({ probe_k: Number(event.target.value) })} />
            </FormField>
            <FormField label={<HelpLabel help="Dual Entropy Gate 在 U = H_π × H_B ≥ u_threshold 时 Branch。">u_threshold</HelpLabel>}>
              <Input type="number" min={0} step="0.01" value={Number(gateParams.u_threshold ?? 0.05)}
                onChange={(event) => patchGateParams({ u_threshold: Number(event.target.value) })} />
            </FormField>
          </div>
        </div>}
      </div>}
    </Section>}

    <ActionBar>
      <Button variant="primary" loading={saving} onClick={() => { void onSave(); }}>保存 Sampling</Button>
    </ActionBar>
  </div>;
});

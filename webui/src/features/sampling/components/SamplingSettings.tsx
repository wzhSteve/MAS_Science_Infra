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
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';
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

export const SamplingSettings = memo(function SamplingSettings({ workflow, palette, onChange, onSave, saving, onLocate, compact = false, expanded = false, onExpand }: {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  onSave?: () => Promise<void>;
  saving?: boolean;
  onLocate?: (agentId: string) => void;
  compact?: boolean; expanded?: boolean; onExpand?: () => void;
}) {
  const { draft: training } = useTrainingConfig();
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
    Array.from(new Set([sampling.mode || 'grpo_n', ...(palette.sampling_modes || []), ...SAMPLING_MODE_OPTIONS.map((item) => item.value)])),
  [palette.sampling_modes, sampling.mode]);
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

  if (!workflow.sampling) {
    const algorithm = String(training.rl.algorithm?.tir_algo || training.rl.algo || 'grpo');
    const group = training.rl.rollout_per_gpu ?? training.rl.actor_rollout_ref?.rollout?.n;
    return <Section title="采样策略">
      {!compact && <p className="field-hint">此实验尚未启用 Workflow Sampling。编辑下列值保存到训练参数；启用后由采样策略统一决定。</p>}
      <div className="form-grid">
        <FormField label="训练算法">
          <Select value={algorithm} onChange={event => {
            const algo = event.target.value;
            training.patch(current => ({ ...current, algo, algorithm: { ...current.algorithm, tir_algo: algo } }));
          }}>
            {Array.from(new Set([algorithm, 'grpo', 'arpo', 'aepo', 'igpo', 'gigpo', 'rae'])).map(value =>
              <option key={value} value={value}>{value.toUpperCase()}</option>)}
          </Select>
        </FormField>
        <FormField label="每题候选数">
          <Input type="number" min={1} step={1} value={group ?? ''} onChange={event => {
            const n = event.target.value === '' ? undefined : Number(event.target.value);
            training.patch(current => ({ ...current, rollout_per_gpu: n,
              actor_rollout_ref: { ...current.actor_rollout_ref, rollout: { ...current.actor_rollout_ref?.rollout, n } } }));
          }} />
        </FormField>
      </div>
      <Button size="sm" disabled={training.dirty || training.pending !== null} onClick={() =>
        onChange({ ...workflow, sampling: { ...DEFAULT_SAMPLING, mode: algorithm === 'grpo' ? 'grpo_n' : algorithm, group_n: group ?? 1 } })}>
        启用分支采样配置
      </Button>
      {training.dirty && <p className="field-hint">请先保存实验，再启用 Sampling。</p>}
    </Section>;
  }
  return <div className="sampling-settings">
    <Section title="采样策略 · Sampling Policy" actions={<StatusBadge tone={branching ? 'info' : 'neutral'}>
      {branching ? 'Branch Sampling' : 'Independent Rollouts'}
    </StatusBadge>}>
      <div className="form-grid">
        <FormField label={<HelpLabel help="选择 GRPO、ARPO、AEPO、APPO 或 RAE 的 Rollout 策略。">算法</HelpLabel>}>
          <Select value={sampling.mode} onChange={(event) => patchSampling({ mode: event.target.value })}>
            {modes.map((mode) => <option key={mode} value={mode}>{modeLabel(mode)}</option>)}
          </Select>
        </FormField>
        <FormField label={<HelpLabel help="每道题最终需要生成的完整 Rollout 数，也是 GRPO Group Size。">每题候选数</HelpLabel>}>
          <Input type="number" min={1} value={sampling.group_n}
            onChange={(event) => patchSampling({ group_n: Number(event.target.value) })} />
        </FormField>
        {branching && <details className="parameter-details sampling-generation">
          <summary>候选生成参数</summary>
          <FormField label={<HelpLabel help="Branch 前先从裸 Prompt 运行的独立 Rollout 数。">初始 Rollout 数</HelpLabel>}>
            <Input type="number" min={1} value={sampling.initial_rollouts}
              onChange={(event) => patchSampling({ initial_rollouts: Number(event.target.value) })} />
          </FormField>
          <FormField label={<HelpLabel help="每个 Snapshot 最多生成的后续 Branch 数。">分支宽度</HelpLabel>}>
            <Input type="number" min={1} value={sampling.beam_size}
              onChange={(event) => patchSampling({ beam_size: Number(event.target.value) })} />
          </FormField>
          <FormField label={<HelpLabel help="单条 Rollout Tree 允许的最大 Branch Depth。">分支深度</HelpLabel>}>
            <Input type="number" min={1} value={sampling.max_branch_depth}
              onChange={(event) => patchSampling({ max_branch_depth: Number(event.target.value) })} />
          </FormField>
        </details>}
      </div>
    </Section>

    {branching && compact && !expanded && <div className="parameter-summary-row"><span>分支位置</span>
      <span>{enabledCount} 处</span><Button size="sm" variant="ghost" onClick={onExpand}>编辑</Button></div>}
    {branching && (!compact || expanded) && <Section title="分支位置"
      description={compact ? undefined : 'Branch Site 定义保存 Snapshot 的执行位置；Gate 再决定到达该位置时是否创建 Branch。'}
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
          {onLocate && <Button size="sm" variant="ghost" onClick={() => onLocate(selected.nodeId)}>定位到画布</Button>}
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
          <FormField label={<HelpLabel help="留空继承全局值；填写数字只覆盖当前分支位置。">局部分支宽度</HelpLabel>}>
            <Input type="number" min={1} step={1} value={selectedSite.fork?.beam_size ?? ''}
              placeholder={`继承全局 ${sampling.beam_size}`} onChange={(event) => {
                const fork = { ...selectedSite.fork };
                if (event.target.value === '') delete fork.beam_size;
                else fork.beam_size = Number(event.target.value);
                patchSite(selected, { fork });
              }} />
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

    {!branching && enabledCount > 0 && <p className="field-hint">已保留 {enabledCount} 个分支位置，当前策略不启用分支。</p>}
    {onSave && <ActionBar>
      <Button variant="primary" loading={saving} onClick={() => { void onSave(); }}>保存 Sampling</Button>
    </ActionBar>}
  </div>;
});

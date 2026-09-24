import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GitBranch, Trash2 } from 'lucide-react';
import type { BranchSiteSpec, Palette, SamplingOpportunity, SamplingPreviewResponse, SamplingSpec, WorkflowSpec } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { HelpLabel } from '../../../shared/components/HelpLabel';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Button } from '../../../shared/ui/button';
import { Checkbox } from '../../../shared/ui/checkbox';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';
import {
  BRANCH_GATE_OPTIONS,
  SAMPLING_MODE_OPTIONS,
  anchorKey,
  deriveBranchCandidates,
  effectiveSites,
  findCandidateSite,
  isBranchingMode,
  legacySiteCount,
  updateCandidateSite,
  type BranchCandidate,
} from '../model/branchSites';
import type { CanvasMode } from '../../mas/types';

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

const SUPPORT = {
  native: { tone: 'success', label: '可执行' },
  compatibility: { tone: 'warning', label: '兼容' },
  unavailable: { tone: 'neutral', label: '暂不支持' },
} as const;

const CandidateRow = memo(function CandidateRow({ opportunity, candidate, site, selected, onSelect, onToggle }: {
  opportunity: SamplingOpportunity;
  candidate: BranchCandidate;
  site?: BranchSiteSpec;
  selected: boolean;
  onSelect: (id: string) => void;
  onToggle: (candidate: BranchCandidate, enabled: boolean) => void;
}) {
  const enabled = Boolean(site && site.enabled !== false);
  const support = SUPPORT[opportunity.support];
  return <button type="button"
    className={`branch-site-item is-${opportunity.support}${selected ? ' is-selected' : ''}`}
    onClick={() => onSelect(opportunity.id)}>
    <Checkbox checked={enabled} aria-label={`${enabled ? '关闭' : '开启'} ${candidate.label}`}
      disabled={opportunity.support === 'unavailable' && !site}
      onClick={(event) => event.stopPropagation()}
      onChange={() => onToggle(candidate, !enabled)} />
    <span className="branch-site-icon"><GitBranch size={14} /></span>
    <span><strong>{opportunity.label}</strong><small>{opportunity.message}</small></span>
    <span className="branch-site-statuses">
      {enabled && <StatusBadge tone="info">已配置</StatusBadge>}
      <StatusBadge tone={support.tone}>{support.label}</StatusBadge>
    </span>
  </button>;
});

export const SamplingSettings = memo(function SamplingSettings({
  workflow, palette, onChange, onSave, saving, onLocate, compact = false, expanded = false, onExpand,
  canvasMode = 'workflow', preview, previewError, selectedOpportunityId, onSelectOpportunity,
}: {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  onSave?: () => Promise<void>;
  saving?: boolean;
  onLocate?: (agentId: string) => void;
  compact?: boolean; expanded?: boolean; onExpand?: () => void;
  canvasMode?: CanvasMode;
  preview?: SamplingPreviewResponse | null;
  previewError?: string | null;
  selectedOpportunityId?: string | null;
  onSelectOpportunity?: (id: string) => void;
}) {
  const { draft: training } = useTrainingConfig();
  const sampling = useMemo(() => ({ ...DEFAULT_SAMPLING, ...(workflow.sampling || {}) }), [workflow.sampling]);
  const candidates = useMemo(() => deriveBranchCandidates(workflow),
    [workflow.agents, workflow.edges, workflow.hub, workflow.routers, workflow.tools]);
  const sites = useMemo(() => effectiveSites(sampling), [sampling]);
  const [localSelectedId, setLocalSelectedId] = useState<string | null>(null);
  const opportunityId = selectedOpportunityId ?? localSelectedId;
  const opportunities = preview?.opportunities || [];
  const selectedOpportunity = opportunities.find(item => item.id === opportunityId)
    || opportunities.find(item => item.configured && item.enabled)
    || opportunities.find(item => item.support === 'native')
    || opportunities[0];
  useEffect(() => {
    if (!selectedOpportunity || selectedOpportunity.id === opportunityId) return;
    if (onSelectOpportunity) onSelectOpportunity(selectedOpportunity.id);
    else setLocalSelectedId(selectedOpportunity.id);
  }, [selectedOpportunity, opportunityId, onSelectOpportunity]);
  const selectOpportunity = useCallback((id: string) => {
    if (onSelectOpportunity) onSelectOpportunity(id);
    else setLocalSelectedId(id);
  }, [onSelectOpportunity]);
  const selected = useMemo(() => {
    if (!selectedOpportunity) return null;
    const existing = candidates.find(candidate => anchorKey(candidate.anchor) === selectedOpportunity.id);
    if (existing) return existing;
    const kind = selectedOpportunity.anchor.kind === 'after_tool' ? 'tool'
      : selectedOpportunity.anchor.kind === 'after_verifier' ? 'verifier'
        : selectedOpportunity.anchor.kind === 'on_edge' ? 'edge' : 'agent';
    return {
      id: selectedOpportunity.id, kind, label: selectedOpportunity.label,
      nodeId: selectedOpportunity.node_id, anchor: selectedOpportunity.anchor,
      recommendedGate: selectedOpportunity.allowed_gates[0] || 'entropy_delta',
    } satisfies BranchCandidate;
  }, [candidates, selectedOpportunity]);
  const selectedSite = useMemo(() => selected ? findCandidateSite(sites, selected) : undefined,
    [sites, selected]);
  const branching = isBranchingMode(sampling.mode);
  const legacyCount = legacySiteCount(sampling);
  const enabledCount = sites.filter((site) => site.enabled !== false).length + legacyCount;
  const modes = useMemo(() =>
    Array.from(new Set([sampling.mode || 'grpo_n', ...(palette.sampling_modes || []), ...SAMPLING_MODE_OPTIONS.map((item) => item.value)])),
  [palette.sampling_modes, sampling.mode]);
  const gates = useMemo(() => {
    if (!selectedOpportunity) return [];
    const compatible = selectedOpportunity.allowed_gates;
    const current = selectedSite?.gate.type;
    return current && !compatible.includes(current) ? [...compatible, current] : compatible;
  }, [selectedOpportunity, selectedSite?.gate.type]);
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
  const removeSite = useCallback((siteId: string) => {
    patchSampling({ barriers: [], sites: sites.filter(site => site.id !== siteId) });
  }, [patchSampling, sites]);
  const modeLabel = (mode: string) => ({
    grpo_n: '独立多次采样', grpo: '独立多次采样', arpo: '自适应分支采样',
    aepo: '信息增益预算', appo: '自适应路径采样', rae: '结果判定与回溯',
  }[mode] || SAMPLING_MODE_OPTIONS.find((item) => item.value === mode)?.label || mode);
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
    <Section title="采样策略" actions={<StatusBadge tone={branching ? 'info' : 'neutral'}>
      {modeLabel(sampling.mode || 'grpo_n')}
    </StatusBadge>}>
      <div className="form-grid">
        <FormField label={<HelpLabel help="选择如何为每道题生成候选。底层实现名称在高级说明中保留。">采样方式</HelpLabel>}>
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
      {branching && <div className="sampling-budget-summary">
        <span><small>最终候选</small><strong>{preview?.policy.group_n ?? sampling.group_n}</strong></span>
        <span><small>先独立运行</small><strong>{preview?.policy.initial_rollouts ?? sampling.initial_rollouts}</strong></span>
        <span><small>剩余分支预算</small><strong>{preview?.policy.remaining_budget ?? Math.max(0, Number(sampling.group_n) - Number(sampling.initial_rollouts))}</strong></span>
      </div>}
      <details className="parameter-details sampling-implementation">
        <summary>实现与配置说明</summary>
        <p className="field-hint">运行实现：{String(sampling.mode || 'grpo_n').toUpperCase()} · VERL estimator：GRPO compatibility。画布声明潜在边界，实际命中结果由 Rollout Tree 展示。</p>
      </details>
    </Section>

    {previewError && <InlineNotice tone="warning">采样能力读取失败：{previewError}</InlineNotice>}
    {branching && canvasMode !== 'sampling' && compact && !expanded && <div className="parameter-summary-row"><span>分支位置</span>
      <span>{enabledCount} 处</span><Button size="sm" variant="ghost" onClick={onExpand}>编辑</Button></div>}
    {branching && (canvasMode === 'sampling' || !compact || expanded) && <Section title="潜在采样位置"
      description={compact ? undefined : '选择一次运行可能到达的边界；这里只表示设计能力，不表示本次训练已经命中。'}
      actions={<StatusBadge tone={enabledCount ? 'success' : 'warning'}>
      {enabledCount} 个已配置
    </StatusBadge>}>
      {preview?.legacy && <InlineNotice tone="warning"><strong>兼容规则：</strong>{preview.legacy.message}</InlineNotice>}
      <div className="branch-site-list">
        {opportunities.map(opportunity => {
          const candidate = candidates.find(item => anchorKey(item.anchor) === opportunity.id) || {
            id: opportunity.id,
            kind: opportunity.anchor.kind === 'after_tool' ? 'tool' : opportunity.anchor.kind === 'after_verifier' ? 'verifier'
              : opportunity.anchor.kind === 'on_edge' ? 'edge' : 'agent',
            label: opportunity.label,
            nodeId: opportunity.node_id,
            anchor: opportunity.anchor,
            recommendedGate: opportunity.allowed_gates[0] || 'entropy_delta',
          } satisfies BranchCandidate;
          const site = findCandidateSite(sites, candidate);
          return <CandidateRow key={opportunity.id} opportunity={opportunity} candidate={candidate} site={site}
            selected={selectedOpportunity?.id === opportunity.id} onSelect={selectOpportunity} onToggle={toggleSite} />;
        })}
        {!opportunities.length && <p className="field-hint">当前 Workflow 还没有可识别的采样边界。</p>}
      </div>

      {selected && selectedOpportunity && <div className="branch-site-editor">
        <div className="branch-site-editor-title">
          <div><small>采样位置</small><strong>{selectedOpportunity.label}</strong></div>
          {canvasMode !== 'sampling' && onLocate && <Button size="sm" variant="ghost" onClick={() => onLocate(selected.nodeId)}>定位到画布</Button>}
        </div>
        <div className={`sampling-capability is-${selectedOpportunity.support}`}>
          <StatusBadge tone={SUPPORT[selectedOpportunity.support].tone}>{SUPPORT[selectedOpportunity.support].label}</StatusBadge>
          <span>{selectedOpportunity.message}</span>
        </div>
        {!selectedSite && selectedOpportunity.support !== 'unavailable' && <Button size="sm" variant="primary"
          onClick={() => toggleSite(selected, true)}>在此位置尝试其他路径</Button>}
        {selectedSite && <>
        <h4 className="sampling-inspector-heading">条件</h4>
        <div className="form-grid">
          <FormField label={<HelpLabel help={BRANCH_GATE_OPTIONS.find((item) => item.value === selectedSite.gate.type)?.help
            || '决定到达该位置时是否创建新的后续路径。'}>什么情况下扩展</HelpLabel>}>
            <Select value={selectedSite.gate.type} onChange={(event) =>
              patchSite(selected, { gate: { ...selectedSite.gate, type: event.target.value } })}>
              {gates.map((gate) => <option key={gate} value={gate}>{gateLabel(gate)}</option>)}
            </Select>
          </FormField>
          <FormField label={<HelpLabel help="同一位置在一次执行中第几次到达时检查条件。">何时检查</HelpLabel>}>
            <Select value={selectedSite.when || 'first'} onChange={(event) =>
              patchSite(selected, { when: event.target.value as BranchSiteSpec['when'] })}>
              <option value="first">第一次到达</option>
              <option value="every">每次到达</option>
              <option value="nth">第 N 次到达</option>
            </Select>
          </FormField>
          {selectedSite.when === 'nth' && <FormField label={<HelpLabel help="第几次命中 Branch Site 时应用 Gate。">nth</HelpLabel>}>
            <Input type="number" min={1} value={selectedSite.nth || 1}
              onChange={(event) => patchSite(selected, { nth: Number(event.target.value) })} />
          </FormField>}
          <FormField label={<HelpLabel help="保留当前位置之前的上下文，只重新生成后续内容。留空继承全局值。">最多新增后续路径</HelpLabel>}>
            <Input type="number" min={1} step={1} value={selectedSite.fork?.beam_size ?? ''}
              placeholder={`继承全局 ${sampling.beam_size}`} onChange={(event) => {
                const fork = { ...selectedSite.fork };
                if (event.target.value === '') delete fork.beam_size;
                else fork.beam_size = Number(event.target.value);
                patchSite(selected, { fork });
              }} />
          </FormField>
          <FormField label={<HelpLabel help="新后续路径使用的奖励与归因方式。">结果评价方式</HelpLabel>}>
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
        <details className="parameter-details sampling-contract">
          <summary>高级合同信息</summary>
          <dl>
            <div><dt>anchor</dt><dd>{selectedOpportunity.anchor.kind}</dd></div>
            <div><dt>agent</dt><dd>{selectedOpportunity.anchor.agent_id || '—'}</dd></div>
            <div><dt>tool</dt><dd>{selectedOpportunity.anchor.tool_id || '—'}</dd></div>
            <div><dt>runtime event</dt><dd>{selectedOpportunity.runtime_event || '未实现'}</dd></div>
            <div><dt>resume</dt><dd>{selectedOpportunity.prefix || '不可用'}</dd></div>
          </dl>
        </details>
        <ActionBar><Button size="sm" variant="ghost" onClick={() => removeSite(selectedSite.id)}>
          <Trash2 size={13} />删除此位置
        </Button></ActionBar>
        </>}
      </div>}
    </Section>}

    {!branching && enabledCount > 0 && <p className="field-hint">已保留 {enabledCount} 个分支位置，当前策略不启用分支。</p>}
    {onSave && <ActionBar>
      <Button variant="primary" loading={saving} onClick={() => { void onSave(); }}>保存 Sampling</Button>
    </ActionBar>}
  </div>;
});

/** Rollout Sampling panel: simplified trajectory + branch site sockets. */

import { useMemo, useState } from 'react';
import type { WorkflowSpec } from '../graph/workflowGraph';
import {
  GATE_TYPES,
  findSiteForCandidate,
  upsertSiteFromCandidate,
  type BranchSiteConfig,
} from '../graph/branchSites';
import {
  branchableNodes,
  buildTrajectoryGraph,
  type TrajNode,
} from './trajectoryGraph';

type Props = {
  workflow: WorkflowSpec;
  onChange: (wf: WorkflowSpec) => void;
  samplingModes?: string[];
};

export default function RolloutSamplingPanel({ workflow, onChange, samplingModes }: Props) {
  const [expandTools, setExpandTools] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const traj = useMemo(() => buildTrajectoryGraph(workflow), [workflow]);
  const sites = useMemo(
    () => ((workflow.sampling?.sites || []) as BranchSiteConfig[]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(workflow.sampling?.sites || [])],
  );
  const sampling = workflow.sampling || { mode: 'grpo_n', group_n: 1, beam_size: 1 };
  const modes = samplingModes || ['grpo_n', 'arpo', 'aepo', 'appo', 'rae'];

  const sockets = useMemo(
    () => branchableNodes(traj, { includeTools: expandTools }),
    [traj, expandTools],
  );

  const selected: TrajNode | null =
    sockets.find((n) => n.id === selectedId) || sockets[0] || null;

  const patchSampling = (patch: Record<string, unknown>) => {
    onChange({
      ...workflow,
      sampling: { ...(workflow.sampling || {}), ...patch },
    });
  };

  const patchSites = (next: BranchSiteConfig[]) => {
    onChange({
      ...workflow,
      sampling: {
        ...(workflow.sampling || {}),
        sites: next as any,
      },
    });
  };

  const mainNodes = traj.nodes.filter(
    (n) =>
      n.kind === 'start' ||
      n.kind === 'end' ||
      n.kind === 'agent' ||
      n.kind === 'verify' ||
      n.kind === 'router',
  );

  return (
    <div className="card" style={{ marginTop: 12, boxShadow: 'none', border: '1px solid #d0d7de' }}>
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 style={{ marginTop: 0, marginBottom: 4 }}>Rollout Sampling</h3>
          <p className="muted" style={{ margin: 0 }}>
            简化轨迹：选 branch 起点与 gate，写入 <code>sampling.sites</code>（与主画布同源）。
          </p>
        </div>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 13 }}>
          <input
            type="checkbox"
            checked={expandTools}
            onChange={(e) => setExpandTools(e.target.checked)}
          />
          展开 tool 屏障
        </label>
      </div>

      <div className="row" style={{ marginTop: 10, flexWrap: 'wrap' }}>
        <label className="muted">mode</label>
        <select
          value={String(sampling.mode || 'grpo_n')}
          onChange={(e) => patchSampling({ mode: e.target.value })}
        >
          {modes.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <label className="muted">group_n</label>
        <input
          type="number"
          min={1}
          style={{ width: 70 }}
          value={Number(sampling.group_n ?? 1)}
          onChange={(e) => patchSampling({ group_n: Number(e.target.value) })}
        />
        <label className="muted">beam_size</label>
        <input
          type="number"
          min={1}
          style={{ width: 70 }}
          value={Number(sampling.beam_size ?? 1)}
          onChange={(e) => patchSampling({ beam_size: Number(e.target.value) })}
        />
      </div>

      {/* Trajectory strip */}
      <div
        style={{
          marginTop: 12,
          padding: '12px 8px',
          overflowX: 'auto',
          background: 'linear-gradient(180deg, #f6f8fa 0%, #fff 100%)',
          borderRadius: 8,
          border: '1px solid #eaeef2',
          minHeight: 96,
        }}
      >
        <div className="row" style={{ alignItems: 'stretch', gap: 0, flexWrap: 'nowrap' }}>
          {mainNodes.map((n, i) => {
            const site = n.candidate ? findSiteForCandidate(sites, n.candidate) : undefined;
            const on = !!site?.enabled;
            const isSel = selected?.id === n.id;
            return (
              <div key={n.id} className="row" style={{ alignItems: 'center', gap: 0, flexShrink: 0 }}>
                {i > 0 ? (
                  <div
                    style={{
                      width: 28,
                      height: 2,
                      background: '#d0d7de',
                      margin: '0 4px',
                    }}
                  />
                ) : null}
                <button
                  type="button"
                  onClick={() => n.branchable && setSelectedId(n.id)}
                  style={{
                    minWidth: 88,
                    padding: '8px 10px',
                    borderRadius: 8,
                    border: isSel ? '2px solid #0969da' : on ? '2px solid #1a7f37' : '1px solid #d0d7de',
                    background: n.kind === 'start' || n.kind === 'end' ? '#f6f8fa' : '#fff',
                    cursor: n.branchable ? 'pointer' : 'default',
                    textAlign: 'left',
                  }}
                >
                  <div style={{ fontSize: 11, color: '#656d76' }}>{n.kind}</div>
                  <div style={{ fontWeight: 650, fontSize: 13 }}>{n.label}</div>
                  {n.branchable ? (
                    <div className={on ? 'chip ok' : 'chip'} style={{ marginTop: 4, fontSize: 10 }}>
                      {on ? `branch:${site?.gate?.type || 'on'}` : 'no branch'}
                    </div>
                  ) : null}
                </button>
              </div>
            );
          })}
        </div>

        {expandTools ? (
          <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {traj.nodes
              .filter((n) => n.kind === 'tool')
              .map((n) => {
                const site = n.candidate ? findSiteForCandidate(sites, n.candidate) : undefined;
                const on = !!site?.enabled;
                const isSel = selected?.id === n.id;
                return (
                  <button
                    key={n.id}
                    type="button"
                    onClick={() => setSelectedId(n.id)}
                    style={{
                      padding: '6px 8px',
                      borderRadius: 6,
                      border: isSel ? '2px solid #0969da' : on ? '1px solid #1a7f37' : '1px dashed #d0d7de',
                      background: '#fff',
                      fontSize: 12,
                    }}
                  >
                    tool:{n.label}
                    {on ? ` · ${site?.gate?.type}` : ''}
                  </button>
                );
              })}
          </div>
        ) : null}
      </div>

      {/* Mini inspector for selected socket */}
      {selected?.candidate ? (
        <div
          className="row"
          style={{
            marginTop: 12,
            alignItems: 'flex-end',
            flexWrap: 'wrap',
            gap: 12,
            padding: 10,
            border: '1px solid #eaeef2',
            borderRadius: 8,
          }}
        >
          <div style={{ minWidth: 140 }}>
            <div className="muted" style={{ fontSize: 12 }}>
              选中屏障
            </div>
            <strong>{selected.candidate.label}</strong>
            <div className="muted" style={{ fontSize: 11 }}>
              {selected.candidate.anchorKind}
            </div>
          </div>
          <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={!!findSiteForCandidate(sites, selected.candidate)?.enabled}
              onChange={(e) => {
                const cur = findSiteForCandidate(sites, selected.candidate!);
                patchSites(
                  upsertSiteFromCandidate(sites, selected.candidate!, {
                    enabled: e.target.checked,
                    gate: {
                      type: cur?.gate?.type || selected.candidate!.recommendedGate,
                      params: cur?.gate?.params || {},
                    },
                  }),
                );
              }}
            />
            启用 branch
          </label>
          <div className="field" style={{ margin: 0 }}>
            <label>gate</label>
            <select
              value={
                findSiteForCandidate(sites, selected.candidate)?.gate?.type ||
                selected.candidate.recommendedGate
              }
              onChange={(e) =>
                patchSites(
                  upsertSiteFromCandidate(sites, selected.candidate!, {
                    enabled: true,
                    gate: {
                      type: e.target.value,
                      params: findSiteForCandidate(sites, selected.candidate!)?.gate?.params || {},
                    },
                  }),
                )
              }
            >
              {GATE_TYPES.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ margin: 0 }}>
            <label>reward</label>
            <select
              value={findSiteForCandidate(sites, selected.candidate)?.reward?.scheme || 'scalar_grpo'}
              onChange={(e) =>
                patchSites(
                  upsertSiteFromCandidate(sites, selected.candidate!, {
                    enabled: true,
                    reward: {
                      ...(findSiteForCandidate(sites, selected.candidate!)?.reward || {}),
                      scheme: e.target.value,
                    },
                  }),
                )
              }
            >
              <option value="scalar_grpo">scalar_grpo</option>
              <option value="rae_adjudicate">rae_adjudicate</option>
            </select>
          </div>
          <div className="field" style={{ margin: 0 }}>
            <label>beam</label>
            <input
              type="number"
              min={1}
              style={{ width: 64 }}
              value={Number(findSiteForCandidate(sites, selected.candidate)?.fork?.beam_size ?? 2)}
              onChange={(e) =>
                patchSites(
                  upsertSiteFromCandidate(sites, selected.candidate!, {
                    enabled: true,
                    fork: {
                      ...(findSiteForCandidate(sites, selected.candidate!)?.fork || {}),
                      beam_size: Number(e.target.value),
                      share_observation: true,
                      resume_mode:
                        findSiteForCandidate(sites, selected.candidate!)?.fork?.resume_mode ||
                        'messages',
                    },
                  }),
                )
              }
            />
          </div>
        </div>
      ) : (
        <p className="muted" style={{ marginTop: 8 }}>
          当前轨迹无可选 branch 点（先在主画布添加 Agent）。
        </p>
      )}

      <p className="muted" style={{ marginTop: 8, marginBottom: 0 }}>
        已启用 {sites.filter((s) => s.enabled).length} 个站点 · 主路径{' '}
        {mainNodes
          .filter((n) => n.kind === 'agent' || n.kind === 'verify' || n.kind === 'router')
          .map((n) => n.label)
          .join(' → ') || '—'}
      </p>
    </div>
  );
}

// Copyright (c) Microsoft. All rights reserved.

export type ScienceEvent = {
  eventId: string;
  kind: string;
  agentId: string;
  payload: Record<string, unknown>;
};

export type ScienceTrajectoryRow = {
  trajectoryId: string;
  reward: number | null;
  answer: string;
  formatOk: boolean;
  nSearch: number;
  nPython: number;
  kinds: string[];
  events: ScienceEvent[];
};

export type ScienceHypothesis = {
  plugin: string;
  eventId: string;
  message: string;
  meta: Record<string, unknown>;
};

export type ScienceTrainSignal = {
  advantage: string;
  loss: string;
  meta: Record<string, unknown>;
};

export type ScienceView = {
  n: number;
  meanReward: number | null;
  rewards: Array<{ index: number; id: string; reward: number }>;
  eventCounts: Record<string, number>;
  nError: number;
  nFeedback: number;
  nHypotheses: number;
  trajectories: ScienceTrajectoryRow[];
  hypotheses: ScienceHypothesis[];
  trainSignal: ScienceTrainSignal;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown, fallback = ''): string {
  if (value == null) {
    return fallback;
  }
  return String(value);
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function eventKind(raw: unknown): string {
  if (typeof raw === 'string') {
    return raw;
  }
  const rec = asRecord(raw);
  if (rec && typeof rec.value === 'string') {
    return rec.value;
  }
  return asString(raw, 'unknown');
}

function parseEvent(raw: unknown): ScienceEvent {
  const rec = asRecord(raw) ?? {};
  const payload = asRecord(rec.payload) ?? {};
  return {
    eventId: asString(rec.event_id ?? rec.eventId),
    kind: eventKind(rec.kind),
    agentId: asString(rec.agent_id ?? rec.agentId, 'hub'),
    payload,
  };
}

function parseTrajectory(raw: unknown): ScienceTrajectoryRow {
  const rec = asRecord(raw) ?? {};
  const events = Array.isArray(rec.events) ? rec.events.map(parseEvent) : [];
  const kinds = events.map((event) => event.kind);
  const meta = asRecord(rec.meta) ?? {};
  return {
    trajectoryId: asString(rec.trajectory_id ?? rec.trajectoryId),
    reward: asNumber(rec.final_reward ?? rec.finalReward ?? rec.reward),
    answer: asString(rec.final_answer ?? rec.finalAnswer ?? rec.answer),
    formatOk: Boolean(rec.format_ok ?? rec.formatOk ?? meta.format_ok),
    nSearch: asNumber(rec.n_search ?? rec.nSearch ?? meta.n_search) ?? 0,
    nPython: asNumber(rec.n_python ?? rec.nPython ?? meta.n_python) ?? 0,
    kinds,
    events,
  };
}

function errorSignature(text: string): string {
  let signature = String(text || '')
    .trim()
    .toLowerCase();
  signature = signature.replace(/[0-9a-f]{8,}/g, '<id>');
  signature = signature.replace(/\d+/g, '<n>');
  signature = signature.replace(/[^\w\s<>]+/g, ' ');
  signature = signature.replace(/\s+/g, ' ').trim();
  return signature.slice(0, 160) || '<empty>';
}

function eventErrorText(event: ScienceEvent): string {
  const payload = event.payload;
  if (typeof payload.error === 'string') {
    return payload.error;
  }
  try {
    return JSON.stringify(payload);
  } catch {
    return String(payload);
  }
}

export function diagnoseScience(trajectories: ScienceTrajectoryRow[], rewards: number[]): ScienceHypothesis[] {
  const out: ScienceHypothesis[] = [];

  for (const traj of trajectories) {
    for (const event of traj.events) {
      if (event.kind !== 'error') {
        continue;
      }
      out.push({
        plugin: 'log_error',
        eventId: event.eventId,
        message: eventErrorText(event),
        meta: {},
      });
    }
  }

  if (rewards.length >= 2) {
    const first = rewards[0];
    const last = rewards[rewards.length - 1];
    if (last <= first) {
      out.push({
        plugin: 'loss_volatility',
        eventId: '',
        message: `reward not rising: first=${first.toFixed(4)} last=${last.toFixed(4)}`,
        meta: { n: rewards.length },
      });
    }
    const diffs = rewards.slice(1).map((value, index) => Math.abs(value - rewards[index]));
    const meanAbs = diffs.reduce((sum, value) => sum + value, 0) / diffs.length;
    if (meanAbs > 0.5) {
      out.push({
        plugin: 'loss_volatility',
        eventId: '',
        message: `reward volatility mean_abs_delta=${meanAbs.toFixed(4)}`,
        meta: { n: rewards.length },
      });
    }
  }

  const n = trajectories.length;
  const split = Math.max(1, Math.floor(n / 2));
  const hits = new Map<string, Array<{ index: number; eventId: string; raw: string }>>();
  trajectories.forEach((traj, index) => {
    for (const event of traj.events) {
      if (event.kind !== 'error') {
        continue;
      }
      const raw = eventErrorText(event);
      const signature = errorSignature(raw);
      const rows = hits.get(signature) ?? [];
      rows.push({ index, eventId: event.eventId, raw });
      hits.set(signature, rows);
    }
  });
  for (const [signature, rows] of hits) {
    const trajIdx = [...new Set(rows.map((row) => row.index))].sort((a, b) => a - b);
    if (trajIdx.length < 2) {
      continue;
    }
    const inLate = trajIdx.filter((index) => index >= split);
    if (inLate.length === 0 && n > split) {
      continue;
    }
    const last = rows[rows.length - 1];
    out.push({
      plugin: 'cognitive_convergence',
      eventId: last.eventId,
      message: `cognitive high deviation: repeated error ${JSON.stringify(signature)} on ${trajIdx.length} trajectories`,
      meta: { signature, n_traj: trajIdx.length, example: last.raw.slice(0, 200) },
    });
  }

  for (const traj of trajectories) {
    if (traj.reward == null || traj.reward < 0.9) {
      continue;
    }
    const errEv = traj.events.find((event) => event.kind === 'error');
    const empty = !traj.answer.trim();
    if (errEv) {
      out.push({
        plugin: 'reward_hacking',
        eventId: errEv.eventId,
        message: `reward hacking: reward=${traj.reward.toFixed(3)} but trajectory has ERROR`,
        meta: { reward: traj.reward },
      });
    } else if (empty || !traj.formatOk) {
      out.push({
        plugin: 'reward_hacking',
        eventId: '',
        message: `reward hacking: reward=${traj.reward.toFixed(3)} but format/answer missing`,
        meta: { reward: traj.reward, format_ok: traj.formatOk },
      });
    }
  }

  return out;
}

export function parseSciencePayload(raw: unknown): { trajectories: ScienceTrajectoryRow[]; trainSignal: ScienceTrainSignal } {
  const root = asRecord(raw) ?? {};
  const source = asRecord(root.batch) ?? root;
  const listRaw = Array.isArray(source.trajectories) ? (source.trajectories as unknown[]) : [];
  const trajectories = listRaw.map(parseTrajectory);
  const signalRec = asRecord(root.train_signal) ?? asRecord(root.trainSignal) ?? {};
  const advantage = asRecord(signalRec.advantage);
  const loss = asRecord(signalRec.loss);
  return {
    trajectories,
    trainSignal: {
      advantage: asString(advantage?.name ?? signalRec.advantage),
      loss: asString(loss?.name ?? signalRec.loss),
      meta: asRecord(signalRec.meta) ?? {},
    },
  };
}

export function buildScienceView(raw: unknown): { view: ScienceView | null; error: string | null } {
  try {
    const { trajectories, trainSignal } = parseSciencePayload(raw);
    const rewards = trajectories
      .map((traj, index) =>
        traj.reward == null ? null : { index: index + 1, id: traj.trajectoryId, reward: traj.reward },
      )
      .filter((point): point is { index: number; id: string; reward: number } => point != null);
    const eventCounts: Record<string, number> = {};
    let nError = 0;
    let nFeedback = 0;
    for (const traj of trajectories) {
      for (const kind of traj.kinds) {
        eventCounts[kind] = (eventCounts[kind] ?? 0) + 1;
        if (kind === 'error') {
          nError += 1;
        }
        if (kind === 'feedback') {
          nFeedback += 1;
        }
      }
    }
    const rewardValues = rewards.map((point) => point.reward);
    const hypotheses = diagnoseScience(trajectories, rewardValues);
    const meanReward =
      rewardValues.length > 0 ? rewardValues.reduce((sum, value) => sum + value, 0) / rewardValues.length : null;
    return {
      view: {
        n: trajectories.length,
        meanReward,
        rewards,
        eventCounts,
        nError,
        nFeedback,
        nHypotheses: hypotheses.length,
        trajectories,
        hypotheses,
        trainSignal,
      },
      error: null,
    };
  } catch (err) {
    return { view: null, error: err instanceof Error ? err.message : 'Failed to parse Science MAS JSON' };
  }
}

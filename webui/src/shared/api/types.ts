export type Config = Record<string, unknown>;

export interface AgentSpec extends Config {
  id: string;
  role?: string;
  skills?: string[];
  tools?: string[];
  memory_scope?: string;
  system_prompt?: string;
  model?: string;
  trainable?: boolean;
}

export interface WorkflowEdgeSpec {
  from: string;
  to: string;
  kind?: string;
  meta?: Config;
}

export interface WorkflowSpec extends Config {
  schema_version?: string;
  topology: string;
  entry_agent?: string;
  hub: {
    role?: string;
    skills?: string[];
    verify?: string | null;
    max_feedback_hops?: number;
    system_prompt?: string;
    [key: string]: unknown;
  };
  tools: string[];
  agents?: AgentSpec[];
  edges?: WorkflowEdgeSpec[];
  llm?: Config;
  memory?: Config;
  archive?: Config;
}

export interface RlConfig extends Config {
  algo?: string;
  profile?: string;
  model_path?: string;
  rollout_per_gpu?: number;
  n_runners?: number;
  devices?: { ids?: number[]; [key: string]: unknown };
  trainer?: { n_gpus_per_node?: number; total_epochs?: number; experiment_name?: string; [key: string]: unknown };
  data?: { train_batch_size?: number; [key: string]: unknown };
  actor_rollout_ref?: {
    actor?: {
      optim?: { lr?: number; [key: string]: unknown };
      clip_ratio_low?: number;
      clip_ratio_high?: number;
      entropy_coeff?: number;
      kl_loss_coef?: number;
      [key: string]: unknown;
    };
    rollout?: { n?: number; gpu_memory_utilization?: number; [key: string]: unknown };
    [key: string]: unknown;
  };
}

export interface LlmConfig extends Config {
  kind?: string;
  model?: string;
  base_url?: string;
  model_path?: string;
  port?: number;
  gpu_memory_utilization?: number;
  api_key_set?: boolean;
}

export interface Bundle {
  id: string;
  meta: Config & { seed?: number; name?: string };
  llm: LlmConfig;
  workflow: WorkflowSpec;
  rl: RlConfig;
  harness: Config & { plugins?: string[] };
  executable: { ok: boolean; reason?: string; topology?: string };
  path: string;
}

export interface MetaResponse {
  algos: string[];
  profiles: string[];
  harness_plugins: string[];
  stub_harness: string[];
  llm_kinds: string[];
}

export interface Palette {
  skills?: string[];
  roles?: string[];
  tools?: string[];
  edge_kinds?: string[];
  templates?: Array<{ id: string; label: string; workflow?: WorkflowSpec }>;
}

export interface GpuInfo {
  id: number;
  name: string;
  mem_total_mb: number;
  mem_used_mb: number;
  util: number;
}

export interface GpuResponse {
  gpus: GpuInfo[];
  count: number;
  recommend?: RlConfig;
  error?: string;
}

export interface HealthResponse {
  ok: boolean;
  error?: string;
  status_code?: number;
  models?: unknown;
  url?: string;
}

export interface AglHealth {
  ok: boolean;
  error?: string;
  ui_path?: string;
  origin?: string;
}

export interface RunSummary {
  run_id: string;
  kind: string;
  running: boolean;
  started_at?: number;
  experiment_id?: string;
}

export interface RunDetail extends RunSummary { log_tail?: string }
export interface Hypothesis { plugin: string; event_id?: string; message: string; [key: string]: unknown }
export interface RewardPoint { index: number; reward: number }
export interface Trajectory {
  trajectory_id: string;
  reward?: number;
  format_ok?: boolean;
  answer?: string;
  events?: unknown[];
  [key: string]: unknown;
}
export interface TrainSignal {
  advantage?: Config & { name?: string };
  loss?: Config & { clip_ratio_low?: number; clip_ratio_high?: number; entropy_coeff?: number; kl_loss_coef?: number };
  meta?: Config;
}
export interface MonitorResponse {
  empty?: boolean;
  n?: number;
  mean_reward?: number | null;
  n_error?: number;
  hypotheses?: Hypothesis[];
  trajectories?: Trajectory[];
  rewards?: RewardPoint[];
  train_rewards?: RewardPoint[];
  step_rewards?: RewardPoint[];
  train_signal?: TrainSignal;
  agl_online?: boolean;
}
export interface CollectRow { id: string; answer?: unknown; reward?: number; tool?: boolean }
export interface CollectResponse {
  n: number;
  mean_reward?: number;
  rows?: CollectRow[];
  path?: string;
  train_signal?: TrainSignal;
}
export interface CollectBody {
  mock?: boolean;
  n?: number;
  algo?: string;
  tasks?: unknown[];
  parquet?: string;
  data_n?: number;
  source?: string;
  sequential?: boolean;
}
export interface ScienceEvent {
  type: string;
  experiment_id: string;
  ts?: number;
  data?: Config;
}

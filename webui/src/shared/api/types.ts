export type Config = Record<string, unknown>;

export type AgentKind = 'hub' | 'planner' | 'tool' | 'verifier' | 'blank';
export type RouterStrategy = 'llm_choice' | 'score' | 'round_robin';

export interface AgentSpec extends Config {
  id: string;
  kind?: AgentKind;
  role?: string;
  skills?: string[];
  tools?: string[];
  memory_scope?: string;
  system_prompt?: string;
  model?: string;
  trainable?: boolean;
  profile?: Config;
  meta?: Config;
}

export interface RouterSpec extends Config {
  id: string;
  candidates: string[];
  strategy?: RouterStrategy;
  scorer?: string | null;
  meta?: Config;
}

export interface WorkflowEdgeSpec {
  from: string;
  to: string;
  kind?: 'message' | 'tool_call' | 'feedback' | 'route' | 'sample_barrier';
  meta?: Config;
}

export interface SamplingSpec extends Config {
  mode?: string;
  group_n?: number;
  beam_size?: number;
  initial_rollouts?: number;
  max_branch_depth?: number;
  expand_in_runner?: boolean;
  barriers?: string[];
  sites?: BranchSiteSpec[];
  extra?: Config;
}

export type BranchAnchorKind = 'after_agent_turn' | 'after_tool' | 'after_verifier' | 'on_edge' | 'on_token';

export interface BranchSiteSpec extends Config {
  id: string;
  enabled?: boolean;
  anchor: Config & {
    kind: BranchAnchorKind;
    agent_id?: string | null;
    tool_id?: string | null;
    edge_id?: string | null;
    skill_id?: string | null;
  };
  when?: 'first' | 'every' | 'nth';
  nth?: number;
  gate: Config & { type: string; params?: Config };
  fork?: Config & {
    beam_size?: number | null;
    share_observation?: boolean;
    resume_mode?: string;
    probe_max_tokens?: number;
  };
  reward?: Config & {
    scheme?: string;
    p_plus?: number;
    k_min?: number;
    epsilon_f?: number;
    dead_end_backprop?: number;
  };
  priority?: number;
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
  routers?: RouterSpec[];
  edges?: WorkflowEdgeSpec[];
  sampling?: SamplingSpec;
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
  algorithm?: Config & {
    adv_estimator?: string;
    use_kl_in_reward?: boolean;
    tir_algo?: string;
    tir?: Config;
  };
  trainer?: Config & {
    n_gpus_per_node?: number;
    total_epochs?: number;
    total_training_steps?: number;
    experiment_name?: string;
    project_name?: string;
    nnodes?: number;
    test_freq?: number;
  };
  data?: Config & {
    train_files?: string;
    val_files?: string;
    train_batch_size?: number;
    val_batch_size?: number | null;
    max_prompt_length?: number;
    max_response_length?: number;
    truncation?: string;
  };
  actor_rollout_ref?: {
    actor?: {
      optim?: { lr?: number; [key: string]: unknown };
      ppo_mini_batch_size?: number;
      ppo_micro_batch_size_per_gpu?: number;
      use_kl_loss?: boolean;
      clip_ratio_low?: number;
      clip_ratio_high?: number;
      entropy_coeff?: number;
      kl_loss_coef?: number;
      [key: string]: unknown;
    };
    rollout?: {
      n?: number;
      tensor_model_parallel_size?: number;
      log_prob_micro_batch_size_per_gpu?: number;
      gpu_memory_utilization?: number;
      name?: string;
      [key: string]: unknown;
    };
    ref?: { log_prob_micro_batch_size_per_gpu?: number; [key: string]: unknown };
    model?: {
      path?: string;
      use_remove_padding?: boolean;
      enable_gradient_checkpointing?: boolean;
      [key: string]: unknown;
    };
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
  credential_source?: 'experiment' | 'resource' | 'service' | 'none';
  config_revision?: string;
  resource_id?: string | null;
  resource_revision?: number | null;
}

export interface ExperimentMeta extends Config {
  id: string;
  seed: number;
  refs: Record<string, string>;
  pipeline: string[];
  name: string;
  agl_metrics_url: string;
}

export interface Bundle {
  id: string;
  meta: ExperimentMeta;
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
  agent_templates?: Array<{
    id: string;
    kind: AgentKind | 'router';
    label: string;
    hint?: string;
  }>;
  tool_agents?: Array<{
    id: string;
    backend: string;
    llm_required: boolean;
    description?: string;
  }>;
  sampling_modes?: string[];
  gate_types?: string[];
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
  status: 'reachable' | 'authentication_failed' | 'rate_limited' | 'unsupported' | 'network_error'
    | 'timeout' | 'server_error' | 'invalid_response' | 'configuration_error' | 'request_failed';
  message: string;
  error?: string;
  status_code?: number;
  models: string[];
  url: string;
  checked_at: string;
  config_revision: string;
  probe_type: 'models';
  inference_verified: false;
  tool_calling_verified: false;
}

export interface ReadinessIssue {
  code: string;
  message: string;
  hint?: string;
}

export interface DependencyReadiness {
  available: boolean;
  missing: string[];
  error: string | null;
  hint: string;
}

export interface ModelReadinessResponse {
  experiment_id: string;
  checked_at: string;
  ready: boolean;
  model: {
    kind: string;
    model: string;
    base_url: string;
    api_key_set: boolean;
    credential_source: 'experiment' | 'resource' | 'service' | 'none';
    resource_id?: string | null;
    resource_revision?: number | null;
    config_revision: string;
  };
  blocking_issues: ReadinessIssue[];
  warnings: ReadinessIssue[];
  dependencies: { live: DependencyReadiness; parquet: DependencyReadiness };
  capabilities: { tool_calling: 'unknown'; token_ids: 'unknown'; logprobs: 'unknown'; policy_version: 'unknown' };
  probe: HealthResponse | null;
}

export interface TrainingCheck {
  id: string;
  label: string;
  status: 'pass' | 'warning' | 'error';
  message: string;
}

export interface TrainingPreflightResponse {
  experiment_id: string;
  ready: boolean;
  revision: string;
  effective: {
    algorithm: string;
    algorithm_source: 'workflow.sampling' | 'rl';
    profile: string;
    model: {
      source: 'resource' | 'legacy';
      model_path: string;
      resource_id?: string | null;
      resource_revision?: number | null;
      resource_name?: string | null;
    };
    gpu_ids: number[];
    group_n: number;
    branch_site_count: number;
    trainable_agents: string[];
    data: Config;
  };
  checks: TrainingCheck[];
  blocking_issues: ReadinessIssue[];
  warnings: ReadinessIssue[];
  launch_preview: string[];
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
  state?: 'preparing' | 'starting' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
  running: boolean;
  started_at?: number;
  ended_at?: number | null;
  returncode?: number | null;
  stop_reason?: string | null;
  failure_stage?: string | null;
  message?: string | null;
  experiment_id?: string;
  meta?: Config;
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
export type RolloutExecution = 'mock' | 'live';
export interface RolloutRunRequest {
  workflow: WorkflowSpec;
  task: { id?: string; question: string };
  execution: RolloutExecution;
}
export interface RolloutRunSummary {
  run_id: string;
  trajectory_id?: string | null;
  status: 'running' | 'succeeded' | 'failed' | 'interrupted';
  execution: RolloutExecution;
  started_at: string;
  finished_at?: string | null;
  model: { source: string; name: string; policy_version: string | null } | null;
  final_answer: string | null;
  termination_reason: string | null;
  format_ok: boolean | null;
  model_call_count: number;
  tool_call_count: number;
  trace_status: 'complete' | 'partial' | 'unavailable';
  error?: { stage: string; code: string; message: string } | null;
}
export interface RolloutTrajectory extends Config {
  trajectory_id?: string;
  events?: unknown[];
  meta?: Config;
}
export interface RolloutRunContext {
  run: RolloutRunSummary;
  workflow: WorkflowSpec;
  task: { id: string; question: string };
  model_binding?: Config | null;
  redaction_applied?: boolean;
}
export interface TrajectoryPage { offset: number; total: number; next_offset: number | null }
export interface RolloutTrajectoryResponse extends RolloutRunContext {
  trajectory: RolloutTrajectory;
  preview?: boolean;
  event_page?: TrajectoryPage;
  message_page?: TrajectoryPage;
  metadata_truncated?: boolean;
  messages_truncated?: boolean;
  snapshots?: Array<{ snapshot_id: string; coverage: Record<string, boolean> | null; error?: string | null }>;
}
export interface RolloutHistoryItem {
  run_id: string;
  run: RolloutRunSummary | null;
  question_preview: string;
  record_error: string | null;
}
export interface RolloutHistoryPage { items: RolloutHistoryItem[]; next_cursor: string | null }
export interface ScienceEvent {
  type: string;
  experiment_id: string;
  ts?: number;
  data?: Config;
}

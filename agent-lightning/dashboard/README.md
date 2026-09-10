# Agent-lightning Dashboard

This is the dashboard for Agent-lightning. It is a web application that allows you to inspect your Agent-lightning store and debug running experiments.

The dashboard is built with React, Mantine UI, and Storybook.

## Pages

- **Rollouts / Traces / Runners / Resources / Settings** — store inspection and debugging
- **Metrics** (`/metrics`) — episode reward curves from the store, plus GRPO step `training/reward` and actor loss from `metrics.jsonl`
- **Science MAS** (`/science`) — load a `science-infra collect` JSON (file / paste / sample) to inspect trajectories, thin harness hypotheses, and TrainSignal. No extra Store API.

### Where to look for training curves

| UI | URL | What it shows |
|----|-----|----------------|
| Agent-Lightning Dashboard Metrics | store host, usually `http://localhost:4747/metrics` | reward / loss for the current experiment |
| Ray Dashboard Metrics | `http://localhost:8265/#/metrics` | Ray cluster Prometheus metrics (often empty without Prometheus); **not** VERL reward/loss |
| TensorBoard | `tensorboard --logdir checkpoints/.../tensorboard` | full VERL Tracking scalars |

Training writes:

- `checkpoints/<project>/<experiment>/metrics.jsonl` (also `AGL_METRICS_JSONL`)
- TensorBoard under `TENSORBOARD_DIR` / experiment `tensorboard/`

Store API: `GET /v1/agl/metrics`.

## npm scripts

## Build and dev scripts

- `dev` – start development server
- `build` – build production version of the app
- `preview` – locally preview production build

### Testing scripts

- `eslint` - runs ESLint
- `stylelint` - runs Stylelint
- `prettier` - runs Prettier
- `typecheck` - runs TypeScript typecheck
- `vitest` – runs vitest tests
- `chromatic` – runs chromatic tests

### Other scripts

- `storybook` – starts storybook dev server
- `build-storybook` – build production storybook bundle to `storybook-static`

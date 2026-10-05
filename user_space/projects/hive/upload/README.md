# HIVE 源码不复制到用户区

HIVE 本体留在仓库 `ref_Rep/HIVE`。本项目只写 thin 适配器：

- `adapted/context.py`：共享 `HiveEpisodeContext`（Planner / Executor / Diagnoser / SystemMemory）
- `adapted/entry.py`：按窗口分发 `run_window`

不要改 `ref_Rep/HIVE`。不要把 `Solver.solve` 当作 Collect / 训练 / 采样入口。

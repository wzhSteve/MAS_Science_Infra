# Phase 5 产物迁移 probe 报告（动手前产出）

日期：2026-10-13

## 1. checkpoints / tensorboard（原地标注，不迁）

`mas/train_tir_agent.py:216`：`local_dir = config["trainer"].get("default_local_dir")
or f"checkpoints/{project}/{experiment}"`——路径来源是 **AGL hydra 配置**
（`default_local_dir`）+ 相对 cwd 的 fallback。写死在训练入口（零移动红线上），
且实际值由 experiments YAML 的 trainer 段决定，**不迁**，原地标注。
产物现存 `mas/checkpoints/`、`mas/tensorboard_log/`（.gitignore 已覆盖）。

## 2. .tir_archives 读取点（2 个代码点 + env 覆盖）

| 读取点 | 消费方式 | 迁移动作 |
|---|---|---|
| `mas/workflow/memory/archive.py:archive_root()` | 写+读单点 | 默认值改 `artifacts/mas/tir_archives` |
| `mas/workflow/memory/archive.py:load_resume_messages` | 磁盘扫描 | 三级回退：内存索引 → 新目录 → 旧目录 |
| 环境变量 `TIR_ARCHIVE_DIR` | 运行时覆盖 | 优先级保持最高，测试用它隔离 |

旧目录 `mas/.tir_archives`（现存有数据）保留只读，fallback 消费。

## 3. .local_expansion 读取点（6 处，1 写 5 读）

| 读取点 | 行 | 迁移动作 |
|---|---|---|
| `mas/workflow/active_set.py:EXPANSION_DIR`（dump_local_expansion 写入） | 37 | **唯一写入点**：默认改 artifacts/mas/local_expansion；`TIR_LOCAL_EXPANSION_DIR` env 优先级不变 |
| `mas/workflow/active_set.py:load_local_expansion` | 462 | 读取点 1：新目录 → 旧目录 fallback |
| `science_infra/control/app.py:331`（树 API，read-only scan） | | 读取点 2：经 `paths.mas_artifacts_root()` + legacy fallback |
| `scripts/branch_rollout_ui_test.py` | exp_dir | 读取点 3：同上 fallback |
| `scripts/rollout_tree_verify.py` | exp_dir | 读取点 4：同上 |
| `scripts/arpo_train_verify.py` | exp_dir | 读取点 5：同上 |

## 4. 禁双写策略

- 新目录是唯一写入点（EXPANSION_DIR 常量 + archive_root() 默认值）；
- 旧目录永不写（读 fallback only）——`mas/.tir_archives` 与 `mas/.local_expansion`
  物理目录**不删**（训练中途数据安全）；
- `.gitignore` 增补 `artifacts/mas/`。

## 5. 测试面

- `test_memory_package.py:test_archive_root_new_default`（覆盖原 Phase 2 的
  位置冻结断言——该断言职责改由 Phase 5 断言接管）
- `test_archive_paths.py`（新增）：resume 三级回退 / expansion 读旧写新
- 回归门加 `python scripts/branch_rollout_ui_test.py --train`（树 API 走新目录）

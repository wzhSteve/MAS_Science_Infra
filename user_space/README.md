# 用户代码区

这是 Science Studio 唯一允许辅助 AI 与用户写入的代码树。

- 管理代码（`mas/`、`science_infra/`、`rl/`、`webui/`）只读。
- 实验 YAML（`experiments/<id>/`）只能通过官方 API / 画布保存，不能由 AI 直接改文件。
- 本目录下的 Python 只经 `mas/workflow/user_gateway` 被管理区调用。

```text
user_space/
  registry.yaml
  projects/<id>/
    manifest.yaml
    upload/          # 原件，ingest 后只读
    adapted/         # AI 构建目录
    contracts/       # io_contract / workflow / sampling
    artifacts/
```

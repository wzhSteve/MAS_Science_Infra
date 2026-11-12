# SupervisorAgent

Independent Science Studio project. The Smolagents fork from [LINs-lab/SupervisorAgent](https://github.com/LINs-lab/SupervisorAgent) is vendored at commit `ab116b557b095ae8d45bdf2d61057ce19519d4ff` under `upload/SupervisorAgent/` with its Apache-2.0 licenses. The upstream example tree is omitted because it includes bundled browser session cookies. See the upstream repository for the original examples.

The platform invokes one terminal planner window. The manager, research worker, and verifier run inside that window. The adapter uses upstream SupervisorAgent for step recording, risk detection and interventions. It uses a tool-calling manager instead of the original GAIA example's Python-executing manager. The original GAIA browser, multimodal tools, and attachments are outside this adapter; results are not a replication of the paper.

## Run locally

From the repository root, install the adapter requirements in your Science Studio Python environment:

```powershell
python -m pip install -r user_space/projects/supervisor_agent/adapted/requirements.txt
$env:OPENAI_API_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:MODEL_NAME = 'your-tool-calling-model'
python user_space/projects/supervisor_agent/adapted/entry.py "What is 17 times 19?"
python user_space/projects/supervisor_agent/adapted/entry.py "What is 17 times 19?" --no-supervisor
```

Set `OPENAI_API_KEY` in your environment if the endpoint requires it. In Science Studio, the current rollout model binding takes precedence. The endpoint must support OpenAI-format tool calls and return token usage. To enable search and page retrieval, set `web_tools: true` in `contracts/runtime.yaml` and provide `SERPER_API_KEY` in your environment.

The last CLI output line is JSON with the answer, agent termination state, elapsed seconds, model calls, action steps, supervisor reviews, and token usage. The same metrics are attached to the platform plan trace. For the supervisor ablation, run the same questions with and without supervision using identical model, tools, and budgets. Compare with `pure_pev` separately on the same dataset and scoring rule, and report differences in tools and model settings.

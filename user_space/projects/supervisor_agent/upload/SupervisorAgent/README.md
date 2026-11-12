# Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

[![Arxiv](https://img.shields.io/badge/arXiv-SupervisorAgent-b31b1b)](http://arxiv.org/abs/2510.26585)
[![Python](https://img.shields.io/badge/Python-3.12.0-green)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **TL;DR**: SupervisorAgent is a lightweight, **runtime, adaptive supervisor** for Multi‑Agent Systems (MAS). It intervenes only at **high‑risk interactions** to **correct errors**, **guide inefficient behaviors**, and **purify long/noisy observations** without modifying the base MAS. On GAIA with the Smolagent baseline, it reduces token usage by **~29.68%** while maintaining competitive success rates (see the tables below).

<p align="center">
  <img src="assets/methodology.png" alt="SupervisorAgent concept and GAIA efficiency" width="75%">
</p>

*Figure 1: **The architecture and workflow of SupervisorAgent.** (a) The LLM-free adaptive filter for identifying high-risk interactions. (b) The context window, aggregating goals and traces for situational awareness. (c) The spectrum of intervention actions, from simple approval to intensive verification. (d, e) Case study on a GAIA task, comparing the baseline MAS (d) with our SMAS (e), which cuts steps by 43% and token cost by over 70%. (f) The supervise workflow for an interaction, from filtering to a final supervision action.*

## 🚀 Key Features

- **Non‑intrusive runtime supervision.** Plug‑in style, zero changes to the base MAS’s architecture.
- **Three complementary strategies.** *Proactive Error Correction*, *Guidance for Inefficiency*, and *Adaptive Observation Purification*—ranging from minimal approval to verification‑level intervention.
- **Model‑ / MAS‑agnostic.** Demonstrated across multiple LLMs and MAS frameworks; consistent token savings with stable or improved accuracy.

## 🔎 What is SMAS (Supervised Multi‑Agent System)?

A **SMAS** is a **Multi‑Agent System** augmented with a **meta‑level control agent**—the **Supervisor**. The Supervisor monitors agent interactions **in real time** and **selectively intervenes** to detect and mitigate operational risks **without modifying the base agents’ core logic**.

- In this repository, the conceptual Supervisor is implemented as the concrete agent **SupervisorAgent**.
- **Relationship:** *SupervisorAgent* is the **component**; **SMAS** is the **resulting system** when a MAS is instrumented with SupervisorAgent.
- In figures and tables, **“Smolagent + SMAS”** means *Smolagent* **instrumented with SupervisorAgent**.

> **In short:** `Your MAS` **+** `SupervisorAgent` → **SMAS**.

## 🧩 How It Works

SupervisorAgent attaches to your MAS as a **runtime callback** around interaction steps (agent↔agent, agent↔tool, agent↔memory). A fast **LLM‑free filter** detects high‑risk situations (explicit errors, inefficient loops, or excessive observations). When triggered, the supervisor builds a **context window** (agent name, global/local tasks and traces, and interaction summary) and selects an **intervention action** from a graded action space: `approve`, `provide_guidance`, `correct_observation`, or `run_verification`.

This design keeps the steady‑state overhead low and focuses LLM calls where they matter most, improving **efficiency** and **robustness** with minimal intrusion.

## 📊 Results at a Glance

* **GAIA Benchmark(based on GPT-4.1):** On the GAIA validation set (pass@1), SMAS achieves a substantial **29.68%** average token reduction while maintaining a statistically equivalent success rate(see the Table 1). The efficiency gains are even more pronounced on difficult tasks (pass@3), saving **32.39%** (Level 2) and **30.10%** (Level 3) in token costs.

* **Broad Generalization(based on Qwen3-32B):** SMAS acts as a universal efficiency enhancer, almost achieving a Pareto improvement across diverse domains(see the Table 2). In mathematical reasoning (AIME), it improves the solve rate by **6.67%** while cutting token costs by **18.92%**. In code generation (HumanEval), it maintains competitive accuracy and further slashes token usage by **23.74%**.

---
*Table 1: **Overall performance on the GAIA validation set (based on GPT-4.1).** Our SMAS consistently reduces the average token cost comparing to Smolagent baseline while achieving competitive success rates.*

| **Method** | **Avg. Acc.** | **Avg. Tokens (K)** | **L1 Acc.** | **L1 Tokens (K)** | **L2 Acc.** | **L2 Tokens (K)** | **L3 Acc.** | **L3 Tokens (K)** |
|:------------|:--------------:|:-------------------:|:------------:|:-----------------:|:------------:|:-----------------:|:------------:|:-----------------:|
| CodeAgent | 40.00 | 120.40 | 56.60 | 92.84 | 34.88 | 131.90 | 23.08 | 138.54 |
| OWL | 45.40 | 111.07 | 56.56 | 67.72 | 43.02 | 110.36 | 29.16 | 209.34 |
| OAgents | 49.09 | 340.50 | 66.04 | 260.27 | 47.67 | 358.63 | 19.23 | 444.11 |
| Smolagent | 50.91 | 527.76 | 62.26 | 298.51 | 53.49 | 619.59 | 19.23 | 691.33 |
| AWorld | 60.00 | 128.27 | 67.92 | 69.61 | 62.79 | 164.08 | 34.62 | 133.65 |
|||||**pass@1**|
| Smolagent | 50.91 | 527.76 | 62.26 | 298.51 | 53.49 | 619.59 | 19.23 | 691.33 |
| + **SMAS (ours)** | 50.91 | 371.12 <sub><span style="color:red;">↓29.68%</span></sub> | 62.26 | 258.28 <sub><span style="color:red;">↓13.48%</span></sub> | 51.16 | 404.96 <sub><span style="color:red;">↓34.64%</span></sub> | 26.92 <sub><span style="color:blue;">↑7.69%</span></sub> | 489.22 <sub><span style="color:red;">↓29.23%</span></sub> |
|||||**pass@2**|
| Smolagent | 58.18 | 467.19 | 69.81 | 275.85 | 59.30 | 548.02 | 30.77 | 589.92 |
| + **SMAS (ours)** | 58.79 <sub><span style="color:blue;">↑0.61%</span></sub> | 389.54 <sub><span style="color:red;">↓16.62%</span></sub> | 73.58 <sub><span style="color:blue;">↑3.77%</span></sub> | 270.07 <sub><span style="color:red;">↓2.10%</span></sub> | 56.98 | 420.97 <sub><span style="color:red;">↓23.18%</span></sub> | 34.62 <sub><span style="color:blue;">↑3.85%</span></sub> | 529.20 <sub><span style="color:red;">↓10.29%</span></sub> |
|||||**pass@3**|
| Smolagent | 61.82 | 502.40 | 71.70 | 282.14 | 63.95 | 605.05 | 34.62 | 611.87 |
| + **SMAS (ours)** | 63.03 <sub><span style="color:blue;">↑1.21%</span></sub> | 369.52 <sub><span style="color:red;">↓26.45%</span></sub> | 75.47 <sub><span style="color:blue;">↑3.77%</span></sub> | 276.84 <sub><span style="color:red;">↓1.88%</span></sub> | 62.79 | 409.05 <sub><span style="color:red;">↓32.39%</span></sub> | 38.46 <sub><span style="color:blue;">↑3.84%</span></sub> | 427.72 <sub><span style="color:red;">↓30.10%</span></sub> |


---
*Table 2: **Generalization across diverse benchmarks (based on Qwen3-32B).** Our SMAS consistently reduces token costs while maintaining or improving accuracy across math, code, and QA. All gains are relative to the Smolagent baseline.*

| **Method** | **Metrics** | **GSM-hard** | **AIME** | **HumanEval** | **MBPP** | **DROP** |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Vanilla | Acc / F1 (%) | 67.17 | 26.67 | 76.82 | 80.09 | 76.36 |
|  | Avg. Tokens (K) | 0.37 | 2.01 | 0.28 | 0.27 | 0.46 |
| CoT SC (3-shot) | Acc / F1 (%) | 69.01 | 30.00 | 77.78 | 81.26 | 77.72 |
|  | Avg. Tokens (K) | 2.62 | 14.26 | 1.42 | 1.29 | 2.73 |
| OWL | Acc / F1 (%) | 72.48 | 33.33 | 90.74 | 79.08 | 79.85 |
|  | Avg. Tokens (K) | 15.67 | 56.11 | 31.87 | 54.80 | 11.47 |
| MetaAgent | Acc / F1 (%) | 72.14 | 26.67 | 74.08 | 79.86 | 78.16 |
|  | Avg. Tokens (K) | 4.35 | 6.24 | 2.59 | 6.39 | 1.43 |
| **Smolagent** | Acc / F1 (%) | **74.33** | **30.00** | **92.07** | **85.68** | **81.08** |
|  | Avg. Tokens (K) | 11.59 | 59.14 | 40.91 | 111.07 | 12.01 |
| + **SMAS (ours)** | Acc / F1 (%) | **75.50** | **36.67** | **92.68** | **84.43** | **79.80** |
|  | Avg. Tokens (K) | 10.55 <sub><span style="color:red;">↓8.92%</span></sub> | 47.95 <sub><span style="color:red;">↓18.92%</span></sub> | 31.19 <sub><span style="color:red;">↓23.74%</span></sub> | 103.71 <sub><span style="color:red;">↓6.62%</span></sub> | 11.34 <sub><span style="color:red;">↓5.60%</span></sub> |

## ⚙️ Setup
To get started, follow the steps below:

### Clone the repository

```bash
git clone https://github.com/LINs-lab/SupervisorAgent.git
cd SupervisorAgent/smolagents_SMAS
```

### Install dependencies

Run the following command to install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Install our development version of `smolagents`

```bash
pip install -e ".[dev]"
```

### Configuration `.env` File

We provide a template file named `.env.template` for your convenience. You can copy this file and rename it to `.env`, then fill in your actual API keys and service addresses. For example:

```bash
cp .env.template .env
```

Then edit `.env` with your actual API keys and service endpoints.

#### Search Services

The agent uses the following external services for web search. Please configure the corresponding API keys:

- **Search Engine: SerpApi**  
  Used for performing web searches. Get your API key from [SerpApi](https://serpapi.com/) and [SerperAPI](https://serper.dev) ,and set it as in `.env`:
  ```
  SERP_API_KEY=<your-serp-api-key>
  SERPER_API_KEY=<your-serper-api-key>
  ```

## 🧪 Evaluation

### Evaluation on GAIA

> **Note**  
> If the GAIA dataset is **not found** under `examples/open_deep_research/data/gaia`, `run_gaia.py` will **automatically download** it from Hugging Face on the first run. No manual preparation is required for the default setup.

#### Quick Start (Auto-download, recommended)

1. Navigate to the GAIA example directory:
   ```bash
   cd ./examples/open_deep_research
   ```
2. Run the evaluator (use your own model IDs if different):
   ```bash
   python run_gaia.py --model-id gpt-4.1 --model-id-summary gpt-4.1 --run-name gaia-test --supervisor
   ```
   - You can choose the `--model-id` and `--model-id-summary` based on your own setup. 
   - The script will fetch GAIA automatically if it’s not already present.

#### Manual Data Preparation (Optional)

If you prefer to prepare GAIA yourself, place it here:

```
./examples/open_deep_research/data/gaia
```

Expected structure:
```
├── data
│   └── gaia
│       └── 2023
│           ├── test
│           └── validation
```

#### Customize the Run

- To use different splits or subsets, edit the arguments or check the logic in `run_gaia.py` and adjust dataset paths/splits as needed.

### Evaluation on other benchmarks
*Will be updated soon.*


## 📚 Citation

If this repository or the paper is helpful to your work, please cite:

```bibtex
@inproceedings{lin2026stop,
  title={Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems},
  author={Fulin Lin and Shaowen Chen and Ruishan Fang and Hongwei Wang and Tao Lin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=pzFhtpkabh}
}
```

## 🙏 Acknowledgements

We gratefully acknowledge the authors and contributors of [smolagents](https://github.com/huggingface/smolagents) for their elegant design and clear implementation, which provided the foundation for this work. Parts of our implementation adapt components from the official repository (used under the Apache-2.0 license). Any errors remain our own.
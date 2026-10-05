# When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems ICML2026

This repository contains the official implementation of **EPC-AW**, a repair framework for  epistemic miscalibration in the planning phase of LLM-based multi-agent systems.

## Environment Setup

### Prerequisites

- **Python 3.11**

### Installation

Clone the repository and set up the virtual environment:

```bash
bash setup.sh
source .venv/bin/activate
```

## API & Model Configuration

EPC-AW relies on LLM APIs for planning, execution, and diagnosis.

Please configure the following variables in a `.env` file:

```
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE_URL=your_api_base_url
MODEL_NAME=your_llm_name
```

## Quick Start: EPC-AW Inference

To quickly experience EPC-AW on a toy example:

```
python quick_start.py
```

## Running Experiments

Supported Benchmarks

The repository supports the following QA benchmarks:

- Bamboogle (bamboogle)
- 2Wiki (2wiki)
- HotpotQA (hotpotqa)
- Musique (musique)
- GAIA (gaia)
- MedQA (medqa)

## Inference with Logging

Use inference_log.py to run and evaluate EPC-AW with detailed system logs:

```
python inference_log.py \
  --test_file test/gaia/data/data.json \
  --dir_name logs \
  --n 9 \
  --max_steps 10
```

Key Arguments

- --test_file:
Path to the benchmark dataset file.
- --dir_name:
Directory for saving logs and intermediate results.
- --n:
Number of candidate plans sampled during the planning phase.
- --max_steps:
Maximum execution steps per episode.

## Citation
```
@inproceedings{wang2026epcaw,
  title={When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems},
  author={Wang, Zehao and Jin, Shilong and Cao, Zhao and Wang, Lanjun},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

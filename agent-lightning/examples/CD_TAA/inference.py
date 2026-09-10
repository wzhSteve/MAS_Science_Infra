import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from MAS.epc_aw.solver import construct_solver
from MAS.epc_aw.models.ablation import resolve_ablation

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

llm_engine_name = os.getenv("MODEL_Name")  # 例如 "gpt-4o" 或 "qwen2.5"
enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool", "Screenshot_Tool", "Vision_OCR_Tool"]
tool_engine = [llm_engine_name] * len(enabled_tools)

# Set by inference_log / evaluate_multiple_testsets for a run.
_ACTIVE_ABLATION: Any = "full"
_OFFLINE_MEMORY_DIR: Optional[str] = "memory"
_TELEMETRY_DIR: Optional[str] = None
# True: load Cap/Inv but do not persist offline JSON after evolve.
# False: evolve + write tool_capability/invocation_memory.json back to disk.
_EVALUATION_MODE: bool = True


def set_run_config(
    ablation: Any = "full",
    offline_memory_dir: Optional[str] = "memory",
    telemetry_dir: Optional[str] = None,
    evaluation_mode: bool = True,
) -> None:
    global _ACTIVE_ABLATION, _OFFLINE_MEMORY_DIR, _TELEMETRY_DIR, _EVALUATION_MODE
    _ACTIVE_ABLATION = ablation
    _OFFLINE_MEMORY_DIR = offline_memory_dir
    _TELEMETRY_DIR = telemetry_dir
    _EVALUATION_MODE = evaluation_mode
    if telemetry_dir:
        Path(telemetry_dir).mkdir(parents=True, exist_ok=True)


def evaluate_with_llm(pred, gt, question):
    """
    Evaluates whether the predicted text (pred) semantically matches
    the ground truth (gt) using an LLM as a judge.
    """
    client = OpenAI(
            api_key= os.getenv("EVALUATE_MODEL_API_KEY"),
            base_url= os.getenv("EVALUATE_MODEL_URL")
        )
    prompt = f"""### Task: Semantic Consistency Evaluation
You are an expert evaluator. Your goal is to determine if the [Predicted Answer] is semantically consistent with the [Ground Truth Answer].

### Evaluation Criteria:
1. **Core Meaning:** Does the prediction convey the same essential information as the ground truth?
2. **Factuality:** Are the key entities, numbers, and logical steps identical in meaning, even if the wording differs?
3. **Completeness:** Does the prediction satisfy the requirements mentioned in the ground truth?
4. **Tone Independence:** Ignore minor differences in phrasing, formatting, or politeness.

### Data:
- [Question]: {question}
- [Ground Truth]: {gt}
- [Predicted Answer]: {pred}

### Output Requirement:
If the prediction is correct and matches the meaning of the ground truth, output exactly: True
If the prediction is incorrect, contains factual errors, output exactly: False

Do not provide any explanation or preamble. You must output a single word:
<True or False>.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("EVALUATE_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )

        result_text = response.choices[0].message.content.strip().lower()

        return result_text == True or "true" in str(result_text).lower()
    except Exception as e:
        print(f"LLM Evaluation Error: {e}")
        return False


def _dump_pid_telemetry(pid: Any, response: Optional[Dict[str, Any]], correct: Optional[bool], pred: Any, gt: Any) -> None:
    if not _TELEMETRY_DIR or not isinstance(response, dict):
        return
    out = {
        "pid": pid,
        "correct": correct,
        "prediction": pred,
        "ground_truth": gt,
        "ablation": response.get("ablation"),
        "layer_telemetry": response.get("layer_telemetry") or [],
        "task_id": response.get("task_id"),
        "scorable_output": response.get("scorable_output"),
    }
    path = Path(_TELEMETRY_DIR) / f"pid_{pid}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Telemetry] wrote {path}")


def evaluate_single_case(
    sample,
    n,
    max_steps,
    ablation=None,
    offline_memory_dir=None,
    evaluation_mode=None,
):
    """
    sample 格式示例:
    {
        "pid": 12,
        "query": "Who first landed on the Moon?",
        "answer": "Neil Armstrong",
        "choices": [...]
    }
    """
    pid = sample["pid"]
    query = sample["query"]
    ground_truth = sample["answer"]
    ablation_arg = ablation if ablation is not None else _ACTIVE_ABLATION
    mem_dir = offline_memory_dir if offline_memory_dir is not None else _OFFLINE_MEMORY_DIR
    eval_mode = _EVALUATION_MODE if evaluation_mode is None else bool(evaluation_mode)
    ablation_cfg = resolve_ablation(ablation_arg)
    if mem_dir:
        ablation_cfg.offline_memory_dir = mem_dir

    print(f"==================================================== [Evaluating] pid={pid} ====================================================")
    print(
        f"[Ablation] preset={ablation_cfg.preset} "
        f"intervention={ablation_cfg.enable_intervention} "
        f"capability={ablation_cfg.read_capability_memory} "
        f"invocation={ablation_cfg.read_invocation_memory} "
        f"offline_dir={ablation_cfg.memory_dir()} "
        f"evaluation_mode={eval_mode}"
    )
    response = None
    try:
        solver = construct_solver(
            llm_engine_name=llm_engine_name,
            enabled_tools=enabled_tools,
            tool_engine=tool_engine,
            n=n,
            temperature=0,
            max_steps=max_steps,
            evaluation_mode=eval_mode,
            ablation=ablation_cfg,
            offline_memory_dir=mem_dir,
        )
        response = solver.solve(query)
        pred = (response.get("scorable_output") or "").strip()
    except Exception as e:
        print(f"[Error] pid={pid} solver error: {e}")
        traceback.print_exc()
        _dump_pid_telemetry(pid, response if isinstance(response, dict) else {"error": str(e), "ablation": ablation_cfg.to_dict()}, False, None, ground_truth)
        return False, None, ground_truth

    evaluate = evaluate_with_llm(pred, ground_truth, query)
    correct = bool(evaluate)
    print(f"[Eval] pid={pid} Correct={correct} | Pred='{pred}' | GT='{ground_truth}'")
    _dump_pid_telemetry(pid, response, correct, pred, ground_truth)
    return correct, pred, ground_truth


def evaluate_single_case_assistantbench(sample):
    pid = sample["pid"]
    query = sample["query"]

    print(
        f"==================================================== "
        f"[Evaluating] pid={pid} "
        f"===================================================="
    )

    try:
        solver = construct_solver(
            llm_engine_name=llm_engine_name,
            enabled_tools=enabled_tools,
            tool_engine=tool_engine,
            n=9,
            temperature=0,
            max_steps=10,
            evaluation_mode=_EVALUATION_MODE,
            ablation=_ACTIVE_ABLATION,
            offline_memory_dir=_OFFLINE_MEMORY_DIR,
        )
        response = solver.solve(query)
        pred = response.get("scorable_output", "").strip()

        return {
            "id": str(pid),
            "answer": pred
        }

    except Exception as e:
        print(f"[Error] pid={pid} solver error: {e}")
        traceback.print_exc()
        return {
            "id": str(pid),
            "answer": ""
        }

def run_and_dump(data_path, output_path):
    """
    dataset: iterable of samples
    output_path: e.g. assistantbench_submission.jsonl
    """
    with open(data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in test_data:
            result = evaluate_single_case_assistantbench(sample)
            fout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )


def evaluate_testset(
    test_data,
    n=9,
    max_steps=10,
    name="testset",
    ablation=None,
    offline_memory_dir=None,
    evaluation_mode=None,
):
    total = len(test_data)
    correct = 0
    results_to_save = []

    print(f"\n=== Evaluating {name} (size={total}) ===")

    for sample in test_data:
        ok, pred, gt = evaluate_single_case(
            sample, n, max_steps,
            ablation=ablation,
            offline_memory_dir=offline_memory_dir,
            evaluation_mode=evaluation_mode,
        )
        if ok:
            correct += 1
        else:
            print(f"[Mismatch] pid={sample['pid']}")
            print(f"  Prediction: {pred}")
            print(f"  Ground Truth: {gt}")

        results_to_save.append({
            "question": sample,
            "ground_truth": gt,
            "prediction": pred
        })

    acc = correct / total if total > 0 else 0.0
    print(f">>> Accuracy for {name}: {acc:.4f}")

    try:
        with open(f"result_{name}.json", "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        print(f"--- Results successfully saved to result_{name}.json ---")
    except Exception as e:
        print(f"Error saving file: {e}")

    return correct, total, acc


def evaluate_multiple_testsets(
    data_path_list,
    start_pid,
    end_pid,
    n=4,
    max_steps=10,
    ablation="full",
    offline_memory_dir="memory",
    telemetry_dir=None,
    evaluation_mode: bool = True,
):
    total_correct = 0
    total_count = 0

    set_run_config(
        ablation=ablation,
        offline_memory_dir=offline_memory_dir,
        telemetry_dir=telemetry_dir,
        evaluation_mode=evaluation_mode,
    )
    print(
        f"[RunConfig] evaluation_mode={evaluation_mode} "
        f"(persist_offline={not evaluation_mode})"
    )

    for path in data_path_list:
        with open(path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        local_end = end_pid
        if local_end == -1:
            local_end = len(test_data)
        slice_data = test_data[start_pid: local_end]
        correct, count, acc = evaluate_testset(
            slice_data,
            n=n,
            max_steps=max_steps,
            name=os.path.basename(path),
            ablation=ablation,
            offline_memory_dir=offline_memory_dir,
            evaluation_mode=evaluation_mode,
        )
        total_correct += correct
        total_count += count

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n==============================")
    print(f"Overall Accuracy (ALL testsets): {overall_acc:.4f}")
    print(f"==============================")

    return overall_acc


if __name__ == "__main__":
    test_files = [
        "test/hotpotqa/data/data.json",
    ]
    start_pid = 99
    end_pid = -1
    evaluate_multiple_testsets(test_files, start_pid, end_pid)

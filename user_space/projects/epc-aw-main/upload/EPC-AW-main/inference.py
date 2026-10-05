import os
import json
from MAS.epc_aw.solver import construct_solver

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

llm_engine_name = os.getenv("MODEL_Name")  # 例如 "gpt-4o" 或 "qwen2.5"
enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool"] # , "Google_Search_Tool"
tool_engine = [llm_engine_name] * len(enabled_tools)

def evaluate_with_llm(pred, gt, question):
    """
    Evaluates whether the predicted text (pred) semantically matches 
    the ground truth (gt) using an LLM as a judge.
    """
    # Using a structured prompt for better reasoning
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
            temperature=0,  # Critical for consistency
            max_tokens=1024,
        )
        
        result_text = response.choices[0].message.content.strip().lower()
        
        # Logical check for "true" in response
        return result_text == True or "true" in str(result_text).lower()
    except Exception as e:
        print(f"LLM Evaluation Error: {e}")
        return False

def evaluate_single_case(sample, n, max_steps):
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
    print(f"==================================================== [Evaluating] pid={pid} ====================================================")
    try:
        solver = construct_solver(llm_engine_name=llm_engine_name, enabled_tools=enabled_tools, tool_engine=tool_engine, n=n, temperature=0.9, max_steps=max_steps)
        response = solver.solve(query)
        pred = response["direct_output"].strip()
    except Exception as e:
        print(f"[Error] pid={pid} solver error: {e}")
        return False, None, ground_truth

    evaluate = evaluate_with_llm(pred, ground_truth, query)
    if evaluate:
        correct = True
    else:
        correct = False
    print(f"[Eval] pid={pid} Correct={correct} | Pred='{pred}' | GT='{ground_truth}'")
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
            temperature=0.9,
            max_steps=10
        )
        response = solver.solve(query)
        pred = response["direct_output"].strip()

        return {
            "id": str(pid), 
            "answer": pred
        }

    except Exception as e:
        print(f"[Error] pid={pid} solver error: {e}")
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

import json

def evaluate_testset(test_data, n=9, max_steps=10, name="testset"):
    total = len(test_data)
    correct = 0
    results_to_save = []

    print(f"\n=== Evaluating {name} (size={total}) ===")

    for sample in test_data:
        ok, pred, gt = evaluate_single_case(sample, n, max_steps)
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


def evaluate_multiple_testsets(data_path_list, start_pid, end_pid, n=9, max_steps=10):
    total_correct = 0
    total_count = 0
    

    for path in data_path_list:
        with open(path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        if end_pid == -1:
            end_pid = len(test_data)
        test_data = test_data[start_pid: end_pid]
        correct, count, acc = evaluate_testset(test_data, n=n, max_steps=max_steps, name=os.path.basename(path))
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
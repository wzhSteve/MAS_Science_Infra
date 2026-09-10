import concurrent.futures
import os, re
import json
import argparse
import tqdm
import sys
from pydantic import BaseModel
from MAS.epc_aw.engine.openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import ResultAnalyzer

class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

class BinaryAnswerVerification(BaseModel):
    true_false: bool

class ResultScorer:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine or ChatOpenAI(model_string="gpt-4o", is_multimodal=False, enable_cache=True)
        print(f"\nLocal OpenAI engine {self.llm_engine.model_string} initialized.\n")

    def answer_verification(self, question, response, correct_answer):
        all_matches = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
        if all_matches:
            response = all_matches[-1].strip()
        else:
            response = response

        query_prompt = f"""
Given a multiple-choice Question, a Model Response, and its Correct Answer, determine whether the Model's prediction is correct.

The prediction is correct only if it **exactly matches** the correct choice letter (e.g., "A", "B", "C", or "D") after necessary normalization. Follow these instructions carefully:

1. If the Model Response is a number (e.g., "2", "3", etc.), map it to the corresponding option letter based on its order in the Question (e.g., 1 → A, 2 → B, etc.).
2. Ignore irrelevant text, explanations, or format differences. Extract the core predicted answer.
3. Compare the final normalized response with the Correct Answer letter.

Question: {question}
Model response: {response}
Correct answer: {correct_answer}

Response Format:
<analysis>: First extract the mathematical answers, then explain the comparison
<true_false>: Return "True" only for exact matches, otherwise "False"
        """

        verification = self.llm_engine(query_prompt, response_format=AnswerVerification)

        analysis = verification.analysis.strip()
        true_false = verification.true_false

        return analysis, true_false

    def score_results(self, results, max_workers=10):
        correct = 0

        def process_single_result(pid_data):
            pid, question_data = pid_data
            question = question_data["question"]
            response = question_data["response"]
            correct_answer = question_data["correct_answer"]
            analysis, true_false = self.answer_verification(question, response, correct_answer)
            return pid, analysis, true_false

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_result, (pid, data))
                      for pid, data in results.items()]

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                  total=len(futures),
                                  desc="Scoring results"):
                pid, analysis, true_false = future.result()
                correct += 1 if true_false else 0
                results[pid].update({
                    "stepwise_analysis": analysis,
                    "true_false": true_false
                })

        return results, correct


def load_data(data_file, result_dir, response_type):
    # Load the benchmark data
    with open(data_file, 'r') as f:
        # convert the benchmark data to a dictionary
        benchmark_data = {data["pid"]: data for data in json.load(f)}

    # Load the results
    results = {}
    for file in os.listdir(result_dir):
        if file.endswith(".json") and "output_" in file:
            file_path = os.path.join(result_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                # Get the index of the result
                index = file.replace(".json", "").replace("output_", "") # "0", "1", "2", ...
                # Try using index as string first, if not found then try as int
                try:
                    pid = int(index)
                    if pid not in benchmark_data:
                        pid = str(int(index))
                except (ValueError, KeyError):
                    pid = index
                assert result["pid"] == benchmark_data[pid]["pid"]

                # Save the results
                results[pid] = benchmark_data[pid]
                assert response_type in result
                results[pid]["response"] = result[response_type]
                results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
                # print(f"successfully read: {file}")

            except json.JSONDecodeError as e:
                print(f"JSON decode error, cannot parse the file: {file}, Error message: {e}")
            except Exception as e:
                print(f"Unknown error: {file}, Error message: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Universal script to extract and score results from benchmark data for all tasks")
    parser.add_argument("--task_name", type=str, required=True,
                        help="The name of the task (e.g., aime24, bamboogle, gaia, gameof24)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="The file containing the benchmark data (default: {task_name}/data/test.json)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="The directory containing the results (default: {task_name}/results/{exp_name})")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="The experiment name (used to construct result_dir if not specified)")
    parser.add_argument("--output_file", type=str, default="final_results.json",
                        help="The file to save the extracted results")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="The directory containing the logs")
    parser.add_argument("--response_type", type=str, default="direct_output",
                        choices=["final_output", "direct_output", "base_response"],
                        help="The type of response to extract from the results")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="The maximum number of workers to use")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get the base directory (tasks folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, args.task_name)

    # Set default paths if not provided
    if args.data_file is None:
        args.data_file = os.path.join(task_dir, "data", "test.json")

    if args.result_dir is None:
        if args.exp_name is None:
            raise ValueError("Either --result_dir or --exp_name must be specified")
        args.result_dir = os.path.join(task_dir, "results", args.exp_name)

    # Validate paths
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not os.path.exists(args.result_dir):
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")

    # Load and print the arguments
    print("#"*50)
    print(f"Task: {args.task_name}")
    print(f"Arguments: {args}")
    for arg, value in args.__dict__.items():
        print(f"# {arg}: {value}")
    print("#"*50)

    scorer = ResultScorer()
    analyzer = ResultAnalyzer()

    # Load the results
    results = load_data(args.data_file, args.result_dir, args.response_type)

    # Score the results
    results, correct = scorer.score_results(results, max_workers=args.max_workers)

    # Calculate accuracy and wrong answers
    acc = round(correct / len(results) * 100, 2)
    print(f"\nAccuracy: {acc}% ({correct}/{len(results)})")

    # Save detailed results
    output_file = os.path.join(args.result_dir, args.output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

    # Calculate wrong answers
    wrong_pids = [pid for pid, data in results.items() if not data["true_false"]]
    wrong_pids = sorted(wrong_pids, key=lambda x: int(x))
    wrong_indices = [int(pid) for pid in wrong_pids]
    print(f"Wrong PIDs: {wrong_pids}")
    print(f"Wrong Indices: {wrong_indices}")

    scores = {
        "correct": correct,
        "total": len(results),
        "accuracy": acc,
        "wrong_pids": wrong_pids,
        "wrong_indices": wrong_indices
    }

    # Calculate additional statistics if log directory is provided
    log_dir = args.log_dir or args.result_dir.replace("results", "logs")
    if os.path.exists(log_dir):

        if args.response_type == "base_response":
            print("Base response is not supported for scoring.")
            print("Exited.\n")
            exit()

         # Calculate the average time and steps
        step_stats = analyzer.calculate_time_steps(log_dir)
        print(f"\nStep stats:")
        for key, value in step_stats.items():
            print(f"- {key}: \t{value}")

        # Calculate the usage of tools
        tool_usage = analyzer.calculate_tool_usage(args.result_dir)
        print(f"\nTool usage:")
        for tool, ratio in tool_usage.items():
            print(f"- {tool}: \t{ratio}")

        # Update the scores
        scores.update({
            "step_stats": step_stats,
            "tool_usage": tool_usage
        })


    # Save the scores
    score_file = os.path.join(args.result_dir, f"final_scores_{args.response_type}.json")
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)
        print(f"Scores saved to {score_file}")


if __name__ == "__main__":
    main()
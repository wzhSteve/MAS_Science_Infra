import sys
import datetime
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test files.")
    parser.add_argument(
        "--test_file",
        action="append",
        required=True,
        help="Path to a test data file. Can be specified multiple times."
    )
    parser.add_argument(
        "--start_pid",
        type=int,
        default=0,
        help="Start PID"
    )
    parser.add_argument(
        "--end_pid",
        type=int,
        default=-1,
        help=""
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="sample number"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="max_steps"
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default="logs",
        help="log directory"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation preset: full|no_intervention|no_capability|no_invocation|no_memory|product_no_l2a|...",
    )
    parser.add_argument(
        "--offline_memory_dir",
        type=str,
        default="memory",
        help="Offline/online memory root (needed under evaluation_mode to load Cap/Inv).",
    )
    parser.add_argument(
        "--evaluation_mode",
        action="store_true",
        default=True,
        help="Eval mode: load Cap/Inv but do NOT persist offline JSON after evolve (default).",
    )
    parser.add_argument(
        "--no_evaluation_mode",
        action="store_true",
        help="Disable eval mode: evolve + write tool_capability/invocation_memory.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_files = args.test_file
    start_pid = args.start_pid
    end_pid = args.end_pid
    n = args.n
    max_steps = args.max_steps
    dir_name = args.dir_name
    ablation = args.ablation
    offline_memory_dir = args.offline_memory_dir
    evaluation_mode = False if args.no_evaluation_mode else args.evaluation_mode

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    for test_file in test_files:
        test_data = test_file.split("/")[1]
        preset_tag = ablation.replace("/", "_")
        log_filename = f"./{dir_name}/test_{test_data}_{timestamp}_{preset_tag}.log"
        telemetry_dir = f"./{dir_name}/ablation_runs/{preset_tag}_{timestamp}"
        Path(telemetry_dir).mkdir(parents=True, exist_ok=True)

        log_file = open(log_filename, "w")
        print(
            f"[inference_log] writing {log_filename} telemetry→{telemetry_dir} "
            f"ablation={ablation} evaluation_mode={evaluation_mode} "
            f"persist_offline={not evaluation_mode}"
        )

        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        if "assistant" not in test_data:
            try:
                import inference
                inference.evaluate_multiple_testsets(
                    [test_file],
                    start_pid,
                    end_pid=end_pid,
                    n=n,
                    max_steps=max_steps,
                    ablation=ablation,
                    offline_memory_dir=offline_memory_dir,
                    telemetry_dir=telemetry_dir,
                    evaluation_mode=evaluation_mode,
                )
            finally:
                sys.stdout = original_stdout
                log_file.close()
        else:
            try:
                import inference
                inference.set_run_config(
                    ablation=ablation,
                    offline_memory_dir=offline_memory_dir,
                    telemetry_dir=telemetry_dir,
                    evaluation_mode=evaluation_mode,
                )
                inference.run_and_dump(
                    data_path=test_file,
                    output_path=f"./test/assistantbench/output.jsonl"
                )
            finally:
                sys.stdout = original_stdout
                log_file.close()


if __name__ == "__main__":
    main()

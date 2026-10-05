import sys
import datetime
import argparse

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
        default=9,
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
    return parser.parse_args()


def main():
    args = parse_args()
    test_files = args.test_file
    start_pid = args.start_pid
    end_pid = args.end_pid
    n = args.n
    max_steps = args.max_steps
    dir_name = args.dir_name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for test_file in test_files:
        test_data = test_file.split("/")[1]

        log_filename = f"./{dir_name}/test_{test_data}_{timestamp}.log"
        log_file = open(log_filename, "w")

        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        if "assistant" not in test_data:
            try:
                import inference
                inference.evaluate_multiple_testsets([test_file], start_pid, end_pid=-1, n=n, max_steps=max_steps)
            finally:
                sys.stdout = original_stdout
                log_file.close()
        else:
            try:
                import inference
                inference.run_and_dump(
                    data_path=test_file,
                    output_path=f"./test/assistantbench/output.jsonl"
                )
            finally:
                sys.stdout = original_stdout
                log_file.close()
            


if __name__ == "__main__":
    main()

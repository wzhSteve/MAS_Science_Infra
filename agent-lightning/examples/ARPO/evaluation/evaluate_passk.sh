#!/bin/bash

declare -A task_map=(
    ["aime24"]="math"
    ["aime25"]="math"
    ["math500"]="math"
    ["gsm8k"]="math"
    ["math"]="math"
    ["webwalker"]="qa"
    ["hotpotqa"]="qa"
    ["2wiki"]="qa"
    ["bamboogle"]="qa"
    ["musique"]="qa"
    ["hle"]="qa"
    ["gaia"]="qa"
    ["SimpleQA"]="qa"
    ["xbench"]="qa"
)

OUTPUT_DIR="your_result_directory"


find "$OUTPUT_DIR" -type f -name '*_output_*.json' | while read -r file_path; do
    filename=$(basename "$file_path")
    dataset_name="${filename%%_output_*}"
    task="${task_map[$dataset_name]}"

    if [ -z "$task" ]; then
        echo "Unknown dataset: $dataset_name. Skipping..."
        continue
    fi

    echo "Evaluating $dataset_name (task: $task)..."

    OUTPUT_PATH="$file_path" TASK="$task" bash evaluate.sh

    echo "Finished evaluating $dataset_name"
    echo "-----------------------------------"
done

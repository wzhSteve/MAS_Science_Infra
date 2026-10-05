import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
import fire


# Instruction template to prompt model for structured answer output
OUTPUT_FORMAT = (
    "When ready, output the final answer enclosed in <answer> and </answer> tags. "
    "Do not generate any content after the </answer> tag."
)


def process_aime_dataset(dataset):
    """
    Processes the AIME_2024 dataset into a standardized format.

    Each example includes:
    - id: unique integer index
    - question: problem text + instruction prompt
    - chain: placeholder for reasoning trace (to be filled later)
    - result: ground truth answer as string
    - source: identifier for dataset origin
    - extra_info: metadata stored as JSON string (original problem, index, etc.)

    Args:
        dataset (Dataset): Hugging Face Dataset object containing 'Problem' and 'Answer'.

    Returns:
        Dataset: Processed dataset ready for saving or further processing.
    """
    processed_data = []

    for idx, item in enumerate(dataset):
        problem = item.get("Problem", "").strip()
        answer = item.get("Answer", "")

        # Combine problem with output formatting instruction
        full_question = f"{problem}\n\n{OUTPUT_FORMAT}" if problem else OUTPUT_FORMAT

        # Build structured entry
        entry = {
            "id": idx,
            "question": full_question,
            "chain": "",  # Placeholder; will be populated during inference
            "result": str(answer).strip(),
            "source": "aime2024",
            "extra_info": {
                "ground_truth": str(answer).strip(),
                "idx": idx,
                "original_problem": problem,
            }
        }
        processed_data.append(entry)

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Serialize 'extra_info' dict into JSON strings for safe Parquet storage, WARNING: this may cause a failure in ecxtract data as str!!!
    # df["extra_info"] = df["extra_info"].apply(json.dumps, ensure_ascii=False)

    # Return as Hugging Face Dataset (required for .to_parquet())
    return Dataset.from_pandas(df, preserve_index=False)


def main(output_dir="./data/val"):
    """
    Main function to load, process, and save the AIME_2024 dataset in Parquet format.

    Steps:
    1. Load the 'train' split of AIME_2024 dataset.
    2. Process each sample into a unified schema.
    3. Shuffle the dataset with a fixed seed.
    4. Re-index IDs to ensure continuity after shuffle.
    5. Save the final dataset to a Parquet file.

    Args:
        output_dir (str): Directory path where the Parquet file will be saved.
    """
    print("--- Loading AIME_2024 train dataset ---")
    try:
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        print(f"✅ Loaded {len(dataset)} examples from AIME_2024.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    print("\n--- Processing dataset into standard format ---")
    processed_dataset = process_aime_dataset(dataset)
    print(f"✅ Processed {len(processed_dataset)} records.")

    # Optional: Shuffle for training stability
    print("--- Shuffling dataset with seed=42 ---")
    shuffled_dataset = processed_dataset.shuffle(seed=42)

    # Reassign IDs after shuffling
    print("--- Re-indexing 'id' field ---")
    final_dataset = shuffled_dataset.map(
        lambda example, idx: {"id": idx},
        with_indices=True,
        desc="Re-indexing"
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "aime24.parquet")

    print(f"\n--- Saving dataset to Parquet format: {output_path} ---")
    final_dataset.to_parquet(output_path)
    print(f"✅ Successfully saved {len(final_dataset)} records to {output_path}")

    # Display one sample for verification
    sample = final_dataset[0]
    print("\n🔍 Sample record:")
    for k, v in sample.items():
        if k != "extra_info":
            print(f"  {k}: {v}")
    print(f"  extra_info (decoded): {sample['extra_info']}")


if __name__ == "__main__":
    fire.Fire(main)


"""
python data/aime24_data.py
"""
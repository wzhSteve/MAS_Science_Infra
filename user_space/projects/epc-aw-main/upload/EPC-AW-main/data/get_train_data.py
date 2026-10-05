import os
import datasets
import pandas as pd
import numpy as np
import fire
from datasets import Dataset, concatenate_datasets

# This function processes the 'golden_answers' field from the nq dataset.
def process_golden_answers(golden_answers, to_string=True):
    """
    Processes 'golden_answers' field and returns a STRING (comma-separated) or empty string.
    Handles: list, tuple, numpy array (1D or scalar), string, number, None, etc.
    """
    items = []

    # Case 1: numpy array
    if isinstance(golden_answers, np.ndarray):
        items = [str(item) for item in golden_answers.flatten() if item is not None and pd.notna(item)]
    # Case 2: list or tuple
    elif isinstance(golden_answers, (list, tuple)):
        items = [str(item) for item in golden_answers if item is not None and pd.notna(item)]
    # Case 3: string
    elif isinstance(golden_answers, str):
        cleaned = golden_answers.strip()
        if cleaned:
            items = [cleaned]
    # Case 4: scalar number (including np.number)
    elif isinstance(golden_answers, (int, float, np.generic)):
        if not pd.isna(golden_answers):
            items = [str(golden_answers).strip()]
    # Case 5: None or empty
    elif golden_answers is None or (isinstance(golden_answers, str) and not golden_answers.strip()):
        items = []
    # Fallback: try string conversion
    else:
        s = str(golden_answers).strip()
        if s and s != "nan":
            items = [s]

    if to_string:
        return "; ".join(items) if items else ""
    else:
        return items

# This function processes the nq dataset to a standard format.
def process_nq_dataset(dataset):
    """
    Processes the NQ dataset to a unified schema.
    """
    processed_data = []
    for idx, item in enumerate(dataset):
        # Clean question
        question = item.get("question", "").strip()
        if question and not question.endswith('?'):
            question += '?'

        # Process golden_answers: convert to comma-separated string
        golden_answers = item.get("golden_answers", [])
        final_result = process_golden_answers(golden_answers, to_string=True)

        # Build entry with a temporary ID for later re-indexing
        new_entry = {
            'id': idx, # Temporary ID
            'question': question,
            'chain': "",
            'result': str(final_result),
            'source': "nq",
            'extra_info': {
                'ground_truth': str(final_result),
                'idx': idx
            }
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return Dataset.from_pandas(df, preserve_index=False)

# This function processes the mathhard dataset to a standard format.
def process_math_dataset(dataset):
    """
    Processes the DeepMath-103K dataset to a unified schema.
    """
    # Use a mapping function to process the dataset
    def map_fn(example, idx):
        question = example.pop('question') if 'question' in example else example.pop('Problem')
        solution = example.pop('final_answer') if 'final_answer' in example else example.pop('Answer')
        
        # Build entry with a temporary ID for later re-indexing
        data = {
            "id": idx, # Temporary ID
            "question": question,
            "chain": "", # The 'chain' field is added for schema consistency
            "result": str(solution),
            "source": "mathhard",
            "extra_info": {
                'ground_truth': str(solution),
                'idx': idx,
            }
        }
        return data

    return dataset.map(function=map_fn, with_indices=True, remove_columns=dataset.column_names)

def main(output_dir='./data/train'):
    """
    Loads, processes, and combines the NQ and MathHard train datasets.
    
    Args:
        output_dir (str): The directory to save the final combined dataset.
    """
    print("--- 1. Loading and processing NQ train dataset ---")
    try:
        nq_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', split='train')
        processed_nq = process_nq_dataset(nq_dataset)
        print(f"✅ Processed {len(processed_nq)} records from NQ.")
    except Exception as e:
        print(f"❌ Failed to process NQ dataset: {e}")
        return

    print("\n--- 2. Loading and processing MathHard train dataset ---")
    try:
        math_dataset = datasets.load_dataset('zwhe99/DeepMath-103K', split='train')
        processed_math = process_math_dataset(math_dataset)
        print(f"✅ Processed {len(processed_math)} records from MathHard.")
    except Exception as e:
        print(f"❌ Failed to process MathHard dataset: {e}")
        return

    print("\n--- 3. Concatenating datasets ---")
    combined_dataset = concatenate_datasets([processed_nq, processed_math])
    
    # Add a shuffle step here
    print("--- 4. Shuffling combined dataset ---")
    shuffled_combined = combined_dataset.shuffle(seed=42)

    print("--- 5. Re-indexing shuffled dataset to ensure unique IDs ---")
    final_combined = shuffled_combined.map(lambda example, idx: {'id': idx}, with_indices=True)
    
    print(f"✅ Successfully combined and shuffled datasets. Total records: {len(final_combined)}")
    print("Example of a combined record:")
    print(final_combined[0])
    
    # --- 6. Saving the combined dataset ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_train.parquet')
    
    print(f"\n--- 6. Saving the combined dataset to {output_path} ---")
    final_combined.to_parquet(output_path)
    print(f"✅ Saved {len(final_combined)} records to {output_path}")

if __name__ == '__main__':
    fire.Fire(main)

# python get_train_data.py
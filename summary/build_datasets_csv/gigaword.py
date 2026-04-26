import os
from pathlib import Path

import pandas as pd
from datasets import Dataset


def main():
    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/gigaword")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train data
    train_dataset = Dataset.from_file("/home/wangwenqiang/huggingface/data/gigaword_tiny/gigaword_tiny-train.arrow")
    train_records = []
    for item in train_dataset:
        text = item["document"]
        label = item["summary"]
        train_records.append({"text": text, "label": label})

    train_df = pd.DataFrame(train_records)
    train_output_path = output_dir / "gigaword_train.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"Saved {train_output_path} ({len(train_df)} rows)")

    # Load validation data
    validation_dataset = Dataset.from_file("/home/wangwenqiang/huggingface/data/gigaword_tiny/gigaword_tiny-validation.arrow")
    test_records = []
    for item in validation_dataset:
        text = item["document"]
        label = item["summary"]
        test_records.append({"text": text, "label": label})

    test_df = pd.DataFrame(test_records)
    test_output_path = output_dir / "gigaword_test.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"Saved {test_output_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()

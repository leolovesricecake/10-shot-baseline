import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk


def main():
    dataset_name = "/home/wangwenqiang/huggingface/data/stsb"
    try:
        stsb = load_from_disk(dataset_name)
    except Exception:
        stsb = load_dataset("stsb")

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/stsb")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train split: all from train
    train_records = []
    raw_train_dataset = stsb["train"]
    for i in range(len(raw_train_dataset)):
        text = f"text1: {raw_train_dataset[i]['text1']} text2: {raw_train_dataset[i]['text2']}"
        label = raw_train_dataset[i]["label"]
        train_records.append({"text": text, "label": label})

    train_df = pd.DataFrame(train_records)
    train_output_path = output_dir / "stsb_train.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"Saved {train_output_path} ({len(train_df)} rows)")

    # Test split: last 75% of validation
    raw_test_dataset = stsb["validation"]
    test_start = int(0.25 * len(raw_test_dataset))
    test_records = []
    for i in range(test_start, len(raw_test_dataset)):
        text = f"text1: {raw_test_dataset[i]['text1']} text2: {raw_test_dataset[i]['text2']}"
        label = raw_test_dataset[i]["label"]
        test_records.append({"text": text, "label": label})

    test_df = pd.DataFrame(test_records)
    test_output_path = output_dir / "stsb_test.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"Saved {test_output_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()

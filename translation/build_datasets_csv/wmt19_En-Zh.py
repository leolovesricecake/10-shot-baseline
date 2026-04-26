import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main():
    # Load wmt19-en-zh dataset from parquet files
    dataset = load_dataset('parquet', data_files={
        'train': r'/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/train.parquet',
        'validation': r'/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/validation.parquet'
    })

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/wmt19_En-Zh")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train split: first 3000 from train
    train_records = []
    raw_train_dataset = dataset["train"]
    for i in range(3000):
        text = raw_train_dataset[i]['translation']['en']
        label = raw_train_dataset[i]['translation']['zh']
        train_records.append({"text": text, "label": label})

    train_df = pd.DataFrame(train_records)
    train_output_path = output_dir / "wmt19_En-Zh_train.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"Saved {train_output_path} ({len(train_df)} rows)")

    # Test split: first 20% of validation
    test_records = []
    raw_test_dataset = dataset["validation"]
    test_end = int(0.2 * len(raw_test_dataset))
    for i in range(test_end):
        text = raw_test_dataset[i]['translation']['en']
        label = raw_test_dataset[i]['translation']['zh']
        test_records.append({"text": text, "label": label})

    test_df = pd.DataFrame(test_records)
    test_output_path = output_dir / "wmt19_En-Zh_test.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"Saved {test_output_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()

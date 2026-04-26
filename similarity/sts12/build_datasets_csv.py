import os
from pathlib import Path

import pandas as pd


def main():
    # Load data from existing CSV files
    train_csv_path = "/mnt/huawei/wwq/project/ICL-in-cotext-order/similarity/sts12/sts12_train_4/wyj/sts12_train_first_300.csv"
    test_csv_path = "/mnt/huawei/wwq/project/ICL-in-cotext-order/similarity/sts12/sts12_train_4/wyj/sts12_test_first_700.csv"

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/sts12")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train data
    train_records = []
    for _, row in train_df.iterrows():
        text = f"text1: {row['sentence1']} text2: {row['sentence2']}"
        label = row["score"]
        train_records.append({"text": text, "label": label})

    train_output_df = pd.DataFrame(train_records)
    train_output_path = output_dir / "sts12_train.csv"
    train_output_df.to_csv(train_output_path, index=False)
    print(f"Saved {train_output_path} ({len(train_output_df)} rows)")

    # Process test data (last 75% as per original logic)
    test_start = int(0.25 * len(test_df))
    test_records = []
    for _, row in test_df.iloc[test_start:].iterrows():
        text = f"text1: {row['sentence1']} text2: {row['sentence2']}"
        label = row["score"]
        test_records.append({"text": text, "label": label})

    test_output_df = pd.DataFrame(test_records)
    test_output_path = output_dir / "sts12_test.csv"
    test_output_df.to_csv(test_output_path, index=False)
    print(f"Saved {test_output_path} ({len(test_output_df)} rows)")


if __name__ == "__main__":
    main()

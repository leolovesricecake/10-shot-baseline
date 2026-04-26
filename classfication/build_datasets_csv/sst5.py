import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk


def main():
    dataset_path = "/home/wangwenqiang/huggingface/data/sst5"
    try:
        sst5 = load_from_disk(dataset_path)
    except Exception:
        sst5 = load_dataset("sst5")

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/sst5")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        if split not in sst5:
            raise ValueError(f"Dataset has no split '{split}'")

        df = sst5[split].to_pandas()
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")

        # Put text and label first, keep other columns if available
        cols = [c for c in ["text", "label"] if c in df.columns] + [c for c in df.columns if c not in {"text", "label"}]
        df = df[cols]

        output_path = output_dir / f"sst5_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

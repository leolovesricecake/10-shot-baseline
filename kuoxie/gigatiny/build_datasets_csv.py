import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk


def main():
    dataset_path = "/home/wangwenqiang/huggingface/data/gigaword_tiny"
    try:
        gigatiny = load_from_disk(dataset_path)
    except Exception:
        gigatiny = load_dataset("gigaword_tiny")

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/gigatiny")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        if split not in gigatiny:
            raise ValueError(f"Dataset has no split '{split}'")

        records = []
        for item in gigatiny[split]:
            records.append(
                {
                    "text": item.get("summary", ""),
                    "label": item.get("document", ""),
                }
            )

        df = pd.DataFrame(records)
        output_path = output_dir / f"gigatiny_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

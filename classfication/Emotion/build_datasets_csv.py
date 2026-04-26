from pathlib import Path
import pandas as pd
from datasets import load_dataset, load_from_disk


def main():
    dataset_path = "/home/wangwenqiang/huggingface/data/emotion"
    try:
        emotion = load_from_disk(dataset_path)
    except Exception:
        emotion = load_dataset("emotion")

    train_split = emotion["train"]
    test_split = emotion["test"]

    output_dir = Path("/mnt/huawei/wwq/project/10-shot-baseline/datasets/emotion")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "text": [item["text"] for item in train_split],
            "label": [item["label"] for item in train_split],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": [item["text"] for item in test_split],
            "label": [item["label"] for item in test_split],
        }
    )

    train_df.to_csv(output_dir / "emotion_train.csv", index=False)
    test_df.to_csv(output_dir / "emotion_test.csv", index=False)


if __name__ == "__main__":
    main()

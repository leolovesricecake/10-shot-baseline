import os
import io
import re
import time
import base64
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from openai import OpenAI

# =========================================================
# 0. 基本配置
# =========================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 你的检索结果 CSV 文件路径
CSV_PATH = "/mnt/huawei/ymb/mm/baseline/dataset/CIFAR10/result/cifar10_fixed_test_to_train_top15.csv"

# 输出结果文件
OUTPUT_CSV = "./results/qwen4b_cifar10_icl_predictions.csv"

# 使用前几个检索样本做 ICL
NUM_SHOTS = 10

# OpenAI-compatible 接口配置（按你的示例）
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "/mnt/huawei/ymb/model/qwen3.5-4b/model"

# 是否只跑前 N 条，调试时很有用；正式跑设为 None
MAX_SAMPLES = 1000

# =========================================================
# 1. CIFAR-10 类别映射
# =========================================================
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

LABEL2ID = {name: i for i, name in enumerate(CIFAR10_LABELS)}

# =========================================================
# 2. OpenAI 客户端
# =========================================================
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# =========================================================
# 3. 加载数据集
# =========================================================
train_ds = load_dataset("uoft-cs/cifar10", split="train")
test_ds = load_dataset("uoft-cs/cifar10", split="test")

IMG_KEY_TRAIN = "img" if "img" in train_ds.column_names else "image"
IMG_KEY_TEST = "img" if "img" in test_ds.column_names else "image"

# =========================================================
# 4. 工具函数：PIL 图转 base64 data URL
# =========================================================
def pil_to_data_url(img: Image.Image, image_format="PNG") -> str:
    """
    将 PIL.Image 转成 data:image/...;base64,... 格式，
    方便直接传给 OpenAI-compatible 接口。
    """
    buffer = io.BytesIO()
    img.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{image_b64}"

# =========================================================
# 5. 工具函数：从模型输出中解析类别
# =========================================================
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-zA-Z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_prediction_to_label(text: str):
    """
    尝试把模型输出解析成 CIFAR-10 的 10 个类别之一。
    """
    if text is None:
        return None

    raw = text.strip()
    norm = normalize_text(raw)

    # 1) 先精确匹配完整类别名
    for label in CIFAR10_LABELS:
        if norm == label:
            return label

    # 2) 再看文本中是否包含类别词
    hits = []
    for label in CIFAR10_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", norm):
            hits.append(label)

    if len(hits) == 1:
        return hits[0]

    # 3) 处理常见别名
    alias_map = {
        "car": "automobile",
        "cars": "automobile",
        "auto": "automobile",
        "vehicle": "automobile",
        "plane": "airplane",
        "aircraft": "airplane",
        "aeroplane": "airplane",
    }
    for alias, target in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", norm):
            return target

    return None

# =========================================================
# 6. 构造多图 ICL prompt
# =========================================================
def build_messages_for_one_example(test_idx: int, retrieved_train_indices: list):
    """
    构造一条请求：
    - 前 10 张 train 图作为上下文示例，每张图配真实标签
    - 最后一张 test 图作为待分类图
    """
    content = []

    # ---- 总指令 ----
    instruction = (
        "You are doing CIFAR-10 image classification.\n"
        "There are exactly 10 possible classes:\n"
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.\n\n"
        "Classify the following test image and output exactly one of the 10 class names.\n"
        "Do not explain. Do not output any extra words or punctuation.\n"
        "let's think step by step\n"
    )
    content.append({"type": "text", "text": instruction})

    # ---- 待预测 test 图 ----
    test_item = test_ds[int(test_idx)]
    test_img = test_item[IMG_KEY_TEST]
    test_data_url = pil_to_data_url(test_img)

    content.append({
        "type": "text",
        "text": "Now classify the following test image. Output only the label."
    })
    content.append({
        "type": "image_url",
        "image_url": {"url": test_data_url}
    })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages

# =========================================================
# 7. 调用模型
# =========================================================
def predict_one(test_idx: int, retrieved_train_indices: list):
    messages = build_messages_for_one_example(test_idx, retrieved_train_indices)
    infer_start = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )

    infer_elapsed = time.time() - infer_start
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    text = response.choices[0].message.content
    pred_label = parse_prediction_to_label(text)

    return text, pred_label, prompt_tokens, completion_tokens, total_tokens, infer_elapsed

# =========================================================
# 8. 读取 CSV
# =========================================================
df = pd.read_csv(CSV_PATH)

# 如果有多余索引列，比如 unnamed，就去掉
drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
if len(drop_cols) > 0:
    df = df.drop(columns=drop_cols)

# 兼容列名检查
required_cols = ["test_idx"] + [f"train_idx_{i}" for i in range(1, 11)]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少必要列: {col}")

if MAX_SAMPLES is not None:
    df = df.iloc[:MAX_SAMPLES].copy()

# =========================================================
# 9. 主循环：逐条做 ICL 分类
# =========================================================
results = []
correct = 0
total_valid = 0
total_prompt_tokens_used = 0
total_completion_tokens_used = 0
total_tokens_used = 0

start_time = time.time()

for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Running CIFAR10 ICL"):
    test_idx = int(row["test_idx"])
    retrieved_train_indices = [int(row[f"train_idx_{i}"]) for i in range(1, NUM_SHOTS + 1)]

    # CIFAR-10 test 真值
    gt_label_id = int(test_ds[test_idx]["label"])
    gt_label_name = CIFAR10_LABELS[gt_label_id]

    try:
        raw_text, pred_label, prompt_tokens, completion_tokens, used_tokens, infer_elapsed = predict_one(
            test_idx, retrieved_train_indices
        )

        is_correct = int(pred_label == gt_label_name) if pred_label is not None else 0
        total_valid += 1
        correct += is_correct
        total_prompt_tokens_used += prompt_tokens
        total_completion_tokens_used += completion_tokens
        total_tokens_used += used_tokens

        results.append({
            "row_id": int(row_id),
            "test_idx": test_idx,
            "gt_label_id": gt_label_id,
            "gt_label_name": gt_label_name,
            "pred_raw_text": raw_text,
            "pred_label_name": pred_label if pred_label is not None else "",
            "pred_label_id": LABEL2ID[pred_label] if pred_label is not None else -1,
            "is_correct": is_correct,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": used_tokens,
            "inference_time_seconds": infer_elapsed,
            **{f"train_idx_{i}": retrieved_train_indices[i-1] for i in range(1, NUM_SHOTS + 1)},
        })

    except Exception as e:
        results.append({
            "row_id": int(row_id),
            "test_idx": test_idx,
            "gt_label_id": gt_label_id,
            "gt_label_name": gt_label_name,
            "pred_raw_text": f"ERROR: {str(e)}",
            "pred_label_name": "",
            "pred_label_id": -1,
            "is_correct": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "inference_time_seconds": None,
            **{f"train_idx_{i}": retrieved_train_indices[i-1] for i in range(1, NUM_SHOTS + 1)},
        })

# =========================================================
# 10. 保存结果
# =========================================================
result_df = pd.DataFrame(results)
elapsed = time.time() - start_time
acc = correct / total_valid if total_valid > 0 else 0.0
num_samples = len(results)
avg_time = elapsed / num_samples if num_samples > 0 else 0.0
avg_prompt_tokens = total_prompt_tokens_used / num_samples if num_samples > 0 else 0.0
avg_completion_tokens = total_completion_tokens_used / num_samples if num_samples > 0 else 0.0
avg_tokens = total_tokens_used / num_samples if num_samples > 0 else 0.0

result_df["summary_total_prompt_tokens"] = total_prompt_tokens_used
result_df["summary_total_completion_tokens"] = total_completion_tokens_used
result_df["summary_total_tokens"] = total_tokens_used
result_df["summary_avg_prompt_tokens"] = avg_prompt_tokens
result_df["summary_avg_completion_tokens"] = avg_completion_tokens
result_df["summary_avg_total_tokens"] = avg_tokens
result_df["summary_total_inference_time_seconds"] = elapsed
result_df["summary_avg_inference_time_seconds"] = avg_time
result_df["summary_accuracy"] = acc

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Finished. Results saved to: {OUTPUT_CSV}")
print(f"Total valid samples: {total_valid}")
print(f"Accuracy: {acc:.4f}")
print(f"Total inference time: {elapsed:.2f} seconds")
print(f"Average inference time: {avg_time:.4f} seconds/sample")
print(f"Total prompt tokens: {total_prompt_tokens_used}")
print(f"Average prompt tokens: {avg_prompt_tokens:.2f} tokens/sample")
print(f"Total completion tokens: {total_completion_tokens_used}")
print(f"Average completion tokens: {avg_completion_tokens:.2f} tokens/sample")
print(f"Total tokens: {total_tokens_used}")
print(f"Average tokens: {avg_tokens:.2f} tokens/sample")
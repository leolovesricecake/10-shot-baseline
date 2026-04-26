import os
from pathlib import Path
# os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 控制用哪张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

from typing import List, Tuple
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 模型（仅用于 GPU 推理阶段）
tokenizer = AutoTokenizer.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', device_map="auto", torch_dtype=torch.bfloat16)

# 数据集（CPU）
import datasets
dataset_name = "/mnt/huawei/ymb/datasets/datasets--mteb--sts12-sts/snapshots/fbe5b9f1f68eac555b70732c0f9a2aab8de5e1cd"
try:
    imdb = datasets.load_from_disk(dataset_name)
except:
    imdb = datasets.load_dataset(dataset_name)
sst5 = imdb

x_train = []
y_train = []
raw_train_dataset = sst5["train"]
for i in range(int(1 * len(raw_train_dataset))):
    x_train.append("text1: " + raw_train_dataset[i]["sentence1"] + " " + " text2: " + raw_train_dataset[i]["sentence2"])
    y_train.append(raw_train_dataset[i]["score"])

raw_test_dataset = sst5["test"]
x_test = []
y_test = []
for i in range(int(0.25 * len(raw_test_dataset))):
    x_test.append("text1: " + raw_test_dataset[i]["sentence1"] + " " + " text2: " + raw_test_dataset[i]["sentence2"])
    y_test.append(raw_test_dataset[i]["score"])

y_train_name = y_train
y_test_name = y_test

prompt = "You are asked to predict the semantic textual similarity of every input text pairs. Your response only contains a single numerical value with the range from 0 to 5. A larger number indicates a higher degree of similarity."
label = "Your response only contains a single numerical value with the range from 0 to 5."


# CPU 阶段：训练集向量化
sentence_model = SentenceTransformer('/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf')

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 8) -> np.ndarray:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CPU: encode train", unit="batch"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch)
        embeddings.extend(emb)
    return np.array(embeddings)

embeddings = encode_texts(sentence_model, x_train, batch_size=128)


def cluster_texts(embeddings: np.ndarray, texts: List[str], y_train_name: List, num_clusters: int = 10):
    """对文本进行聚类，返回每个聚类的代表性样本"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    representative_examples = []
    for cluster_idx in range(num_clusters):
        cluster_samples_idx = np.where(kmeans.labels_ == cluster_idx)[0]
        center = kmeans.cluster_centers_[cluster_idx]
        distances = np.array([np.linalg.norm(embeddings[i] - center) for i in cluster_samples_idx])
        closest_idx = cluster_samples_idx[np.argmin(distances)]
        representative_examples.append((texts[closest_idx], y_train_name[closest_idx]))
    return representative_examples


def generate_in_context(representative_examples: List[Tuple], num_clusters: int) -> str:
    """为每个查询文本选择多个不同簇的示例（随机顺序）"""
    in_context = "For example,"
    random_numbers = random.sample(range(num_clusters), num_clusters)
    for cluster_idx in random_numbers:
        in_context += f"\n {representative_examples[cluster_idx][0]} label: {representative_examples[cluster_idx][1]}"
    return in_context


# 5) 构造消息（CPU 阶段）
def build_messages_constant_ic(x_test: List[str], in_context: str, prompt: str, label: str) -> List[list]:
    """在 CPU 上构建所有消息（所有测试样本共享同一个 in_context）"""
    messages = []
    for text in x_test:
        combined_prompt = f"{prompt} {label} {in_context} {text}"
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ]
        messages.append(message)
    return messages


# 6) CPU 先完成：把所有消息先分词为 CPU 张量（不把 GPU 生成夹杂在循环中）
def tokenize_messages_cpu(messages: List[list]):
    """在 CPU 上计算所有消息的长度，用于后续分桶"""
    tokenizer.pad_token = tokenizer.eos_token
    input_lengths = []
    for message in tqdm(messages, desc="CPU: tokenize(len only)", unit="msg"):
        inputs = tokenizer.apply_chat_template(
            message,
            return_tensors='pt',
            add_generation_prompt=True,
            return_dict=True,
        )
        input_lengths.append(int(inputs['attention_mask'].sum(dim=1).item()))
    return messages, input_lengths


# 7) GPU 阶段：仅做批量生成
def generate_batched(
    messages: List[list],
    input_lengths,
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> List[str]:
    """在 GPU 上批量生成，按长度排序以减少 padding 浪费"""
    outputs_by_idx = {}
    total = len(messages)
    import time
    input_tokens_arr = [0] * total
    output_tokens_arr = [0] * total
    runtime_arr = [0.0] * total

    # 按长度排序以减少 padding 浪费
    sorted_indices = sorted(range(total), key=lambda i: input_lengths[i])
    
    for start in tqdm(range(0, total, batch_size), desc="GPU: generate", unit="batch"):
        end = min(start + batch_size, total)
        idx_batch = sorted_indices[start:end]
        messages_batch = [messages[i] for i in idx_batch]
        
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            return_tensors='pt',
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            enable_thinking=False,
        )
        
        # 更快的拷贝：pin_memory + non_blocking
        input_ids = inputs['input_ids'].pin_memory().to(model.device, non_blocking=True)
        attention_mask = inputs['attention_mask'].pin_memory().to(model.device, non_blocking=True)
        
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # 以批内 attention_mask 计算真实输入长度，并按原索引写回
        batch_lens = attention_mask.sum(dim=1).tolist()
        for j, i in enumerate(idx_batch):
            inp_len = int(batch_lens[j])
            seq_len = int(out[j].shape[-1])
            input_tokens_arr[i] = inp_len
            output_tokens_arr[i] = max(0, seq_len - inp_len)
            decoded = tokenizer.decode(out[j][inp_len:], skip_special_tokens=True)
            outputs_by_idx[i] = decoded.strip()
        t1 = time.time()
        dt = t1 - t0
        for ii in idx_batch:
            runtime_arr[ii] = dt
        
        del input_ids, attention_mask, out
        torch.cuda.empty_cache()
    
    gens_order = [outputs_by_idx[i] for i in range(total)]
    return gens_order, input_tokens_arr, output_tokens_arr, runtime_arr


import numpy as np
from scipy.stats import pearsonr, spearmanr

def mean_squared_error(y_true, y_pred):
    """计算均方误差（MSE）"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """计算均方根误差（RMSE）"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def pearson_correlation_coefficient(x, y):
    """计算皮尔逊相关系数"""
    return pearsonr(x, y)[0]

def spearman_rank_correlation_coefficient(x, y):
    """计算斯皮尔曼秩相关系数"""
    return spearmanr(x, y)[0]

def metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'MSE: {mse}')

    # 计算RMSE
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'RMSE: {rmse}')

    # 计算Pearson相关系数
    pearson_coeff = pearson_correlation_coefficient(y_true, y_pred)
    print(f'Pearson Correlation Coefficient: {pearson_coeff}')

    # 计算Spearman秩相关系数
    spearman_coeff = spearman_rank_correlation_coefficient(y_true, y_pred)
    print(f'Spearman Rank Correlation Coefficient: {spearman_coeff}')

    return mse, rmse, pearson_coeff, spearman_coeff

import re

def extract_and_convert(s):
    matches = re.findall(r'[-+]?\d*\.?\d+', s)
    for match in matches:
        try:
            value = float(match)
        except ValueError:
            continue
        value = max(0.0, min(5.0, value))
        return value
    return 2.5

if __name__ == "__main__":
    for num_clusters in [10]:
        representative_examples = cluster_texts(
            embeddings,
            x_train,
            y_train_name,
            num_clusters=num_clusters,
        )

        # CPU：为该顺序构造一次 in_context（与原脚本一致，整批共享）
        in_context = generate_in_context(representative_examples, num_clusters)
        
        # CPU：构造消息并分词
        messages = build_messages_constant_ic(x_test, in_context, prompt, label)
        messages, input_lengths = tokenize_messages_cpu(messages)
        
        # GPU：批量生成
        gens, input_tokens, output_tokens, runtimes = generate_batched(messages, input_lengths, batch_size=128, max_new_tokens=128)
        
        # 解析预测结果
        score = [extract_and_convert(g) for g in gens]
        
        # 计算评估指标
        mse, rmse, pearson_coeff, spearman_coeff = metric(y_test, score)

        df = pd.DataFrame({
            'Text': x_test,
            'truth': y_test_name,
            'pre': score,
            'mse': [mse] * len(score),
            'rmse': [rmse] * len(score),
            'pearson_coeff': [pearson_coeff] * len(score),
            'spearman_coeff': [spearman_coeff] * len(score),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': [a + b for a, b in zip(input_tokens, output_tokens)],
        'runtime': runtimes
        })
        
        file_name = f'sts16_cluster_kate_{num_clusters}_data_related.csv'
        df.to_csv(RESULT_DIR / file_name, index=False)
    total_in = sum(input_tokens)
    total_out = sum(output_tokens)
    total_tok = total_in + total_out
    total_rt = sum(runtimes)
    n = len(input_tokens)
    with open(RESULT_DIR / file_name, "a") as f:
        f.write("\nmetric,value\n")
        f.write(f"total_input_tokens,{total_in}\n")
        f.write(f"total_output_tokens,{total_out}\n")
        f.write(f"total_tokens,{total_tok}\n")
        f.write(f"total_runtime_sec,{total_rt}\n")
        f.write(f"avg_input_tokens,{total_in / max(n, 1)}\n")
        f.write(f"avg_output_tokens,{total_out / max(n, 1)}\n")
        f.write(f"avg_total_tokens,{total_tok / max(n, 1)}\n")
        f.write(f"avg_runtime_sec,{total_rt / max(n, 1)}\n")
        print(f"Results saved in {file_name}\n")

import os
from pathlib import Path
# os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# # 控制用哪张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', device_map="auto", torch_dtype=torch.bfloat16)


import random
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans



# 加载 Sentence-BERT 模型
sentence_model = SentenceTransformer('/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf')

# 加载 SST-5 数据集

import datasets
dataset_name="/mnt/huawei/ymb/datasets/datasets--mteb--sts12-sts/snapshots/fbe5b9f1f68eac555b70732c0f9a2aab8de5e1cd"
try:
    imdb =  datasets.load_from_disk(dataset_name)
except:
    imdb =  datasets.load_dataset(dataset_name)
#imdb = datasets.load_from_disk("/home/data_cloud/wwq/huggingface/data/sst5")
sst5=imdb 

import csv
x_train = []
y_train = []
raw_train_dataset=sst5["train"]
# 读取CSV文件
print(raw_train_dataset)
for i in range(int(1*len(raw_train_dataset))):
    #print(i)
    x_train.append("text1: "+raw_train_dataset[i]["sentence1"]+" "+" text2: "+raw_train_dataset[i]["sentence2"])
    y_train.append(raw_train_dataset[i]["score"])
print(y_train)
raw_test_dataset = sst5["test"]
raw_test_dataset[0]
x_test=[]
y_test=[]
# for i in range(400):
# for i in range(len(raw_test_dataset_sst5)):
for i in range(int(0.25*len(raw_test_dataset))):
    #print(i)
    x_test.append("text1: "+raw_test_dataset[i]["sentence1"]+" "+" text2: "+raw_test_dataset[i]["sentence2"])
    y_test.append(raw_test_dataset[i]["score"])
# 标签对应的名称

y_test_name = y_test
y_train_name=y_train
# 文本编码
def encode_texts(model, texts, batch_size=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", unit="batch"):
        batch = texts[i:i + batch_size]
        embeddings_batch = model.encode(batch)
        embeddings.extend(embeddings_batch)
    return np.array(embeddings)

embeddings = encode_texts(sentence_model, x_train, batch_size=128)

# 构建倒排索引，以减少BM25计算的开销。以我的理解，这里应该是一开始对x_train统一处理计算一次就可以了？
from collections import defaultdict
import math

# 构建倒排索引
def build_inverted_index(texts):
    inverted_index = defaultdict(list)
    for doc_id, text in enumerate(texts):
        tokens = set(text.split())  # 使用 set 去重
        for token in tokens:
            inverted_index[token].append(doc_id)
    return inverted_index

# 计算 IDF（逆文档频率）
def compute_idf(inverted_index, num_docs):
    idf = {}
    for token, doc_ids in inverted_index.items():
        idf[token] = math.log((num_docs + 1) / (len(doc_ids) + 1)) + 1  # 防止除零
    return idf


# 计算BM25得分
def compute_bm25(query, inverted_index, idf, texts, k=1.5, b=0.75):
    doc_lengths = [len(doc.split()) for doc in texts]
    avg_doc_len = sum(doc_lengths) / len(texts)

    scores = []
    for doc_id, doc in enumerate(texts):
        score = 0
        tokens = set(query.split())
        for token in tokens:
            if token in inverted_index:
                f = doc.split().count(token)  # 词频
                idf_val = idf.get(token, 0)
                doc_len = doc_lengths[doc_id]
                score += idf_val * f * (k + 1) / (f + k * (1 - b + b * (doc_len / avg_doc_len)))
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores]


# 根据BM25检索算法找最相关的示例句子
# 根据BM25检索最相关的文档
def select_relevant_examples(query, top_k, inverted_index=None, idf=None, texts=None):
    relevant_docs = compute_bm25(query, inverted_index, idf, texts)
    top_relevant_docs = relevant_docs[:top_k]  # 获取得分最高的top_k个文档
    return top_relevant_docs



# 生成 in_context 格式的上下文（CPU 阶段：批量生成所有 in-context）

def generate_in_context_for_test(x_test, x_train, y_train_name, top_k, inverted_index=None, idf=None):
    all_in_contexts = []
    for text in tqdm(x_test, desc="CPU: BM25 & in-context", unit="text"):
        # 用select_relevant_examples函数为该text挑选top_k个相关性最强的示例
        relevant_idxs = select_relevant_examples(text, top_k, inverted_index, idf, x_train)
        
        # 生成in-context字符串
        in_context = "For example,"
        for idx in relevant_idxs:
            in_context += f"\n {x_train[idx]} label: {y_train_name[idx]}"
        
        all_in_contexts.append(in_context)
    return all_in_contexts


# 5) 构造消息（CPU 阶段）
def build_messages(x_test, all_in_contexts, prompt, label):
    """在 CPU 上构建所有消息"""
    messages = []
    for text, in_context in zip(x_test, all_in_contexts):
        # 保持原有的 prompt 格式
        combined_prompt = f"{prompt} {label} {in_context} \n {text}"
        # 创建消息格式（与原代码保持一致）
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ]
        messages.append(message)
    return messages


# 6) CPU 先完成：把所有消息先分词为 CPU 张量（不把 GPU 生成夹杂在循环中）
def tokenize_messages_cpu(messages):
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
    messages,
    input_lengths,
    batch_size=128,
    max_new_tokens=128,
):
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
from scipy import spatial
import heapq


def get_topk_smallest_indices(lst, k):
    # 使用enumerate获取元素及其索引，然后使用heapq.nsmallest获取最小的k个元素及其索引
    return [index for (index, value) in heapq.nsmallest(k, lst, key=lambda x: x[1])]


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
# 分类执行并计算准确率（使用批量处理加速）
def classify_texts(x_test, y_test, prompt, label, top_k, inverted_index=None, idf=None, batch_size=128):
    # CPU 阶段：生成所有 in-context 示例
    all_in_contexts = generate_in_context_for_test(
        x_test, x_train, y_train_name, top_k=top_k, 
        inverted_index=inverted_index, idf=idf
    )
    
    # CPU 阶段：构建所有消息
    all_messages = build_messages(x_test, all_in_contexts, prompt, label)
    
    # CPU 阶段：计算所有消息的长度
    messages, input_lengths = tokenize_messages_cpu(all_messages)
    
    # GPU 阶段：批量生成
    gens, input_tokens, output_tokens, runtimes = generate_batched(messages, input_lengths, batch_size=batch_size, max_new_tokens=128)
    
    # 解析预测结果
    score = [extract_and_convert(g) for g in gens]
    
    # 计算评估指标
    mse, rmse, pearson_coeff, spearman_coeff = metric(y_test, score)
    return mse, rmse, pearson_coeff, spearman_coeff, score, input_tokens, output_tokens, runtimes


# 构建倒排索引和计算IDF
inverted_index = build_inverted_index(x_train)
idf = compute_idf(inverted_index, len(x_train))

# 执行分类任务
prompt = "You are asked to predict the semantic textual similarity of every input text pairs. Your response only contains a single numerical value with the range from 0 to 5. A larger number indicates a higher degree of similarity."
label = "Your response only contains a single numerical value with the range from 0 to 5."
import pandas as pd
for w in [10]:
    mse, rmse, pearson_coeff, spearman_coeff, score, input_tokens, output_tokens, runtimes = classify_texts(
        x_test, y_test, prompt, label, top_k=w, 
        inverted_index=inverted_index, idf=idf, batch_size=128
    )
    
    # 保存结果
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
    
    file_name = f'sts16_BM25_{w}_data_related.csv'
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
    print(f"Results for {w} clusters saved in {file_name}\n")
    print("=============================\n")

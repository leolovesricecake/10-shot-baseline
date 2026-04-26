import os
from pathlib import Path
# os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
import re
import datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Models
tokenizer = AutoTokenizer.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('/mnt/huawei/ymb/model/glm-4-9b/model', device_map="auto")

# Load Data
dataset_name = "/mnt/huawei/ymb/datasets/datasets--mteb--sts12-sts/snapshots/fbe5b9f1f68eac555b70732c0f9a2aab8de5e1cd"
try:
    imdb = datasets.load_from_disk(dataset_name)
except:
    imdb = datasets.load_dataset(dataset_name)
sst5 = imdb

x_train = []
y_train = []
raw_train_dataset = sst5["train"]
# Use all training data
for i in range(len(raw_train_dataset)):
    x_train.append("text1: " + raw_train_dataset[i]["sentence1"] + " " + " text2: " + raw_train_dataset[i]["sentence2"])
    y_train.append(raw_train_dataset[i]["score"])

raw_test_dataset = sst5["test"]
x_test = []
y_test = []
# Use 25% of test data as in reference
for i in range(int(0.25 * len(raw_test_dataset))):
    x_test.append("text1: " + raw_test_dataset[i]["sentence1"] + " " + " text2: " + raw_test_dataset[i]["sentence2"])
    y_test.append(raw_test_dataset[i]["score"])

y_test_name = y_test
y_train_name = y_train

def find_most_similar(x_train: List[str], x_test: List[str], model_name: str = '/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf', top_k: int = 5, batch_size: int = 32):
    model_st = SentenceTransformer(model_name)
    # train
    train_embs = []
    for i in tqdm(range(0, len(x_train), batch_size), desc="CPU: encode x_train", unit="batch"):
        batch = x_train[i:i + batch_size]
        embs = model_st.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        train_embs.extend(embs)
    train_embs = np.array(train_embs)
    # test
    test_embs = []
    for i in tqdm(range(0, len(x_test), batch_size), desc="CPU: encode x_test", unit="batch"):
        batch = x_test[i:i + batch_size]
        embs = model_st.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        test_embs.extend(embs)
    test_embs = np.array(test_embs)
    # cosine sim
    cos_sim = cosine_similarity(test_embs, train_embs)
    most_similar = []
    for i in tqdm(range(len(cos_sim)), desc="CPU: topK", unit="text"):
        sim_scores = cos_sim[i]
        top_k_indices = np.argsort(sim_scores)[-top_k:][::-1]
        most_similar.append(top_k_indices)
    return most_similar

# Prompt Definitions
prompt0 = "You are asked to predict the semantic textual similarity of every input text pairs. Your response only contains a single numerical value with the range from 0 to 5. A larger number indicates a higher degree of similarity."
label_desc = "Your response only contains a single numerical value with the range from 0 to 5."

def build_messages_per_sample(x_test: List[str], in_contexts: List[str]) -> List[list]:
    messages: List[list] = []
    for text, in_context in zip(x_test, in_contexts):
        composed = prompt0 + " " + label_desc + " " + in_context + " " + "The text you need to predict the label is:" + text
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": composed},
        ]
        messages.append(message)
    return messages

def tokenize_messages_cpu(messages: List[list]):
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

def generate_batched(
    messages: List[list],
    input_lengths,
    batch_size: int = 16,
    max_new_tokens: int = 128
) -> Tuple[List[str], List[int], List[int], List[float]]:
    outputs_by_idx = {}
    total = len(messages)
    import torch
    import time
    input_tokens_arr = [0] * total
    output_tokens_arr = [0] * total
    runtime_arr = [0.0] * total

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
        )
        input_ids = inputs['input_ids'].pin_memory().to(model.device, non_blocking=True)
        attention_mask = inputs['attention_mask'].pin_memory().to(model.device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        batch_lens = attention_mask.sum(dim=1).tolist()
        for j, i in enumerate(idx_batch):
            inp_len = int(batch_lens[j])
            gen_ids = out[j][inp_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs_by_idx[i] = decoded
            input_tokens_arr[i] = inp_len
            output_tokens_arr[i] = int(gen_ids.shape[0])
            runtime_arr[i] = elapsed / max(len(idx_batch), 1)
        del input_ids, attention_mask, out
        torch.cuda.empty_cache()
    return (
        [outputs_by_idx[i] for i in range(total)],
        input_tokens_arr,
        output_tokens_arr,
        runtime_arr,
    )

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

def metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pearson_coeff = pearsonr(y_true, y_pred)[0]
    spearman_coeff = spearmanr(y_true, y_pred)[0]
    return mse, rmse, pearson_coeff, spearman_coeff

if __name__ == "__main__":
    # Loop for different k values if needed, but here we stick to one configuration or loop as in reference
    # The reference looped over clusters, here we loop over k=5, 10 like in sst5_bm25
    
    for k in [10]:
        print(f"Processing k={k}")
        most_similar = find_most_similar(x_train, x_test, top_k=k)
        in_contexts_per_sample: List[str] = []
        for i, _ in enumerate(x_test):
            ii = most_similar[i][:k][::-1] # Reverse to have most similar closest to input? Or just order.
            in_context = "For example,"
            for idx in ii:
                in_context += "\n" + x_train[idx] + " label: " + str(y_train[idx])
            in_contexts_per_sample.append(in_context)

        messages = build_messages_per_sample(x_test, in_contexts_per_sample)
        messages, input_lengths = tokenize_messages_cpu(messages)
        gens, input_tokens, output_tokens, runtimes = generate_batched(messages, input_lengths, batch_size=128, max_new_tokens=128)
        
        y_pre_score = [extract_and_convert(g) for g in gens]
        
        mse, rmse, pearson_coeff, spearman_coeff = metric(y_test, y_pre_score)
        print(f"k={k}: MSE={mse}, RMSE={rmse}, Pearson={pearson_coeff}, Spearman={spearman_coeff}")

        df = pd.DataFrame({
            'Text': x_test,
            'truth': y_test,
            'pre': y_pre_score,
            'mse': [mse] * len(y_pre_score),
            'rmse': [rmse] * len(y_pre_score),
            'pearson_coeff': [pearson_coeff] * len(y_pre_score),
            'spearman_coeff': [spearman_coeff] * len(y_pre_score),
            'in_context': in_contexts_per_sample,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': [a + b for a, b in zip(input_tokens, output_tokens)],
        'runtime': runtimes
        })

        file_name = f'sts16_iccl_batched_{k}_data_related.csv'
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

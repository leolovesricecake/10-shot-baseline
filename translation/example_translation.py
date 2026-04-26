import os
os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 控制用哪张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method1



from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat-hf')
model = AutoModelForCausalLM.from_pretrained('THUDM/glm-4-9b-chat-hf', device_map="auto")


def llm_output(prompt,label,in_context,text):
    prompt=prompt+" "+label+" "+in_context+" "+"English: "+text
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # message = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant."
    #     },
    #     {
    #         "role": "user",
    #         "content": "How many legs does a cat have?"
    #     }
    # ]

    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors='pt',
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    input_len = inputs['input_ids'].shape[1]
    generate_kwargs = {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "max_new_tokens": 128,
        "do_sample": False,
    }
    out = model.generate(**generate_kwargs)
    ans=tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return ans


from datasets import load_dataset

import numpy as np 
from scipy import stats
#SetFit/sst5
from datasets import load_dataset
import torch
import time
print("1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
from datasets import load_from_disk, load_dataset
dataset_name="/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh"


dataset = load_dataset('parquet', data_files={
'train': r'/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/train.parquet',
'test': r'/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/validation.parquet',
'validation': r'/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/validation.parquet'
})
sst5=dataset
# 准备训练和测试数据



sst5=dataset

sst5=dataset

raw_train_dataset_sst5 = sst5['train']
raw_train_dataset_sst5[0]
x_train=[]
y_train=[]
for i in range(3000):
#for i in range(len(raw_train_dataset_sst5)):
    #print(i)
    x_train.append(raw_train_dataset_sst5[i]['translation']['en'])
    y_train.append(raw_train_dataset_sst5[i]['translation']['zh'])
    #y_train.append(raw_train_dataset_sst5[i]["label"])
raw_test_dataset_sst5 = sst5['validation']
raw_test_dataset_sst5[0]
x_test=[]
y_test=[]
# for i in range(10):
for i in range(int(0.2*(len(raw_test_dataset_sst5)))):
    #print(i)
    x_test.append(raw_test_dataset_sst5[i]['translation']['en'])
    y_test.append(raw_test_dataset_sst5[i]['translation']['zh'])
    
import os
os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 控制用哪张卡

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from tqdm import tqdm

def find_most_similar(x_train, x_test, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=10, batch_size=4, use_gpu=True):
    # 使用 GPU 或 CPU
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # 初始化 SentenceTransformer 模型，启用 fp16 精度
    model = SentenceTransformer(model_name, device=device)
    
    # 对 x_train 进行向量化，分批处理，并添加进度条
    embeddings_train = []
    for i in tqdm(range(0, len(x_train), batch_size), desc="Encoding x_train", unit="batch"):
        batch = x_train[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False, device=device)
        embeddings_train.append(batch_embeddings)
    embeddings_train = torch.cat(embeddings_train, dim=0)

    # 对 x_test 进行向量化，分批处理，并添加进度条
    embeddings_test = []
    for i in tqdm(range(0, len(x_test), batch_size), desc="Encoding x_test", unit="batch"):
        batch = x_test[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False, device=device)
        embeddings_test.append(batch_embeddings)
    embeddings_test = torch.cat(embeddings_test, dim=0)
    
    # 计算余弦相似度，并添加进度条
    cos_sim = cosine_similarity(embeddings_test.cpu(), embeddings_train.cpu())  # 转换为CPU以防止CUDA内存问题
    
    # 为每个 x_test 找到最相似的 top_k 个 x_train 文本
    most_similar = []
    for i in tqdm(range(len(cos_sim)), desc="Finding Most Similar", unit="text"):
        sim_scores = cos_sim[i]
        top_k_indices = np.argsort(sim_scores)[-top_k:][::-1]  # 获取余弦相似度最高的 top_k 个索引
        most_similar.append(top_k_indices)
    
    return most_similar, cos_sim


# 调用函数
most_similar, cos_sim = find_most_similar(x_train, x_test, use_gpu=True)


label_name=["very negative","negative","neutral","positive","very positive"]
y_train_name=y_train
y_test_name=y_test


in_contexts_all=[]
for i, test_text in enumerate(x_test):
    print(f"Test text: {test_text}")
    print("Most similar texts from x_train:")
    in_contexts_i=[]
    for idx in most_similar[i]:
        # "document: "+x_train[j]+" "+"summary: "+y_train_name[j]
        in_contexts_ii="English: "+x_train[idx]+" "+"Chinese: "+y_train_name[idx]
        in_contexts_i.append(in_contexts_ii)
        print(f"  {x_train[idx]} (Similarity: {cos_sim[i][idx]:.4f})")
    in_contexts_all.append(in_contexts_i)
    print()


prompt = "Translate the following text from English to Chinese."
label=" "

import math
from collections import Counter

def char_tokenize(s: str):
    """
    按字符切分中文文本，去掉空白。
    """
    s = "".join(s.split())
    return list(s)

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def clipped_count(candidate_ngrams, reference_ngrams):
    cand_counts = Counter(candidate_ngrams)
    ref_counts = Counter(reference_ngrams)
    return sum(min(cand_counts[ng], ref_counts.get(ng, 0)) for ng in cand_counts)

def bleu_our(candidate: str, reference: str, max_n=4, smooth=True):
    cand_tokens = char_tokenize(candidate)
    ref_tokens  = char_tokenize(reference)

    c, r = len(cand_tokens), len(ref_tokens)
    if c == 0:
        return 0.0

    # Brevity Penalty
    bp = math.exp(min(0.0, 1.0 - r / c))

    # n-gram precisions
    weights = [1.0/max_n] * max_n
    log_p_sum = 0.0
    for n in range(1, max_n+1):
        cand_ng = ngrams(cand_tokens, n)
        ref_ng  = ngrams(ref_tokens, n)
        match = clipped_count(cand_ng, ref_ng)
        total = max(len(cand_ng), 1)
        p_n = match / total

        # 平滑：如果精确率为 0，则替换为一个极小值
        if p_n == 0 and smooth:
            p_n = 1e-9
        log_p_sum += weights[n-1] * math.log(p_n)

    bleu_score = bp * math.exp(log_p_sum)
    return bleu_score



import pandas as pd
from sklearn.metrics import accuracy_score
import random
for j in range(3,6,1):
    outs=[]
    for i in tqdm(range(len(x_test))):
        ii=in_contexts_all[i][:j]
        # print(ii)
        in_context="For example,"
        for w in range(len(ii)):
            in_context=in_context+"\n"+ii[w]
        out=llm_output(prompt,label,in_context,x_test[i])
        outs.append(out)
        
    y_pre=[]
    y_pre_text=[]
    for i in range(len(outs)):
        bleu = bleu_our(outs[i],y_test[i])
        y_pre.append(bleu)

        y_pre_text.append(outs[i])
    accuracy = sum(y_pre) / len(y_pre)

    accs=[accuracy]*len(y_pre)
    df = pd.DataFrame({
    'Text': x_test,
    'truth': y_test_name,
    'pre': y_pre_text,
    'acc':accs
    })
    # accuracy = accuracy_score(y_pre, y_test)
    # accs=[accuracy]*len(y_pre)

    # 写入 CSV 文件
    file_name = f'wmt19_en-zh_kate_{j}_data_related.csv'
    df.to_csv(file_name, index=False)
    
    
    
for j in range(10,11,1):
    outs=[]
    for i in tqdm(range(len(x_test))):
        ii=in_contexts_all[i][:j]
        # print(ii)
        in_context="For example,"
        for w in range(len(ii)):
            in_context=in_context+"\n"+ii[w]
        out=llm_output(prompt,label,in_context,x_test[i])
        outs.append(out)
        
    y_pre=[]
    y_pre_text=[]
    for i in range(len(outs)):
        bleu = bleu_our(outs[i],y_test[i])
        y_pre.append(bleu)

        y_pre_text.append(outs[i])
    accuracy = sum(y_pre) / len(y_pre)

    accs=[accuracy]*len(y_pre)
    df = pd.DataFrame({
    'Text': x_test,
    'truth': y_test_name,
    'pre': y_pre_text,
    'acc':accs
    })
    # accuracy = accuracy_score(y_pre, y_test)
    # accs=[accuracy]*len(y_pre)

    # 写入 CSV 文件
    file_name = f'wmt19_en-zh_kate_{j}_data_related.csv'
    df.to_csv(file_name, index=False)
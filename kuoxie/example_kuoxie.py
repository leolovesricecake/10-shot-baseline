import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# --- 2. 数据预处理 ---

# 定义超参数

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义超参数
MAX_SEQ_LENGTH = 512  # BERT处理的最大序列长度 (Text + Context)
BATCH_SIZE = 16       # BERT模型较大，可能需要减小Batch Size
EPOCHS = 4            # 对于微调任务，通常不需要很多Epoch
LEARNING_RATE_BERT = 2e-5      # 微调BERT的学习率
LEARNING_RATE_CLASSIFIER = 1e-3 # 分类层的学习率
BERT_MODEL_NAME = '/home/wangwenqiang/huggingface/model/bert-base-uncased' # 使用基础的、不区分大小写的BERT模型

LLM = 'glm'
Model_PATH_FRONT = "/mnt/huawei/wwq/project/ICL-in-cotext-order/kuoxie/giga_test/train_4/model_glm/"
all_filenames = [
    f"./split_output_files_{LLM}/part_1.csv",
    f"./split_output_files_{LLM}/part_2.csv",
    f"./split_output_files_{LLM}/part_3.csv",
    f"./split_output_files_{LLM}/part_4.csv",
    f"./split_output_files_{LLM}/part_5.csv",
    f"./split_output_files_{LLM}/part_6.csv",
    f"./split_output_files_{LLM}/part_7.csv",
    f"./split_output_files_{LLM}/part_8.csv",
    f"./split_output_files_{LLM}/part_9.csv",
    f"./split_output_files_{LLM}/part_10.csv",
    f"./split_output_files_{LLM}/part_11.csv",
    f"./split_output_files_{LLM}/part_12.csv",
    f"./split_output_files_{LLM}/part_13.csv",
    f"./split_output_files_{LLM}/part_14.csv",
    f"./split_output_files_{LLM}/part_15.csv",
    f"./split_output_files_{LLM}/part_16.csv",
    f"./split_output_files_{LLM}/part_17.csv",
    f"./split_output_files_{LLM}/part_18.csv",
    f"./split_output_files_{LLM}/part_19.csv",
    f"./split_output_files_{LLM}/part_20.csv",
    f"./split_output_files_{LLM}/part_21.csv",
    f"./split_output_files_{LLM}/part_22.csv",
    f"./split_output_files_{LLM}/part_23.csv",
    f"./split_output_files_{LLM}/part_24.csv"
]
# --- 2. 使用BERT分词器进行数据预处理 ---
print(f"正在加载BERT分词器: {BERT_MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


class BertDualInputDataset(Dataset):
    def __init__(self, texts, contexts, labels, tokenizer, max_len):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        context = str(self.contexts[idx])
        label = self.labels[idx]

        # BERT的标准做法是将两个句子用[SEP]隔开
        # 格式: [CLS] text [SEP] context [SEP]
        encoding = self.tokenizer.encode_plus(
            text,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class DeepBertClassifier(nn.Module):
    def __init__(self, bert_model_name,n_layers, hidden_dim=512):
        """
        初始化一个包含六个 nn.Linear 层的深度分类器。
        结构: [输入层 -> 4x 隐藏层 -> 输出层]
        """
        super(DeepBertClassifier, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size # 通常是 768

        # --- 构建六层线性网络 ---
        layers = []
        
        # 第 1 层 (输入层)
        # 将BERT的输出从 768 维映射到 hidden_dim 维
        layers.append(nn.Linear(bert_output_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))

        # 第 2, 3, 4, 5 层 (四个隐藏层)
        # 每个隐藏层都是从 hidden_dim 维到 hidden_dim 维的映射
        for _ in range(n_layers): # 循环n次，创建n个隐藏层
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        # 第 6 层 (输出层)
        # 从 hidden_dim 维映射到最终的输出维度 (1, 用于二分类)
        layers.append(nn.Linear(hidden_dim, 1))
        
        # 将所有层打包进 nn.Sequential 容器
        self.classifier = nn.Sequential(*layers)
        
        print(f"深度分类器初始化完成，包含 {n_layers} 个 nn.Linear 层。")

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 通过BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 使用 [CLS] 标记的池化输出作为整个序列的表示
        pooled_output = outputs.pooler_output
        
        # 将BERT的输出送入深度分类头
        logits = self.classifier(pooled_output)
        return logits



def predict_and_rank_by_confidence(texts, contexts, true_labels, filenames, model_to_use, tokenizer, max_len, device):
    if len(texts) != len(true_labels):
        raise ValueError("文本数量必须与标签数量一致！")

    model_to_use.eval()
    results = []

    with torch.no_grad():
        # 批量tokenizer编码
        encoding = tokenizer(
            texts,
            contexts,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        logits = model_to_use(input_ids, attention_mask, token_type_ids)
        # prob_one = torch.sigmoid(logits.squeeze())
        # prob_zero = 1 - prob_one

        for i in range(len(texts)):
            true_label = true_labels[i]
            confidence = logits[i].item()
            results.append({
                "text": texts[i],
                "context": contexts[i],
                "true_label": true_label,
                # "predicted_prob_one": prob_one[i].item(),
                "confidence_in_true_label": confidence,
                "filename": filenames[i]
            })

    return sorted(results, key=lambda x: x['confidence_in_true_label'], reverse=True) # 按置信度降序排序 rouge 1 期望越高越好




# 将所有CSV文件读入pandas DataFrame列表
try:
    all_dataframes = [pd.read_csv(f) for f in all_filenames]
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()


# 核心前提假设：所有文件的行数必须相同
num_rows = len(all_dataframes[0])
if not all(len(df) == num_rows for df in all_dataframes):
    raise ValueError("错误：所有输入文件的行数必须完全相同！")

print(f"所有文件均包含 {num_rows} 行数据，开始处理...")

for layers in range(6):
    for epoch in range(10):
        MODEL_SAVE_PATH = Model_PATH_FRONT + f"llm_{LLM}_layers_{layers + 1}_epoch_{epoch + 1}.pth"
        new_model = DeepBertClassifier(bert_model_name=BERT_MODEL_NAME, n_layers = layers + 1).to(device)


        # b. 加载已保存的状态字典
        print(f"从 {MODEL_SAVE_PATH} 加载参数...")
        new_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        # c. 将模型设置为评估模式
        #    这对于关闭 Dropout 和 BatchNorm 等层在推理时的行为至关重要
        new_model.eval()
        # 用于存储每一行最终胜出的文件名
        best_filename_per_row = []
        best_label=[]

        # 外层循环：遍历每一行
        for i in range(num_rows):
            
            # -- 步骤 2a: 为当前行构建一个临时批次 --
            # 这个批次包含来自10个文件的第 i 行数据
            batch_texts = []
            batch_contexts = []
            batch_labels = []
            batch_filenames = []
            
            for de_index,df in enumerate(all_dataframes):
                row = df.iloc[i]
                batch_texts.append(row['Text'])
                batch_contexts.append(row['in_context'])
                batch_labels.append(row['score'])
                batch_filenames.append(all_filenames[de_index])

            # -- 步骤 2b: 调用预测函数 --
            # 注意：函数返回的是一个排序后的列表
            # 列表里的每个元素是一个包含text, context, confidence等信息的字典
            ranked_results = predict_and_rank_by_confidence(
                batch_texts,
                batch_contexts,
                batch_labels,
                batch_filenames,
                new_model,
                tokenizer,
                MAX_SEQ_LENGTH,
                device
            )


            winner_sample = ranked_results[0]
            try:
                winner_filename = winner_sample['filename']
                winner_true_label = winner_sample['true_label']
                best_label.append(winner_true_label)
                best_filename_per_row.append(winner_filename)
            except ValueError:
                print(f"警告：在第 {i} 行处理中未能定位到胜出样本，将跳过此行。")
                best_filename_per_row.append(None) # 或者记录一个错误标记
                
        if best_label:
            average_of_all_labels = sum(best_label) / len(best_label)
        else:
            average_of_all_labels = 0 # 或者设置为 None，取决于你希望如何处理空列表的情况
        # 创建最终的DataFrame
        
        output_df = pd.DataFrame({'best_source_file': best_filename_per_row, "Score": best_label, "acc": average_of_all_labels})

        # 保存到CSV
        output_filename = f'./category_test_{LLM}/{LLM}_emotion_layers_{layers + 1}_epoch{epoch + 1}.csv'
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        output_df.to_csv(output_filename, index=False)

        print(f"结果已成功保存到: {output_filename}")
        

import glob

def rank_csv_by_acc(folder_path: str):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"找到了 {len(csv_files)} 个CSV文件，正在处理...")
    # --- 3. 提取每个文件的ACC值 ---
    file_accuracies = []
    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查 'acc' 列是否存在
            if 'acc' not in df.columns:
                print(f"  -> 警告：文件 '{os.path.basename(file_path)}' 中没有 'acc' 列，已跳过。")
                continue

            # 提取 'acc' 列，并处理空值
            acc_series = df['acc'].dropna()
            
            if acc_series.empty:
                print(f"  -> 警告：文件 '{os.path.basename(file_path)}' 的 'acc' 列为空，已跳过。")
                continue
            file_acc = acc_series.max()

            # 记录文件名和其对应的acc值
            file_accuracies.append({
                'filename': os.path.basename(file_path),
                'acc': file_acc
            })

        except pd.errors.EmptyDataError:
            print(f"  -> 警告：文件 '{os.path.basename(file_path)}' 是空的，已跳过。")
        except Exception as e:
            print(f"  -> 错误：处理文件 '{os.path.basename(file_path)}' 时出错: {e}")

    # --- 4. 按ACC值进行降序排序 ---
    # 使用 lambda 函数作为排序的 key，并设置 reverse=True 实现降序
    sorted_files = sorted(file_accuracies, key=lambda item: item['acc'], reverse=True)
    print("此任务acc为rouge1 平均值 期望越大越好")
    # --- 5. 打印最终排序结果 ---
    print("\n" + "="*40)
    print("      CSV文件按 'acc' 列降序排名结果")
    print("="*40)

    if not sorted_files:
        print("未能从任何文件中成功提取'acc'值。")
    else:
        for item in sorted_files:
            # 格式化输出，acc值保留4位小数
            print(f"文件名: {item['filename']:<30} | ACC: {item['acc']:.4f}")
    
    print("="*40)

target_folder = f"./category_test_{LLM}"
rank_csv_by_acc(target_folder)
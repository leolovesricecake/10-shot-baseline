import os
import csv
os.environ["HF_HOME"] = "/data/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_TOKEN'] = "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"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 控制用哪张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat-hf')
model = AutoModelForCausalLM.from_pretrained('THUDM/glm-4-9b-chat-hf', device_map="auto")

message = [
    {
        "role": "system",
        "content": "Answer the following question."
    },
    {
        "role": "user",
        "content": "How many legs does a cat have?"
    }
]

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
print(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))


def llm_output(prompt,label,in_context,text):
    prompt=prompt+" "+label+" "+in_context+" "+text
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
dataset_name="/home/wangwenqiang/huggingface/data/emotion"
try:
    imdb =  datasets.load_from_disk(dataset_name)
except:
    imdb =  datasets.load_dataset(dataset_name)
#imdb = datasets.load_from_disk("/home/data_cloud/wwq/huggingface/data/sst5")
sst5=imdb 


raw_train_dataset_sst5 = sst5["train"]
raw_train_dataset_sst5[0]
x_train=[]
y_train=[]

raw_train_dataset = sst5['train']
raw_test_dataset = sst5['test']
import csv
x_train = []
y_train = []

# 读取CSV文件
for i in range(int(len(raw_train_dataset))):
    #print(i)
    x_train.append(raw_train_dataset[i]["text"])
    y_train.append(raw_train_dataset[i]["label"])

print("x_train:", x_train)
print("y_train:", y_train)
raw_test_dataset_sst5 = sst5["test"]
raw_test_dataset_sst5[0]
x_test=[]
y_test=[]
for i in range(int(0.2*len(raw_test_dataset_sst5))):
# for i in range(len(raw_test_dataset_sst5)):
    #print(i)
    x_test.append(raw_test_dataset_sst5[i]["text"])
    y_test.append(raw_test_dataset_sst5[i]["label"])
    
    
prompt = "Predict the label of the input text, only give me the label is enough, for instance, label = 'Anger', label = 'Fear', label = 'Joy', label = 'Love', label = 'Sadness', label = 'Surprise'"
label="labels are Anger, Fear, Joy, Love, Sadness, and Surprise, not other labels"

in_context="""
"""
text="the rock is destined to be the 21st century's new  conan  and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal ."
from tqdm import tqdm
def out_need(x_test,in_context,label):
    outs=[]
    d2=x_test
    #Anger, Fear, Joy, Love, Sadness, and Surprise
    
    for i in tqdm(range(len(x_test))):
        text=x_test[i]
        out=llm_output(prompt,label,in_context,text)
        outs.append(out)
    return outs


print(in_context)


label_name=["sadness","joy","love","anger","fear","surprise"]
y_test_name=[]
for i in range(len(y_test)):
    id=y_test[i]
    y_test_name.append(label_name[id])
y_train_name=[]
for i in range(len(y_train)):
    id=y_train[i]
    y_train_name.append(label_name[id])
    
import pandas as pd
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.metrics import accuracy_score
import random
for w in range(1,13,3):
    in_context="For example,"
    random_numbers = [random.randint(0, len(x_train)-2) for _ in range(w)]
    for j in random_numbers:
        in_context=in_context+"\n"+"input text: "+x_train[j]+" "+"label: "+y_train_name[j]
    #print(in_context)
    outs=out_need(x_test,in_context,label)
    y_pre=[]
    y_pre_text=[]
    for i in range(len(outs)):
        if "nger" in outs[i]:
            y_pre.append(3)
            y_pre_text.append("anger")
        elif "ear" in outs[i]:
            y_pre.append(4)
            y_pre_text.append("fear")
        
            
        elif "ove" in outs[i]:
            y_pre.append(2)
            y_pre_text.append("love")
            #technology
        elif "adness" in outs[i]:
            y_pre.append(0)
            y_pre_text.append("sadness")
        elif "urprise" in outs[i]:
            y_pre.append(5)
            y_pre_text.append("surprise")
        elif "oy" in outs[i]:
            y_pre.append(1)
            y_pre_text.append("joy")
        else:
            y_pre.append(6)
            y_pre_text.append("other")
    accuracy = accuracy_score(y_pre, y_test)
    accs=[accuracy]*len(y_pre)

    df = pd.DataFrame({
    'Text': x_test,
    'truth': y_test_name,
    'pre': y_pre_text,
    'acc':accs
    })

    # 写入 CSV 文件
    file_name = f'emotion_random_{w}_data_related.csv'
    df.to_csv(file_name, index=False)




for w in range(2,11,3):
    in_context="For example,"
    random_numbers = [random.randint(0, len(x_train)-2) for _ in range(w)]
    for j in random_numbers:
        in_context=in_context+"\n"+"input text: "+x_train[j]+" "+"label: "+y_train_name[j]
    #print(in_context)
    outs=out_need(x_test,in_context,label)
    y_pre=[]
    y_pre_text=[]
    for i in range(len(outs)):
        if "nger" in outs[i]:
            y_pre.append(3)
            y_pre_text.append("anger")
        elif "ear" in outs[i]:
            y_pre.append(4)
            y_pre_text.append("fear")
        
            
        elif "ove" in outs[i]:
            y_pre.append(2)
            y_pre_text.append("love")
            #technology
        elif "adness" in outs[i]:
            y_pre.append(0)
            y_pre_text.append("sadness")
        elif "urprise" in outs[i]:
            y_pre.append(5)
            y_pre_text.append("surprise")
        elif "oy" in outs[i]:
            y_pre.append(1)
            y_pre_text.append("joy")
        else:
            y_pre.append(6)
            y_pre_text.append("other")
    accuracy = accuracy_score(y_pre, y_test)
    accs=[accuracy]*len(y_pre)

    df = pd.DataFrame({
    'Text': x_test,
    'truth': y_test_name,
    'pre': y_pre_text,
    'acc':accs
    })

    # 写入 CSV 文件
    file_name = f'emotion_random_{w}_data_related.csv'
    df.to_csv(file_name, index=False)




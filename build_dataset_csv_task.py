import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
dataset_name="/home/wangwenqiang/huggingface/data/stsb"
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
    x_train.append("text1: "+raw_train_dataset[i]["text1"]+" "+"text2: "+raw_train_dataset[i]["text2"])
    y_train.append(raw_train_dataset[i]["label"])
print(y_train)
raw_test_dataset = sst5["validation"]
raw_test_dataset[0]
x_test=[]
y_test=[]
for i in range(int(0.25*len(raw_test_dataset)),int(1*len(raw_test_dataset))):
    #print(i)
    x_test.append("text1: "+raw_test_dataset[i]["text1"]+" "+"text2: "+raw_test_dataset[i]["text2"])
    y_test.append(raw_test_dataset[i]["label"])

'''
text label
生成2个CSV文件   训练集合 x_train, y_train  stsb_train.csv  测试集合 x_test, y_test  stsb_train.csv

/mnt/huawei/wwq/project/ICL-in-cotext-order/classfication/Emotion

/mnt/huawei/wwq/project/ICL-in-cotext-order/classfication/sst5

gigatiny
/mnt/huawei/wwq/project/ICL-in-cotext-order/kuoxie/giga_test/text_3/sst5_kate_data_related.py

sts12
/mnt/huawei/wwq/project/ICL-in-cotext-order/similarity/sts12/sts12_train_3/sst5_kate_data_related.py


/mnt/huawei/wwq/project/ICL-in-cotext-order/similarity/stsb/stsb_train_4/sst5_kate_data_related.py


gigaword
/mnt/huawei/wwq/project/ICL-in-cotext-order/summary/gigatiny/train_3/sst5_kate_data_related.py


wmt19_Zh-En
/mnt/huawei/wwq/project/ICL-in-cotext-order/translation/wmt19_Zh-En_baseline/glm/sst5_kate_data_related.py


做的好，接下来看到`/mnt/huawei/wwq/project/ICL-in-cotext-order/translation/wmt19_baseline/glm/sst5_kate_data_related.py`。
请根据它实现脚本`/mnt/huawei/wwq/project/10-shot-baseline/translation/wmt19_En-Zh/build_datasets_csv.py`：
1. 读取数据集，按照原逻辑进行训练、测试集划分。
2. 保存到 `/mnt/huawei/wwq/project/10-shot-baseline/datasets/wmt19_En-Zh` 目录下，文件名分别为 `wmt19_En-Zh_train.csv` 和 `wmt19_En-Zh_test.csv`，包含text和label列。
wmt19_En-Zh
/mnt/huawei/wwq/project/ICL-in-cotext-order/translation/wmt19_baseline/glm/sst5_kate_data_related.py
'''
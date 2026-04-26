# 文本任务多基线使用说明

## 1. 统一入口
- 全部基线入口：`python run_baselines.py ...`
- 仅 COT 入口：`python run_cot.py ...`（本质是 `run_baselines.py --baselines cot`）

## 2. 单卡约束
- 现在强制单卡运行。
- 可以用两种方式指定：
  - 方式 A：命令行 `--cuda-device 0`
  - 方式 B：环境变量 `CUDA_VISIBLE_DEVICES=0`
- 若传入多卡（如 `0,1`）会直接报错退出。

## 3. 一键脚本
### 3.1 跑完所有 COT（两模型）
```bash
bash scripts/run_all_cot.sh <datasets_root> <output_root> [cuda_device]
```

### 3.2 跑完全部基线（cot/sc-cot/sv-cot/few-shot-cot/random/fixed）
```bash
bash scripts/run_all_baselines.sh <datasets_root> <output_root> [cuda_device] [retrieval_model_path]
```

### 3.3 全基线冒烟（每个数据集前 N 条）
```bash
bash scripts/run_smoke_all_baselines.sh <datasets_root> <output_root> [max_samples] [cuda_device] [retrieval_model_path]
```

### 3.4 COT 冒烟
```bash
bash scripts/run_cot_smoke.sh <datasets_root> <output_root> [max_samples] [cuda_device]
```

## 4. few-shot 检索与网络告警
- `few-shot-cot` 默认 `--retrieval-backend auto`。
- `auto` 模式现在只在“本地路径存在”时启用语义检索；否则直接回退 BM25，不会再主动走远程下载。
- 建议脚本传本地向量模型路径（脚本默认已给本地路径）：
  - `/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- 如果手动用 Python 跑，也建议显式传：
```bash
python run_baselines.py \
  --baselines few-shot-cot \
  --retrieval-backend auto \
  --retrieval-model-name /mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf \
  --cuda-device 0 \
  ...
```

## 5. 输出结构
- 路径：`<output_root>/<baseline>/<model_alias>/result/`
- 文件：`<dataset>_<baseline>[_k10]_data_related.csv`
- 公共列：`Text, truth, pre, pred_raw, input_tokens, output_tokens, total_tokens, runtime`
- 文件末尾追加 `metric,value` footer（总 token、均值耗时、主指标、策略参数）。

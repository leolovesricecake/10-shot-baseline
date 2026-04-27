# 文本任务多基线使用说明

## 1. 统一入口
- 全部基线入口：`python run_baselines.py ...`
- 仅 COT 入口：`python run_cot.py ...`（等价于 `run_baselines.py --baselines cot`）

## 2. 单卡运行规范
- 只允许单卡运行。
- 可通过两种方式指定 GPU：
  - 命令行：`--cuda-device 0`
  - 环境变量：`CUDA_VISIBLE_DEVICES=0`
- 如果传入多卡（如 `0,1`）或非法值（如 `0 1`、`gpu0`），程序会直接报错退出。

## 3. 一键脚本
说明：统一脚本会逐个执行 Python 任务；单个任务失败不会中断后续任务，最后统一汇总状态。

### 3.1 跑完所有 COT（2 模型）
```bash
bash scripts/run_all_cot.sh <datasets_root> <output_root> [cuda_device]
```

### 3.2 跑完 6 个 baseline（cot/sc-cot/sv-cot/few-shot-cot/random/fixed）
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

## 4. Python 直接运行示例
### 4.1 一条命令跑完所有 COT
```bash
python run_cot.py \
  --datasets-root /mnt/huawei/wwq/project/10-shot-baseline/datasets \
  --output-root /mnt/huawei/wwq/project/10-shot-baseline/results \
  --cuda-device 0 \
  --models glm4_9b llama3_1_8b
```

### 4.2 一条命令跑完所有 baseline
```bash
python run_baselines.py \
  --baselines cot sc-cot sv-cot few-shot-cot random fixed \
  --datasets-root /mnt/huawei/wwq/project/10-shot-baseline/datasets \
  --output-root /mnt/huawei/wwq/project/10-shot-baseline/results \
  --cuda-device 0 \
  --retrieval-backend semantic \
  --models glm4_9b llama3_1_8b
```

## 5. few-shot 检索与 MiniLM 加载说明
- 现在只支持两种后端：`semantic`、`bm25`。
- 默认后端是 `semantic`。
- `semantic` 模式是严格模式：
  - 如果 `--retrieval-model-name` 路径不存在，直接报错退出。
  - 如果模型加载失败，直接报错退出。
- 想强制使用 BM25 时，显式传：`--retrieval-backend bm25`。
- 统一默认检索模型路径：
  - `/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`

## 6. 日志输出（新增）
每次运行会生成独立日志目录：
- `<output_root>/_logs/<run_id>/events.jsonl`
- `<output_root>/_logs/<run_id>/run.log`
- `<output_root>/_logs/<run_id>/run_summary.json`

其中：
- `events.jsonl`：结构化事件（推荐机器排查）。
- `run.log`：文本日志（推荐人工快速查看）。
- `run_summary.json`：本次运行汇总（成功/失败、结果路径、失败原因）。

### 6.1 如何确认 MiniLM 是否成功加载
在 `events.jsonl` 里检索事件 `retriever_init`，关注字段：
- `requested_backend`：请求的检索后端（semantic/bm25）
- `selected_backend`：实际使用后端（semantic 或 bm25）
- `semantic_model_name`：MiniLM 配置路径
- `semantic_model_path_exists`：路径是否存在
- `semantic_load_success`：是否成功加载 MiniLM
- `fallback_reason`：失败原因（通常在 semantic 失败时出现）

判断规则：
- MiniLM 成功：`selected_backend=semantic` 且 `semantic_load_success=true`
- MiniLM 失败：查看 `strategy_prepare_failed` / `dataset_failed` / `run_fatal` 事件

## 7. 结果输出结构
- 目录：`<output_root>/<baseline>/<model_alias>/result/`
- 文件：`<dataset>_<baseline>[_k10]_data_related.csv`
- 公共列：`Text, truth, pre, pred_raw, input_tokens, output_tokens, total_tokens, runtime`
- 文件末尾追加 `metric,value` footer（总 token、平均耗时、主指标、策略参数）

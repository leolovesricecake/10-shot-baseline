## 1. 目录分层
- 全局通用层（仓库根目录）：
  - `baseline_runner.py`：统一调度器（数据读取、批推理、结果落盘、footer 汇总）。
  - `baseline_strategies.py`：六种 baseline 策略实现。
  - `baseline_retrieval.py`：few-shot 检索（语义检索 + BM25 回退）。
  - `baseline_task_base.py`：任务处理器基类、通用解析与指标工具。
  - `baseline_runner_config.py`：模型注册、任务入口路径、公共输出列。
- 任务层（各任务目录）：
  - `classfication/task_entry.py`
  - `similarity/task_entry.py`
  - `translation/task_entry.py`
  - `summary/task_entry.py`
  - `kuoxie/task_entry.py`
- 每个 `task_entry.py` 都提供两个入口对象：
  - `TASK_DATASETS`：数据集注册配置。
  - `TASK_PROCESSOR`：该任务的 prompt/解析/指标/示例格式化逻辑。

## 2. 支持的 baseline
- `cot`：零样本 COT，单次生成。
- `few-shot-cot`：每条样本检索 top-k 示例，默认语义检索，失败回退 BM25。
- `random`：每个数据集随机抽 k 个训练示例，整个测试集共享。
- `fixed`：每个数据集取前 k 个训练示例，整个测试集共享。
- `sc-cot`：零样本 COT 多次采样（默认 5 次）后聚合。
- `sv-cot`：零样本 COT 初答 + 自校验二次回答。

## 3. 输出格式
- 目录：`<output_root>/<baseline>/<model_alias>/result/`
- 文件：`<dataset>_<baseline>[_k{k}]_data_related.csv`
  - `few-shot-cot/random/fixed` 自动带 `_k{k}`。
- 公共列：
  - `Text, truth, pre, pred_raw, input_tokens, output_tokens, total_tokens, runtime`
- 任务指标列：
  - 分类：`acc, accuracy, macro_f1`
  - 回归：`mse, rmse, pearson_coeff, spearman_coeff`
  - 翻译：`acc, bleu, chrf`
  - 摘要/扩写：`acc, rouge1, rouge2, rougeL`
- baseline 过程列（按策略追加）：
  - few-shot/random/fixed：`in_context, context_indices, retrieval_scores`
  - sc-cot：`sc_raw_samples, sc_parsed_samples`
  - sv-cot：`sv_initial_raw, sv_verified_raw`
- CSV 末尾追加 `metric,value` footer（总 token、平均耗时、主指标、策略参数）。

## 4. 一键运行命令
### 4.1 跑完所有 COT 实验（两模型 × 全数据集）
```bash
bash scripts/run_all_cot.sh /mnt/huawei/wwq/project/10-shot-baseline/datasets /mnt/huawei/wwq/project/10-shot-baseline/results
```

### 4.2 跑完所有 baseline（六策略 × 两模型 × 全数据集）
```bash
bash scripts/run_all_baselines.sh /mnt/huawei/wwq/project/10-shot-baseline/datasets /mnt/huawei/wwq/project/10-shot-baseline/results
```

### 4.3 冒烟（每个数据集只跑前 2 条）
```bash
bash scripts/run_smoke_all_baselines.sh /mnt/huawei/wwq/project/10-shot-baseline/datasets /mnt/huawei/wwq/project/10-shot-baseline/results 2
```

## 5. 直接用 Python 命令
### 5.1 只跑 COT
```bash
CUDA_VISIBLE_DEVICES=3 python run_baselines.py \
  --baselines cot \
  --models glm4_9b llama3_1_8b \
  --datasets-root /mnt/huawei/wwq/project/10-shot-baseline/datasets \
  --output-root /mnt/huawei/wwq/project/10-shot-baseline/results \
  --skip-missing-datasets
```

### 5.2 跑全部 baseline
```bash
CUDA_VISIBLE_DEVICES=3 python run_baselines.py \
  --baselines cot sc-cot sv-cot few-shot-cot random fixed \
  --models glm4_9b llama3_1_8b \
  --datasets-root /mnt/huawei/wwq/project/10-shot-baseline/datasets \
  --output-root /mnt/huawei/wwq/project/10-shot-baseline/results \
  --skip-missing-datasets
```

## 6. 常用参数
- `--shot-k 10`：few-shot/random/fixed 的示例数。
- `--sc-num-samples 5 --sc-temperature 0.7 --sc-top-p 0.9`：sc-cot 采样参数。
- `--retrieval-backend auto|semantic|bm25`：few-shot 检索后端。
- `--skip-missing-datasets`：缺失数据集时跳过而不是报错。
- `--overwrite`：覆盖已有结果。
- `--max-samples N`：只跑每个测试集前 N 条。

## 7. 扩展方式
- 新增模型：改 `baseline_runner_config.py` 的 `MODEL_REGISTRY`。
- 新增数据集：改对应任务目录的 `TASK_DATASETS`。
- 新增 baseline：在 `baseline_strategies.py` 新增策略类并注册到 `STRATEGY_REGISTRY`。

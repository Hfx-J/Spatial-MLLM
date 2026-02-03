# Spatial-MLLM 训练脚本使用说明

Stage 1（SFT）假设已经完成。这里是 Stage 2 和 Stage 3 的完整流程。

---

## 目录结构

所有文件放在同一个目录下，与你原有的 `train_sft.py` 并排：

```
project/
├── train_sft.py                  # 你原有的 Stage 1 脚本（会被 import）
├── reward.py                     # 共用模块，stage2 和 stage3 都依赖
├── stage2_cold_start_data.py     # Stage 2a：用 API 生成 CoT 训练数据
├── stage2b_cold_start_train.py   # Stage 2b：用生成的数据微调 200 steps
└── stage3_grpo_train.py          # Stage 3：GRPO RL 训练
```

> **注意**：`stage2b` 和 `stage3` 内部会 `from train_sft import get_model, set_model`，
> 所以你的 Stage 1 脚本文件名必须是 `train_sft.py`。如果不是，改一下对应的 import 语句。

---

## 依赖安装

```bash
pip install openai pillow numpy torch transformers accelerate
```

如果你的环境已经跑过 Stage 1，基本都装齐了，多装一个 `openai` 就行：

```bash
pip install openai
```

---

## Stage 2a：生成 Cold Start 数据

用通义千问 API 对训练集子集生成带推理链的 CoT 数据。

**前提**：设置 API key

```bash
export DASHSCOPE_API_KEY=your_dashscope_api_key
```

**运行**：

```bash
python stage2_cold_start_data.py \
    --spatial_data_path  ./spatial_mllm_120k.json \
    --video_base_dir     ./videos \
    --output_path        ./cold_start_data.json \
    --ns                 5000 \
    --k                  3 \
    --num_frames         16
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--spatial_data_path` | 必填 | Spatial-MLLM-120k 数据集路径（json 或 jsonl） |
| `--video_base_dir` | 必填 | 视频帧目录。每个 video_id 对应一个子目录，里面放帧图片 |
| `--output_path` | `cold_start_data.json` | 生成的 cold start 数据输出路径 |
| `--ns` | `5000` | 从数据集中采样多少个样本 |
| `--k` | `3` | 每个样本生成几条推理路径（取最高分的那条） |
| `--num_frames` | `16` | 每个视频抽取的帧数 |
| `--seed` | `42` | 随机种子 |

**输出**：`cold_start_data.json`，约 2500 条带 `<think>...</think><answer>...</answer>` 的数据。

---

## Stage 2b：Cold Start 微调

用上一步生成的数据，对 SFT 模型微调 200 steps，教它学会 `<think>` 输出格式。

**运行**：

```bash
python stage2b_cold_start_train.py \
    --sft_checkpoint_path      ./sft_output \
    --cold_start_data_path     ./cold_start_data.json \
    --video_base_dir           ./videos \
    --output_dir               ./cold_start_output \
    --num_steps                200 \
    --batch_size               4 \
    --gradient_accumulation_steps  4
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sft_checkpoint_path` | 必填 | Stage 1 SFT 输出的模型目录 |
| `--cold_start_data_path` | 必填 | Stage 2a 生成的 `cold_start_data.json` |
| `--video_base_dir` | 必填 | 视频帧目录（同上） |
| `--output_dir` | `./cold_start_output` | 微调后模型保存路径 |
| `--model_type` | `spatial-mllm` | 模型类型，和 Stage 1 保持一致 |
| `--num_steps` | `200` | 训练步数（论文设 200） |
| `--learning_rate` | `1e-5` | 学习率 |
| `--batch_size` | `4` | 单卡 batch size |
| `--gradient_accumulation_steps` | `4` | 梯度累积步数（等效 global batch = 16） |
| `--bf16` | `True` | 是否用 bf16 混合精度 |

**输出**：`./cold_start_output/` 目录下保存的模型 checkpoint。

---

## Stage 3：GRPO 训练

核心 RL 阶段。每个 step 对一个问题采样 8 条输出，打分后用 GRPO 优化。

**运行**：

```bash
python stage3_grpo_train.py \
    --cold_start_checkpoint_path   ./cold_start_output \
    --training_data_path           ./spatial_mllm_120k.json \
    --video_base_dir               ./videos \
    --output_dir                   ./grpo_output \
    --total_steps                  1000 \
    --group_size                   8 \
    --kl_coefficient               0.04 \
    --learning_rate                1e-6
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cold_start_checkpoint_path` | 必填 | Stage 2b 输出的模型目录 |
| `--training_data_path` | 必填 | 训练数据（Spatial-MLLM-120k） |
| `--video_base_dir` | 必填 | 视频帧目录 |
| `--output_dir` | `./grpo_output` | 输出目录（checkpoints + 日志） |
| `--model_type` | `spatial-mllm` | 模型类型 |
| `--group_size` | `8` | 每个问题采样几条输出（论文设 8） |
| `--clip_epsilon` | `0.2` | PPO clip 范围 |
| `--kl_coefficient` | `0.04` | KL 散度惩罚系数（论文设 0.04） |
| `--sampling_temperature` | `1.0` | 采样温度（论文设 1.0） |
| `--learning_rate` | `1e-6` | 学习率（论文设 1e-6） |
| `--total_steps` | `1000` | 总训练步数（论文设 1000） |
| `--max_new_tokens` | `1024` | 每条输出最大生成 token 数 |
| `--log_steps` | `10` | 每隔多少 steps 打印日志 |
| `--save_steps` | `200` | 每隔多少 steps 保存 checkpoint |

**输出**：
- `./grpo_output/checkpoint-200/`、`checkpoint-400/`…… 中间 checkpoint
- `./grpo_output/grpo_training_log.jsonl` 训练日志（reward、loss、completion length 等）
- `./grpo_output/` 最终模型

---

## 完整流程一键参考

按顺序跑以下四个命令即可从 SFT 后的模型一路跑到最终 GRPO 模型：

```bash
# 0. 设置 API key（Stage 2a 需要）
export DASHSCOPE_API_KEY=your_key

# 1. 生成 CoT 数据
python stage2_cold_start_data.py \
    --spatial_data_path ./spatial_mllm_120k.json \
    --video_base_dir    ./videos

# 2. Cold Start 微调
python stage2b_cold_start_train.py \
    --sft_checkpoint_path   ./sft_output \
    --cold_start_data_path  ./cold_start_data.json \
    --video_base_dir        ./videos

# 3. GRPO 训练
python stage3_grpo_train.py \
    --cold_start_checkpoint_path  ./cold_start_output \
    --training_data_path          ./spatial_mllm_120k.json \
    --video_base_dir              ./videos
```

---

## 常见问题

**Q: Stage 2a 跑到一半崩了怎么办？**
目前没有断点续跑机制。可以把 `--ns` 减小，分批生成后手动合并 json，或者自己在主循环外加一层已处理 idx 的 skip 逻辑。

**Q: Stage 3 显存不够？**
GRPO 需要同时放 policy model 和 ref model，显存大概是单模型的 2x。如果不够，可以把 ref model 放到 CPU，在计算 KL 时临时 move 到 GPU（会慢一些但能跑）。

**Q: 视频帧目录格式要求？**
每个 video_id 对应一个子目录，里面放 jpg/png 帧文件，文件名排序后会均匀采样。例如：
```
videos/
├── scene_001/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── scene_002/
│   └── ...
```

**Q: 模型类型 `model_type` 可以填什么？**
和 Stage 1 一致，支持 `spatial-mllm`、包含 `qwen2.5` 的字符串、或其他（默认走 Qwen2-VL）。
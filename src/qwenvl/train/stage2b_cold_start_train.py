# ==============================================================================
# Stage 2b: Cold Start 微调训练脚本
# ==============================================================================
# 前提：已经通过 stage2_cold_start_data.py 生成了 cold_start_data.json
#
# 功能：用带 <think>...</think><answer>...</answer> 格式的 CoT 数据
#       对 SFT 后的模型继续微调 200 steps，让它学会正确的推理输出格式。
#       这是进入 GRPO 阶段前必要的"格式对齐"步骤。
#
# 对应论文 Section 3.3 "RL Training" 段落：
#   "we first perform a simple cold start to help the model adapt to the
#    correct reasoning format"
#
# 与第一阶段 SFT 差别：
#   - 使用的数据不同（带 CoT 的 cold start 数据，约 2459 条）
#   - 训练步数更少（200 steps）
#   - 冻结策略相同（冻结视觉编码器，训练连接器+LLM）
#   - 损失函数相同（cross-entropy），但 target 序列变成带 <think> 的长序列
# ==============================================================================

import json
import os
import sys
import math
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 项目根目录加入路径
sys.path.append(str(Path(__file__).resolve().parents[0]))

# 导入第一阶段的模型加载和冻结工具
# NOTE: 假设 stage1 的代码可以作为模块导入。
#       如果不可以，需要将 get_model / set_model 复制到这里。
from train_sft import get_model, set_model
from data_utils import load_video_frames


# ===========================================================================
# Cold Start 数据集类
# ===========================================================================
class ColdStartDataset(Dataset):
    """
    Cold Start 训练数据集。
    
    每个样本的 target 是带 CoT 推理的完整输出：
        <think> ... 推理过程 ... </think><answer> ... 答案 ... </answer>
    
    训练时的 loss 在 target 序列的所有 token 上计算（包括 think 部分），
    这样模型会学会生成完整的推理链。
    """

    def __init__(self, data_path: str, tokenizer, processor, video_base_dir: str,
                 model_max_length: int = 2048, num_frames: int = 16):
        """
        Args:
            data_path: cold_start_data.json 路径
            tokenizer: 模型的 tokenizer
            processor: Qwen2.5-VL 的 AutoProcessor
            video_base_dir: 视频帧目录
            model_max_length: 最大序列长度
            num_frames: 输入视频帧数
        """
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_base_dir = video_base_dir
        self.model_max_length = model_max_length
        self.num_frames = num_frames

        print(f"  Cold Start Dataset 加载完成: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # cold_start_data.json 的字段：video / question / task_type / target
        return {
            "question":   item["question"],
            "target":     item["target"],          # <think>...</think><answer>...</answer>
            "video_path": item["video"],           # mp4 路径，如 scannet/videos/scene0000_00.mp4
            "task_type":  item.get("task_type", "verbal"),
        }


# ===========================================================================
# Data Collator — 负责将 batch 中的样本编码为模型输入
# ===========================================================================
class ColdStartDataCollator:
    """
    将 raw 样本编码为模型可处理的 input_ids / labels。
    
    输入格式（Qwen2.5-VL chat template）：
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>
        <|im_start|>user\n<video>{video frames}</video>\n{question + type_template}<|im_end|>
        <|im_start|>assistant\n{target}<|im_end|>
    
    Labels 设置：
        - 用户输入部分的 labels 设为 -100（不计入 loss）
        - 只有 assistant 回复部分（即 <think>...<answer>...） 参与 loss 计算
    """

    def __init__(self, tokenizer, processor, video_base_dir: str, num_frames: int = 16,
                 model_max_length: int = 2048):
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_base_dir = video_base_dir
        self.num_frames = num_frames
        self.model_max_length = model_max_length

    def __call__(self, batch: list[dict]) -> dict:
        from PIL import Image

        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for item in batch:
            # --- 构建 messages ---
            task_type = item.get("task_type", "verbal")
            type_template = get_type_template_for_grpo(task_type)

            user_text = item["question"] + "\n" + type_template

            # 从 mp4 抽帧
            video_frames = load_video_frames(
                item["video_path"], self.video_base_dir, self.num_frames
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_frames},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": item["target"],  # CoT 目标输出
                },
            ]

            # --- 用 processor 编码（带 chat template）---
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            encoded = self.processor(
                text, videos=video_frames,
                truncation=True, max_length=self.model_max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]

            # --- 构建 labels（只让 assistant 部分参与 loss）---
            labels = input_ids.clone()

            # 找到 assistant 回复开始的位置
            # Qwen2.5-VL 的 assistant token 标记为 "<|im_start|>assistant\n"
            assistant_token_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            assistant_start = find_subsequence(input_ids.tolist(), assistant_token_ids)

            if assistant_start is not None:
                # assistant 开始位置之前的全部设为 -100
                labels[:assistant_start + len(assistant_token_ids)] = -100
            else:
                # 找不到的话整个序列都不算 loss（不应该发生）
                labels[:] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        # --- Padding ---
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_input_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (ids, lbls, mask) in enumerate(zip(all_input_ids, all_labels, all_attention_masks)):
            length = ids.shape[0]
            padded_input_ids[i, :length] = ids
            padded_labels[i, :length] = lbls
            padded_attention_mask[i, :length] = mask

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


# ===========================================================================
# 辅助函数
# ===========================================================================
def find_subsequence(seq: list, subseq: list) -> Optional[int]:
    """在序列中查找子序列的起始位置"""
    sub_len = len(subseq)
    for i in range(len(seq) - sub_len + 1):
        if seq[i:i + sub_len] == subseq:
            return i
    return None


# load_video_frames 已移至 data_utils.py（从 mp4 直接抽帧）


def get_type_template_for_grpo(task_type: str) -> str:
    """返回 GRPO stage 的 type template（带 <think> 要求）"""
    templates = {
        "multiple_choice": (
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
            "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
            "expressions. It's encouraged to include self-reflection or verification in the "
            "reasoning process. Please provide your detailed reasoning between the <think> </think> "
            "tags, and then answer the question with the option's letter from the given choices "
            "(e.g., A, B, etc.) within the <answer> </answer> tags."
        ),
        "numerical": (
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
            "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
            "expressions. It's encouraged to include self-reflection or verification in the "
            "reasoning process. Please provide your detailed reasoning between the <think> </think> "
            "tags, and then answer the question with the only numerical value "
            "(e.g., 42, 3.14, etc.) within the <answer> </answer> tags."
        ),
        "verbal": (
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
            "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
            "expressions. It's encouraged to include self-reflection or verification in the "
            "reasoning process. Please provide your detailed reasoning between the <think> </think> "
            "tags, and then answer the question simply within the <answer> </answer> tags."
        ),
    }
    return templates.get(task_type, templates["verbal"])


# ===========================================================================
# 主训练函数
# ===========================================================================
def train_cold_start(
    # 模型路径（SFT 阶段输出的 checkpoint）
    sft_checkpoint_path: str,
    # 数据
    cold_start_data_path: str,
    video_base_dir: str,
    # 模型配置（与 SFT 阶段一致）
    model_type: str = "spatial-mllm",
    # 训练超参
    num_steps: int = 200,           # 论文: 200 steps
    learning_rate: float = 1e-5,    # 与 SFT 阶段相同的 lr schedule
    batch_size: int = 4,            # 根据显存调整
    gradient_accumulation_steps: int = 4,  # 等效 global batch size = 16
    model_max_length: int = 2048,
    num_frames: int = 16,
    # 输出
    output_dir: str = "./cold_start_output",
    # 硬件
    bf16: bool = True,
    local_rank: int = -1,
):
    """Cold Start 微调训练入口"""

    print("=" * 60)
    print("Stage 2b: Cold Start 微调训练")
    print("=" * 60)

    # ---------- 1. 加载 SFT 后的模型 ----------
    print("\n[1] 加载 SFT checkpoint...")

    # 构建一个简易的 model_args 对象传给 get_model
    class SimpleArgs:
        pass

    model_args = SimpleArgs()
    model_args.pretrained_model_name_or_path = sft_checkpoint_path
    model_args.model_type = model_type
    model_args.vggt_checkpoints_path = None       # SFT 后已经加载过了
    model_args.connector_type = "mlp"
    model_args.spatial_embeds_layer_idx = 12
    # 冻结配置：和论文一致，冻结视觉编码器，训练连接器+LLM
    model_args.tune_mm_vision = False
    model_args.tune_mm_connector = True
    model_args.tune_mm_llm = True
    model_args.tune_mm_spatial_encoder = False
    
    data_args = SimpleArgs()
    data_args.image_processor = None

    training_cfg = SimpleArgs()
    training_cfg.cache_dir = None
    training_cfg.bf16 = bf16
    training_cfg.output_dir = output_dir

    model, image_processor = get_model(
        model_args=model_args,
        data_args=data_args,
        training_args=training_cfg,
        attn_implementation="flash_attention_2",
    )

    # 应用参数冻结
    set_model(model_args, model)

    # 关闭 KV cache（训练时不需要）
    model.config.use_cache = False

    # ---------- 2. 加载 Tokenizer 和 Processor ----------
    print("\n[2] 加载 Tokenizer 和 Processor...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        sft_checkpoint_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(sft_checkpoint_path)

    # ---------- 3. 构建数据集和 Collator ----------
    print("\n[3] 构建数据集...")
    dataset = ColdStartDataset(
        data_path=cold_start_data_path,
        tokenizer=tokenizer,
        processor=processor,
        video_base_dir=video_base_dir,
        model_max_length=model_max_length,
        num_frames=num_frames,
    )

    collator = ColdStartDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        video_base_dir=video_base_dir,
        num_frames=num_frames,
        model_max_length=model_max_length,
    )

    # ---------- 4. 配置 HuggingFace Trainer ----------
    print("\n[4] 配置 Trainer...")

    # 计算 epochs（因为要精确控制 steps）
    # total_steps = num_epochs * (dataset_size / (batch_size * grad_accum))
    # 反算 epochs
    steps_per_epoch = math.ceil(len(dataset) / (batch_size * gradient_accumulation_steps))
    num_epochs = math.ceil(num_steps / steps_per_epoch)
    actual_steps = num_epochs * steps_per_epoch
    print(f"  数据量: {len(dataset)}, 每 epoch steps: {steps_per_epoch}")
    print(f"  目标 steps: {num_steps}, 实际将训练: {actual_steps} steps ({num_epochs} epochs)")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        # 步数控制
        max_steps=num_steps,                        # 精确控制为 200 steps
        # Batch
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 学习率
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=10,                            # 短 warmup
        # 精度
        bf16=bf16,
        # 保存
        save_strategy="steps",
        save_steps=num_steps,                       # 只在最后保存一次
        save_total_limit=1,
        # 日志
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        # 其他
        dataloader_num_workers=4,
        remove_unused_columns=False,                # 我们的 collator 自己处理字段
        report_to="none",                           # 关闭外部日志平台
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # ---------- 5. 开始训练 ----------
    print("\n[5] 开始 Cold Start 训练...")
    trainer.train()

    # ---------- 6. 保存 ----------
    print("\n[6] 保存模型...")
    trainer.save_model(output_dir)
    trainer.save_state()
    model.config.use_cache = True  # 恢复 KV cache
    print(f"  Cold Start 模型已保存至: {output_dir}")


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cold Start fine-tuning (Stage 2b)")
    parser.add_argument("--sft_checkpoint_path", type=str, required=True,
                        help="Stage 1 SFT 输出的模型路径")
    parser.add_argument("--cold_start_data_path", type=str, required=True,
                        help="stage2_cold_start_data.py 生成的 cold_start_data.json")
    parser.add_argument("--video_base_dir", type=str, required=True,
                        help="视频帧目录")
    parser.add_argument("--model_type", type=str, default="spatial-mllm")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./cold_start_output")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    train_cold_start(
        sft_checkpoint_path=args.sft_checkpoint_path,
        cold_start_data_path=args.cold_start_data_path,
        video_base_dir=args.video_base_dir,
        model_type=args.model_type,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        bf16=args.bf16,
    )
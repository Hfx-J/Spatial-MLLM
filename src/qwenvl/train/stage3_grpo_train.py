# ==============================================================================
# Stage 3: GRPO 训练脚本
# ==============================================================================
# 前提：已经完成 Cold Start 训练，模型已经学会 <think>...<answer>... 格式。
#
# 功能：实现 Group Relative Policy Optimization (GRPO)，
#       对模型进行 RL 训练以增强空间推理的长链思维能力。
#
# 对应论文：
#   - Equation (7): GRPO 目标函数
#   - Section 3.3 "RL Training"
#   - Appendix A.4: Reward 设计细节
#
# 训练流程（每个 step）：
#   1. 对 batch 中每个问题，从当前 policy 采样 G=8 条输出（rollout）
#   2. 用 reward 函数对每条输出打分
#   3. 在 group 内用均值和标准差归一化 reward → 得到 advantage A_i
#   4. 计算 policy ratio + clip + KL 约束 → GRPO loss
#   5. 反向传播更新模型
#
# 超参（来自论文 Section 4.1）：
#   - G = 8 (rollouts per question)
#   - temperature = 1.0 (采样温度)
#   - β = 0.04 (KL divergence 系数)
#   - lr = 1e-6
#   - ε = 0.2 (clip 范围，PPO 标准值)
#   - total steps = 1000
# ==============================================================================

import json
import os
import sys
import re
import math
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import AutoProcessor, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[0]))

from reward import compute_reward, compute_batch_rewards
from train_sft import get_model, set_model
from data_utils import parse_item, load_video_frames


# ===========================================================================
# GRPO 超参配置
# ===========================================================================
class GRPOConfig:
    """GRPO 训练超参配置，对应论文 Section 4.1"""

    def __init__(self):
        # --- 核心 GRPO 超参 ---
        self.group_size = 8              # G: 每个问题采样的输出数量
        self.clip_epsilon = 0.2          # ε: PPO-style clip 范围
        self.kl_coefficient = 0.04       # β: KL 散度惩罚系数（论文设 0.04）

        # --- 采样超参 ---
        self.sampling_temperature = 1.0  # 论文设 1.0
        self.sampling_top_p = 0.95
        self.max_new_tokens = 1024       # 生成的最大 token 数

        # --- 训练超参 ---
        self.learning_rate = 1e-6        # 论文设 1e-6
        self.total_steps = 1000          # 论文设 1000 steps
        self.batch_size = 1              # 每个 step 处理 1 个问题（但采样 G 条）
        self.gradient_accumulation_steps = 8
        self.warmup_steps = 50

        # --- Reward 权重 ---
        self.lambda_format = 1.0         # format reward 权重
        self.lambda_task = 1.0           # task reward 权重
        self.lambda_length = 0.1         # reasoning length reward 权重

        # --- 模型 ---
        self.bf16 = True
        self.num_frames = 16


# ===========================================================================
# Rollout: 从模型采样 G 条输出
# ===========================================================================
@torch.no_grad()
def rollout(
    model,
    processor,
    tokenizer,
    parsed: dict,
    video_frames: list,
    config: GRPOConfig,
) -> list[dict]:
    """
    对单个问题从当前 policy 采样 G 条独立输出。
    
    Args:
        model: 当前 policy 模型
        parsed: parse_item() 输出的字典（含 question, task_type 等）
        video_frames: 预处理的视频帧
        config: GRPO 配置
    
    Returns:
        G 条采样结果的列表
    """
    task_type = parsed["task_type"]
    type_template = get_grpo_type_template(task_type)
    user_text = parsed["question"] + "\n" + type_template

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # 编码输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text, videos=video_frames, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    samples = []
    for _ in range(config.group_size):
        # 采样生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.sampling_temperature,
            top_p=config.sampling_top_p,
            output_scores=True,           # 返回每步的 logits
            return_dict_in_generate=True, # 返回包含 scores 的字典
        )

        # 提取生成的 token ids（去掉输入部分）
        output_ids = outputs.sequences[0][input_length:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 从 scores 中计算每个生成 token 的 log_prob
        # outputs.scores 是每一步的 logits tuple，长度 = 生成序列长度
        log_probs = compute_log_probs_from_scores(outputs.scores, output_ids)

        samples.append({
            "output_ids": output_ids,
            "output_text": output_text,
            "log_probs": log_probs,         # shape: (seq_len,)
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })

    return samples


def compute_log_probs_from_scores(scores: tuple, output_ids: torch.Tensor) -> torch.Tensor:
    """
    从 model.generate 返回的 scores 中计算实际采样 token 的 log prob。
    
    Args:
        scores: tuple of logits, 每个元素 shape (1, vocab_size)
        output_ids: 实际生成的 token ids, shape (seq_len,)
    
    Returns:
        log_probs: shape (seq_len,), 每个位置实际 token 的 log probability
    """
    log_probs_list = []
    for step_idx, step_logits in enumerate(scores):
        # step_logits: (1, vocab_size)
        log_softmax = F.log_softmax(step_logits[0], dim=-1)  # (vocab_size,)
        token_id = output_ids[step_idx]
        log_probs_list.append(log_softmax[token_id])

    return torch.stack(log_probs_list)  # (seq_len,)


# ===========================================================================
# 计算 Reference model 的 log probs（用于 KL 约束）
# ===========================================================================
@torch.no_grad()
def compute_ref_log_probs(
    ref_model,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    用 reference model（frozen 的 policy 初始版本）计算同一序列的 log prob。
    用于 KL(π_θ || π_ref) 的计算。
    
    Args:
        ref_model: reference model（不更新参数）
        input_ids: 输入 token ids
        output_ids: 生成的 token ids
        attention_mask: 输入 attention mask
    
    Returns:
        ref_log_probs: shape (output_seq_len,)
    """
    # 拼接输入和输出作为完整序列
    full_ids = torch.cat([input_ids[0], output_ids], dim=0).unsqueeze(0)  # (1, full_len)

    # 构建 attention mask（输入部分 + 输出部分全为 1）
    full_mask = torch.ones_like(full_ids)

    # forward pass
    outputs = ref_model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits  # (1, full_len, vocab_size)

    # 只取输出部分对应的 logits（shifted by 1，因为是 next-token prediction）
    input_len = input_ids.shape[-1]
    # logits[0, input_len-1 : input_len-1+output_len] 对应输出序列每个 token 的预测
    output_len = output_ids.shape[0]
    output_logits = logits[0, input_len - 1: input_len - 1 + output_len, :]  # (output_len, vocab)

    log_softmax = F.log_softmax(output_logits, dim=-1)  # (output_len, vocab)
    ref_log_probs = log_softmax.gather(1, output_ids.unsqueeze(-1)).squeeze(-1)  # (output_len,)

    return ref_log_probs


# ===========================================================================
# 计算当前 policy 的 log probs（用于 ratio）
# ===========================================================================
def compute_policy_log_probs(
    model,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    用当前 policy model 对已生成序列做 forward，得到 log probs。
    和 compute_ref_log_probs 逻辑相同，但这个要计算梯度。
    """
    full_ids = torch.cat([input_ids[0], output_ids], dim=0).unsqueeze(0)
    full_mask = torch.ones_like(full_ids)

    outputs = model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits

    input_len = input_ids.shape[-1]
    output_len = output_ids.shape[0]
    output_logits = logits[0, input_len - 1: input_len - 1 + output_len, :]

    log_softmax = F.log_softmax(output_logits, dim=-1)
    policy_log_probs = log_softmax.gather(1, output_ids.unsqueeze(-1)).squeeze(-1)

    return policy_log_probs


# ===========================================================================
# 计算 GRPO Loss（论文 Eq.7）
# ===========================================================================
def compute_grpo_loss(
    model,
    ref_model,
    samples: list[dict],
    rewards: list[float],
    config: GRPOConfig,
) -> torch.Tensor:
    """
    对单个问题的 G 条采样计算 GRPO loss。
    
    对应论文 Eq.7:
        J_GRPO = E[ (1/G) Σ min(ratio * A_i, clip(ratio, 1±ε) * A_i) - β * KL ]
    
    其中:
        ratio = π_θ(o_i | q) / π_θ_old(o_i | q)
        A_i = (r_i - mean(r)) / std(r)        # group 内归一化的 advantage
    
    Args:
        model: 当前 policy 模型（要更新的）
        ref_model: reference model（冻结，用于 KL）
        samples: G 条采样结果
        rewards: G 条对应的 reward 值
        config: GRPO 配置
    
    Returns:
        loss: 标量 loss（取负号，因为要最大化 J）
    """
    G = len(samples)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)

    # ---------- 1. 计算 Advantage（group 内归一化）----------
    # A_i = (r_i - mean(r)) / std(r)
    # 如果所有 reward 相同，std=0，advantage 全为 0
    reward_mean = rewards_tensor.mean()
    reward_std = rewards_tensor.std()
    if reward_std < 1e-8:
        # 所有 reward 相同，无法学习，跳过
        advantages = torch.zeros_like(rewards_tensor)
    else:
        advantages = (rewards_tensor - reward_mean) / reward_std

    # ---------- 2. 逐条计算 ratio、clip loss 和 KL ----------
    policy_losses = []
    kl_losses = []

    for i, sample in enumerate(samples):
        output_ids = sample["output_ids"]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        old_log_probs = sample["log_probs"]  # rollout 时记录的 log probs
        advantage = advantages[i]

        # -- 当前 policy 的 log probs --
        new_log_probs = compute_policy_log_probs(
            model, input_ids, output_ids, attention_mask
        )

        # -- Reference model 的 log probs（用于 KL）--
        ref_log_probs = compute_ref_log_probs(
            ref_model, input_ids, output_ids, attention_mask
        )

        # -- 序列级别的 log prob（对所有 token 求和）--
        # ratio 在序列级别计算（取 sum log prob 的差）
        new_seq_log_prob = new_log_probs.sum()
        old_seq_log_prob = old_log_probs.sum().detach()
        ref_seq_log_prob = ref_log_probs.sum()

        # -- Policy Ratio --
        # log(ratio) = log π_θ - log π_θ_old
        log_ratio = new_seq_log_prob - old_seq_log_prob
        ratio = torch.exp(log_ratio)

        # -- Clipped Surrogate Loss（PPO 式）--
        # surrogate1 = ratio * A
        # surrogate2 = clip(ratio, 1-ε, 1+ε) * A
        # loss = -min(surrogate1, surrogate2)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage
        policy_loss = -torch.min(surrogate1, surrogate2)
        policy_losses.append(policy_loss)

        # -- KL Divergence: KL(π_θ || π_ref) --
        # 用序列级别近似：KL ≈ log π_θ - log π_ref
        # 更精确的 token 级别 KL: Σ exp(log_π_θ) * (log_π_θ - log_π_ref)
        # 这里用简化版本（与 DeepSeek-R1 实现一致）
        kl = (new_seq_log_prob - ref_seq_log_prob)
        kl_losses.append(kl)

    # ---------- 3. 汇总 loss ----------
    # 对 G 条取平均
    avg_policy_loss = torch.stack(policy_losses).mean()
    avg_kl = torch.stack(kl_losses).mean()

    # 总 loss = policy_loss + β * KL
    # policy_loss 已经取了负号（因为要最大化 surrogate），所以直接加
    total_loss = avg_policy_loss + config.kl_coefficient * avg_kl

    return total_loss, {
        "policy_loss": avg_policy_loss.item(),
        "kl_loss": avg_kl.item(),
        "reward_mean": reward_mean.item(),
        "reward_std": reward_std.item(),
        "avg_reward": rewards_tensor.mean().item(),
        "avg_completion_length": np.mean([s["output_ids"].shape[0] for s in samples]),
    }


# ===========================================================================
# 辅助函数
# ===========================================================================
def get_grpo_type_template(task_type: str) -> str:
    """GRPO stage 的 prompt type template"""
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


# load_video_frames 已移至 data_utils.py（从 mp4 直接抽帧）


# ===========================================================================
# 主训练循环
# ===========================================================================
def train_grpo(
    # 模型路径（Cold Start 阶段输出）
    cold_start_checkpoint_path: str,
    # 数据
    training_data_path: str,        # Spatial-MLLM-120k（或子集）
    video_base_dir: str,
    # 模型配置
    model_type: str = "spatial-mllm",
    # GRPO 超参
    group_size: int = 8,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.04,
    sampling_temperature: float = 1.0,
    learning_rate: float = 1e-6,
    total_steps: int = 1000,
    max_new_tokens: int = 1024,
    # 输出
    output_dir: str = "./grpo_output",
    # 日志
    log_steps: int = 10,
    save_steps: int = 200,
):
    """GRPO 训练主流程"""

    print("=" * 60)
    print("Stage 3: GRPO 训练")
    print("=" * 60)

    # ---------- 配置 ----------
    config = GRPOConfig()
    config.group_size = group_size
    config.clip_epsilon = clip_epsilon
    config.kl_coefficient = kl_coefficient
    config.sampling_temperature = sampling_temperature
    config.max_new_tokens = max_new_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. 加载模型 ----------
    print("\n[1] 加载 Policy 和 Reference 模型...")

    class SimpleArgs:
        pass

    model_args = SimpleArgs()
    model_args.pretrained_model_name_or_path = cold_start_checkpoint_path
    model_args.model_type = model_type
    model_args.vggt_checkpoints_path = None
    model_args.connector_type = "mlp"
    model_args.spatial_embeds_layer_idx = 12
    # GRPO 阶段同样冻结视觉编码器
    model_args.tune_mm_vision = False
    model_args.tune_mm_connector = True
    model_args.tune_mm_llm = True
    model_args.tune_mm_spatial_encoder = False

    data_args = SimpleArgs()
    data_args.image_processor = None
    training_cfg = SimpleArgs()
    training_cfg.cache_dir = None
    training_cfg.bf16 = True
    training_cfg.output_dir = output_dir

    # Policy model（要更新的）
    policy_model, image_processor = get_model(
        model_args=model_args, data_args=data_args,
        training_args=training_cfg, attn_implementation="flash_attention_2",
    )
    set_model(model_args, policy_model)
    policy_model.config.use_cache = True   # generate 时需要 KV cache
    policy_model.to(device)
    policy_model.train()

    # Reference model（冻结的副本，用于 KL 约束）
    # 深拷贝 policy 的初始状态作为 ref
    print("  创建 Reference model (frozen copy)...")
    ref_model, _ = get_model(
        model_args=model_args, data_args=data_args,
        training_args=training_cfg, attn_implementation="flash_attention_2",
    )
    ref_model.to(device)
    ref_model.eval()
    # 冻结 ref model 的所有参数
    for p in ref_model.parameters():
        p.requires_grad = False
    print("  Reference model 已冻结")

    # ---------- 2. Tokenizer + Processor ----------
    print("\n[2] 加载 Tokenizer / Processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        cold_start_checkpoint_path, use_fast=False, padding_side="right"
    )
    processor = AutoProcessor.from_pretrained(cold_start_checkpoint_path)

    # ---------- 3. 加载训练数据 ----------
    print("\n[3] 加载训练数据...")
    if training_data_path.endswith(".jsonl"):
        data = []
        with open(training_data_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(training_data_path) as f:
            data = json.load(f)
    print(f"  训练数据总量: {len(data)}")

    # ---------- 4. Optimizer ----------
    print("\n[4] 配置 Optimizer...")
    # 只优化需要梯度的参数
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # 学习率调度（简单线性 warmup + decay）
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return max(0.1, 1.0 - (step - config.warmup_steps) / (total_steps - config.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- 5. 输出目录 ----------
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, "grpo_training_log.jsonl"), "w")

    # ---------- 6. 训练循环 ----------
    print(f"\n[5] 开始 GRPO 训练 (total_steps={total_steps}, G={group_size})...")
    print("-" * 60)

    # 随机打乱数据、循环迭代
    rng = np.random.default_rng(42)
    data_indices = list(range(len(data)))
    rng.shuffle(data_indices)
    data_iter_idx = 0

    # 统计量
    running_reward = []
    running_completion_length = []

    policy_model.train()

    for step in range(total_steps):
        # --- 获取下一个样本 ---
        if data_iter_idx >= len(data_indices):
            # 一轮数据用完，重新打乱
            rng.shuffle(data_indices)
            data_iter_idx = 0

        sample_idx = data_indices[data_iter_idx]
        data_iter_idx += 1
        item = data[sample_idx]

        # 解析原始字段 → 统一格式
        parsed = parse_item(item)

        # --- 从 mp4 抽帧 ---
        video_frames = load_video_frames(
            parsed["video_path"], video_base_dir, config.num_frames
        )

        # =====================================================================
        # Step A: Rollout — 从当前 policy 采样 G 条输出
        # =====================================================================
        policy_model.eval()  # eval mode 做 generate
        with torch.no_grad():
            samples = rollout(
                model=policy_model,
                processor=processor,
                tokenizer=tokenizer,
                parsed=parsed,
                video_frames=video_frames,
                config=config,
            )

        # =====================================================================
        # Step B: Reward 计算
        # =====================================================================
        gt_answer = parsed["answer"]
        task_type  = parsed["task_type"]

        rewards = []
        for s in samples:
            r = compute_reward(
                pred_answer=extract_answer(s["output_text"]),
                gt_answer=gt_answer,
                task_type=task_type,
                output_text=s["output_text"],
                lambda1=config.lambda_format,
                lambda2=config.lambda_task,
                lambda3=config.lambda_length,
            )
            rewards.append(r)

        running_reward.extend(rewards)
        running_completion_length.append(
            np.mean([s["output_ids"].shape[0] for s in samples])
        )

        # =====================================================================
        # Step C: 计算 GRPO Loss 并反传
        # =====================================================================
        policy_model.train()  # 切回 train mode

        loss, metrics = compute_grpo_loss(
            model=policy_model,
            ref_model=ref_model,
            samples=samples,
            rewards=rewards,
            config=config,
        )

        # 梯度累积
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # =====================================================================
        # Step D: 日志
        # =====================================================================
        if (step + 1) % log_steps == 0:
            avg_reward = np.mean(running_reward[-group_size * log_steps:])
            avg_length = np.mean(running_completion_length[-log_steps:])
            current_lr = optimizer.param_groups[0]["lr"]

            log_entry = {
                "step": step + 1,
                "loss": loss.item() * config.gradient_accumulation_steps,
                "policy_loss": metrics["policy_loss"],
                "kl_loss": metrics["kl_loss"],
                "avg_reward": float(avg_reward),
                "reward_std": metrics["reward_std"],
                "avg_completion_length": float(avg_length),
                "learning_rate": current_lr,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            print(
                f"  Step {step+1:>5d}/{total_steps} | "
                f"Loss: {log_entry['loss']:.4f} | "
                f"Reward: {avg_reward:.4f} | "
                f"KL: {metrics['kl_loss']:.4f} | "
                f"AvgLen: {avg_length:.0f} | "
                f"LR: {current_lr:.2e}"
            )

        # =====================================================================
        # Step E: 保存 checkpoint
        # =====================================================================
        if (step + 1) % save_steps == 0 or (step + 1) == total_steps:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # 保存优化器状态（用于恢复训练）
            torch.save({
                "step": step + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, os.path.join(ckpt_dir, "training_state.pt"))
            print(f"  Checkpoint 保存至: {ckpt_dir}")

    # ---------- 7. 训练结束，保存最终模型 ----------
    print("\n[6] 训练完成，保存最终模型...")
    policy_model.config.use_cache = True
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_file.close()
    print(f"  最终模型保存至: {output_dir}")
    print("=" * 60)
    print("  GRPO 训练完成!")
    print("=" * 60)


# ===========================================================================
# 辅助
# ===========================================================================
def extract_answer(text: str) -> str:
    """从输出文本中提取 <answer>...</answer> 内的内容"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training (Stage 3)")
    parser.add_argument("--cold_start_checkpoint_path", type=str, required=True,
                        help="Cold Start 阶段输出的模型路径")
    parser.add_argument("--training_data_path", type=str, required=True,
                        help="训练数据路径 (Spatial-MLLM-120k)")
    parser.add_argument("--video_base_dir", type=str, required=True,
                        help="视频帧目录")
    parser.add_argument("--model_type", type=str, default="spatial-mllm")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--kl_coefficient", type=float, default=0.04)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()

    train_grpo(
        cold_start_checkpoint_path=args.cold_start_checkpoint_path,
        training_data_path=args.training_data_path,
        video_base_dir=args.video_base_dir,
        model_type=args.model_type,
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        kl_coefficient=args.kl_coefficient,
        sampling_temperature=args.sampling_temperature,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
    )
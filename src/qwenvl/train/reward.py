# ==============================================================================
# reward.py — 共用的 Reward 计算模块
# ==============================================================================
# Cold Start 和 GRPO 两个阶段共用此模块。
#
# 对应论文 Equation (13)~(16) 和 Appendix A.4：
#   Reward = λ1 * R_format + λ2 * R_task
#
#   R_task 按题目类型分三种：
#     - multiple_choice: Exact Match（选项字母精确匹配）
#     - numerical:       Mean Relative Accuracy (MRA)
#     - verbal:          Levenshtein 模糊匹配
#
# 此外还包含 reasoning_length_reward（鼓励更长的推理过程）
# ==============================================================================

import re
import math
from typing import Optional


# ===========================================================================
# 1. Format Reward — 检查输出是否符合 <think>...</think><answer>...</answer> 格式
# ===========================================================================
def compute_format_reward(output_text: str) -> float:
    """
    检查模型输出是否包含合法的 <think> 和 <answer> 标签。
    
    Returns:
        1.0 如果格式正确（同时包含 think 和 answer 标签）
        0.0 否则
    """
    has_think = bool(re.search(r"<think>.*?</think>", output_text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", output_text, re.DOTALL))
    return 1.0 if (has_think and has_answer) else 0.0


# ===========================================================================
# 2. Task-Specific Reward
# ===========================================================================

# ---------- 2a. Multiple Choice: Exact Match（论文 Eq.14）----------
def compute_mc_reward(pred_answer: str, gt_answer: str) -> float:
    """
    选择题 reward：精确匹配选项字母。
    
    对 pred 和 gt 都做 normalize（去空白、大写统一）后比较。
    对应论文 Eq.14: R_MC = I(ψ(A_pred) = ψ(A_gt))
    """
    return 1.0 if normalize_choice(pred_answer) == normalize_choice(gt_answer) else 0.0


def normalize_choice(text: str) -> str:
    """提取并 normalize 选项字母（如 'A', 'B' 等）"""
    text = text.strip().upper()
    # 尝试提取单个选项字母
    match = re.match(r"^([A-Z])", text)
    if match:
        return match.group(1)
    return text


# ---------- 2b. Numerical: Mean Relative Accuracy (论文 Eq.15)----------
# 论文使用的置信度阈值集合
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
EPS = 1e-8  # 防止除以零


def compute_numerical_reward(pred_answer: str, gt_answer: str) -> float:
    """
    数值题 reward：Mean Relative Accuracy (MRA)。
    
    对应论文 Eq.15:
        R_MRA = (1/|T|) * Σ_{τ∈T} I( |α(A_pred) - α(A_gt)| / (|α(A_gt)| + ε) < τ )
    
    其中 T = {0.50, 0.55, ..., 0.95} 是一组相对误差阈值。
    
    逻辑：相对误差越小，通过的阈值越多，reward 越高。
      - 完全正确（误差=0）-> reward = 1.0
      - 误差 < 0.05       -> 通过所有阈值 -> reward = 1.0
      - 误差 = 0.30       -> 只通过 τ > 0.30 的阈值 -> reward ≈ 0.6
      - 误差 >= 0.95      -> 不通过任何阈值 -> reward = 0.0
    """
    pred_val = parse_number(pred_answer)
    gt_val = parse_number(gt_answer)

    # 如果无法解析数值，reward = 0
    if pred_val is None or gt_val is None:
        return 0.0

    # 计算相对误差
    relative_error = abs(pred_val - gt_val) / (abs(gt_val) + EPS)

    # 统计通过多少个阈值
    # 注意：论文里的阈值 τ 是 "相对误差 < τ 时算正确"
    # 即 τ=0.5 对应允许 50% 的误差，τ=0.95 对应允许 95% 的误差
    # 所以 τ 越大越容易通过
    passed = sum(1 for tau in THRESHOLDS if relative_error < tau)
    return passed / len(THRESHOLDS)


def parse_number(text: str) -> Optional[float]:
    """从文本中提取数值"""
    text = text.strip()
    # 尝试直接转换
    try:
        return float(text)
    except ValueError:
        pass
    # 用正则提取第一个数字（含负号、小数点）
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


# ---------- 2c. Verbal: Levenshtein 模糊匹配（论文 Eq.16）----------
def compute_verbal_reward(pred_answer: str, gt_answer: str) -> float:
    """
    开放式回答 reward：基于 Levenshtein 距离的归一化相似度。
    
    对应论文 Eq.16:
        R_Verbal = 1 - D_Lev(ϕ(A_pred), ϕ(A_gt)) / (|ϕ(A_pred)| + |ϕ(A_gt)|)
    
    返回值范围 [0, 1]，越接近 1 表示越相似。
    """
    pred_norm = normalize_verbal(pred_answer)
    gt_norm = normalize_verbal(gt_answer)

    if len(pred_norm) == 0 and len(gt_norm) == 0:
        return 1.0
    if len(pred_norm) == 0 or len(gt_norm) == 0:
        return 0.0

    dist = levenshtein_distance(pred_norm, gt_norm)
    similarity = 1.0 - dist / (len(pred_norm) + len(gt_norm))
    return max(0.0, similarity)  # 确保非负


def normalize_verbal(text: str) -> str:
    """对开放式回答做文本 normalize：去空白、小写"""
    return text.strip().lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的 Levenshtein 编辑距离。
    使用动态规划实现，时间复杂度 O(m*n)。
    
    如果项目环境中有 Levenshtein 库，可以替换为：
        from Levenshtein import distance as levenshtein_distance
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = s1[:i] 和 s2[:j] 之间的编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # base case
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充 dp 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需操作
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 删除
                    dp[i][j - 1],      # 插入
                    dp[i - 1][j - 1],  # 替换
                )

    return dp[m][n]


# ===========================================================================
# 3. Reasoning Length Reward（鼓励更长的推理，来自 Video-R1）
# ===========================================================================
def compute_reasoning_length_reward(
    output_text: str,
    min_length: int = 50,
    max_length: int = 1024,
) -> float:
    """
    鼓励模型生成更长的推理过程的 reward。
    
    提取 <think>...</think> 中的内容长度：
      - 长度 < min_length: reward = 0
      - min_length <= 长度 <= max_length: 线性插值 [0, 1]
      - 长度 > max_length: reward = 1.0（不惩罚过长，因为论文观察到长 CoT 有益）
    
    NOTE: 具体的 min/max 阈值可根据实验调整。
    """
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    if not think_match:
        return 0.0

    think_length = len(think_match.group(1).strip())

    if think_length < min_length:
        return 0.0
    elif think_length <= max_length:
        return (think_length - min_length) / (max_length - min_length)
    else:
        return 1.0


# ===========================================================================
# 4. 总 Reward 计算入口
# ===========================================================================
def compute_reward(
    pred_answer: str,
    gt_answer: str,
    task_type: str,
    output_text: Optional[str] = None,  # 完整输出文本（用于 format + length reward）
    lambda1: float = 1.0,               # format reward 权重
    lambda2: float = 1.0,               # task reward 权重
    lambda3: float = 0.1,               # reasoning length reward 权重（较小，辅助信号）
) -> float:
    """
    计算总 reward。对应论文 Eq.13:
        Reward = λ1 * R_format + λ2 * R_task [+ λ3 * R_length]
    
    Args:
        pred_answer: 模型预测的答案（已从 <answer> 中提取）
        gt_answer: 真实答案
        task_type: 题目类型 ("multiple_choice" / "numerical" / "verbal")
        output_text: 模型完整输出（如果提供则计算 format 和 length reward）
        lambda1/2/3: 各项 reward 的权重
    
    Returns:
        总 reward 值
    """
    total_reward = 0.0

    # --- Format Reward ---
    if output_text is not None:
        r_format = compute_format_reward(output_text)
        total_reward += lambda1 * r_format
    # 如果没有 output_text（如 cold start 阶段只评估答案质量），跳过 format reward

    # --- Task-Specific Reward ---
    if task_type == "multiple_choice":
        r_task = compute_mc_reward(pred_answer, gt_answer)
    elif task_type == "numerical":
        r_task = compute_numerical_reward(pred_answer, gt_answer)
    elif task_type == "verbal":
        r_task = compute_verbal_reward(pred_answer, gt_answer)
    else:
        # 未知类型，fallback 到 verbal
        r_task = compute_verbal_reward(pred_answer, gt_answer)

    total_reward += lambda2 * r_task

    # --- Reasoning Length Reward（仅 GRPO 阶段使用）---
    if output_text is not None and lambda3 > 0:
        r_length = compute_reasoning_length_reward(output_text)
        total_reward += lambda3 * r_length

    return total_reward


# ===========================================================================
# 5. Batch Reward 计算（GRPO 阶段使用）
# ===========================================================================
def compute_batch_rewards(
    predictions: list[str],     # 模型生成的完整输出列表
    gt_answers: list[str],      # 对应的真实答案列表
    task_types: list[str],      # 对应的题目类型列表
    **kwargs,
) -> list[float]:
    """
    批量计算 reward，用于 GRPO 的 rollout 评分。
    
    Args:
        predictions: 模型生成的完整文本（含 <think><answer> 标签）
        gt_answers: 真实答案列表
        task_types: 题目类型列表
    
    Returns:
        reward 列表，与 predictions 等长
    """
    import re
    rewards = []
    for pred_text, gt, t_type in zip(predictions, gt_answers, task_types):
        # 从完整输出中提取 answer
        answer_match = re.search(r"<answer>(.*?)</answer>", pred_text, re.DOTALL)
        pred_answer = answer_match.group(1).strip() if answer_match else pred_text.strip()

        reward = compute_reward(
            pred_answer=pred_answer,
            gt_answer=gt,
            task_type=t_type,
            output_text=pred_text,
            **kwargs,
        )
        rewards.append(reward)

    return rewards
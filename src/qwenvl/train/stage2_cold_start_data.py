# ==============================================================================
# Stage 2: Cold Start 数据构建脚本（API 版本）
# ==============================================================================
# 将本地加载 Qwen2.5-VL-72B 替换为通过 DashScope API 调用。
#
# 主要变化：
#   - 帧图像通过 base64 编码后作为 image_url 传入
#   - 使用 enable_thinking=True，模型原生返回 reasoning_content + content
#     → reasoning_content 直接作为 thinking，无需自己解析 <think> 标签
#   - 加入 retry 机制处理网络抖动和速率限制
#   - 用 asyncio 做并发加速（可控制并发数避免被限流）
#
# 环境变量：
#   DASHSCOPE_API_KEY — 通义千问 API key
# ==============================================================================

import json
import os
import sys
import re
import base64
import time
import asyncio
import numpy as np
from io import BytesIO
from pathlib import Path
from collections import defaultdict

from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).resolve().parents[0]))
from reward import compute_reward
from data_utils import parse_item, load_video_frames


# ===========================================================================
# 配置
# ===========================================================================
MODEL_NAME = "qwen3-vl-plus"          # 可切换为其他模型

MAX_CONCURRENT = 5                    # 并发数，避免限流
MAX_RETRIES = 5                       # 最大重试次数
RETRY_BASE_DELAY = 2.0                # 指数退避基础延迟（秒）
THINKING_BUDGET = 8192                # reasoning 最大 token 数

SYSTEM_PROMPT = "You are a helpful assistant."

TYPE_TEMPLATES = {
    "multiple_choice": (
        "Please provide your detailed reasoning, "
        "and then answer the question with the option's letter from the given choices "
        "(e.g., A, B, etc.) within the <answer> </answer> tags."
    ),
    "numerical": (
        "Please provide your detailed reasoning, "
        "and then answer the question with the only numerical value "
        "(e.g., 42, 3.14, etc.) within the <answer> </answer> tags."
    ),
    "verbal": (
        "Please provide your detailed reasoning, "
        "and then answer the question simply within the <answer> </answer> tags."
    ),
}


# ===========================================================================
# 图像编码
# ===========================================================================
def pil_to_base64_url(img) -> str:
    """PIL Image → data:image/jpeg;base64,... 格式的 URL"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# load_video_frames 已移至 data_utils.py（从 mp4 直接抽帧）


# ===========================================================================
# 构建 API messages
# ===========================================================================
def build_api_messages(parsed: dict, video_frames: list) -> list[dict]:
    """
    帧逐帧以 base64 image_url 传入，前加说明文字，后接问题+格式要求。
    parsed 是经过 parse_item() 处理后的字典。
    """
    task_type = parsed["task_type"]
    type_template = TYPE_TEMPLATES.get(task_type, TYPE_TEMPLATES["verbal"])

    user_content = [
        {"type": "text", "text": "The following images are frames sampled from a video scene."},
    ]
    for frame in video_frames:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": pil_to_base64_url(frame)},
        })
    user_content.append({
        "type": "text",
        "text": f"{parsed['question']}\n\n{type_template}",
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ===========================================================================
# 异步 API 调用（retry + 信号量）
# ===========================================================================
async def call_api_with_retry(
    client: AsyncOpenAI,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    带指数退避的异步 API 调用。
    enable_thinking 模式需要 stream=True 才能拿到 reasoning_content。
    """
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                reasoning_content = ""
                answer_content = ""

                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True,
                    extra_body={
                        "enable_thinking": True,
                        "thinking_budget": THINKING_BUDGET,
                    },
                )

                async for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    elif delta.content:
                        answer_content += delta.content

            return {
                "reasoning_content": reasoning_content.strip(),
                "answer_content": answer_content.strip(),
            }

        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [Retry {attempt+1}/{MAX_RETRIES}] {e}. Wait {delay:.1f}s...")
            await asyncio.sleep(delay)

    raise RuntimeError(f"API failed after {MAX_RETRIES} retries")


async def generate_cot_paths_api(
    client: AsyncOpenAI,
    parsed: dict,
    video_frames: list,
    semaphore: asyncio.Semaphore,
    k: int = 3,
) -> list[dict]:
    """对单个样本并发生成 K 条路径。parsed 是 parse_item() 输出。"""
    messages = build_api_messages(parsed, video_frames)
    tasks = [call_api_with_retry(client, messages, semaphore) for _ in range(k)]
    results = await asyncio.gather(*tasks)

    paths = []
    for res in results:
        answer = extract_answer(res["answer_content"])
        paths.append({
            "thinking": res["reasoning_content"],
            "answer": answer,
            "raw_text": res["answer_content"],
        })
    return paths


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ===========================================================================
# Adaptive 筛选（论文 Algorithm 2）
# ===========================================================================
def adaptive_filter(scored_items: list[dict]) -> list[dict]:
    """每类取中位数阈值，保留 reward >= median 且 > 0 的样本"""
    type_rewards = defaultdict(list)
    for si in scored_items:
        type_rewards[si["task_type"]].append(si["best_reward"])

    type_thresholds = {}
    for t, rewards in type_rewards.items():
        type_thresholds[t] = float(np.median(rewards))
        print(f"  Type '{t}': n={len(rewards)}, "
              f"median={type_thresholds[t]:.4f}, mean={np.mean(rewards):.4f}")

    return [
        si for si in scored_items
        if si["best_reward"] >= type_thresholds[si["task_type"]] and si["best_reward"] > 0
    ]


# ===========================================================================
# 数据 IO
# ===========================================================================
def load_spatial_dataset(data_path: str) -> list[dict]:
    if data_path.endswith(".jsonl"):
        with open(data_path) as f:
            return [json.loads(line) for line in f]
    with open(data_path) as f:
        return json.load(f)


def sample_subset(data: list[dict], ns: int = 5000, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=min(ns, len(data)), replace=False)
    return [data[i] for i in indices]


# ===========================================================================
# 异步主流程
# ===========================================================================
async def _build_cold_start_async(
    spatial_data_path: str,
    video_base_dir: str,
    output_path: str,
    ns: int, k: int, seed: int, num_frames: int,
):
    print("=" * 60)
    print("Stage 2: Cold Start 数据构建 (DashScope API)")
    print("=" * 60)

    # 1. 采样
    print("\n[1/4] 加载并采样...")
    full_data = load_spatial_dataset(spatial_data_path)
    subset = sample_subset(full_data, ns=ns, seed=seed)
    print(f"  总量: {len(full_data)} → 采样: {len(subset)}")

    # 2. API client
    print("\n[2/4] 初始化 client...")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    print(f"  Model: {MODEL_NAME}, 并发: {MAX_CONCURRENT}")

    # 3. 生成 + 打分
    print(f"\n[3/4] 生成 K={k} 条路径...")
    scored_items = []
    t0 = time.time()

    for idx, item in enumerate(subset):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(idx, 1)) * (len(subset) - idx)
            print(f"  {idx}/{len(subset)} | {elapsed/60:.1f}min elapsed | ETA {eta/60:.1f}min")

        # 解析原始字段 → 统一格式
        parsed = parse_item(item)
        task_type  = parsed["task_type"]
        question   = parsed["question"]
        answer     = parsed["answer"]
        video_path = parsed["video_path"]

        # 从 mp4 抽帧
        video_frames = load_video_frames(video_path, video_base_dir, num_frames)

        try:
            paths = await generate_cot_paths_api(client, parsed, video_frames, semaphore, k)
        except Exception as e:
            print(f"  [SKIP] idx={idx}, video={video_path}: {e}")
            continue

        best_path, best_reward = None, -float("inf")
        for path in paths:
            r = compute_reward(
                pred_answer=path["answer"],
                gt_answer=answer,
                task_type=task_type,
            )
            if r > best_reward:
                best_reward, best_path = r, path

        scored_items.append({
            "item": item,          # 保留原始条目，后面存输出时用
            "parsed": parsed,      # 解析后的字段
            "best_path": best_path,
            "best_reward": best_reward,
            "task_type": task_type,
        })

    print(f"  成功: {len(scored_items)}/{len(subset)}")

    # 4. 筛选 + 保存
    print("\n[4/4] Adaptive 筛选...")
    filtered = adaptive_filter(scored_items)
    print(f"  筛选前: {len(scored_items)} → 筛选后: {len(filtered)}")

    dataset = []
    for si in filtered:
        p      = si["best_path"]
        parsed = si["parsed"]
        dataset.append({
            "video":      parsed["video_path"],       # 原始 mp4 路径，后续 stage 也用这个
            "question":   parsed["question"],
            "task_type":  si["task_type"],
            "gt_answer":  parsed["answer"],
            "target":     f"<think>{p['thinking']}</think><answer>{p['answer']}</answer>",
            "thinking":   p["thinking"],
            "answer":     p["answer"],
            "reward":     si["best_reward"],
        })

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  保存至: {output_path} ({len(dataset)} 条, {(time.time()-t0)/60:.1f}min)")
    counts = defaultdict(int)
    for d in dataset:
        counts[d["task_type"]] += 1
    for t, c in sorted(counts.items()):
        print(f"    {t}: {c} ({100*c/len(dataset):.1f}%)")


# ===========================================================================
# 入口
# ===========================================================================
def build_cold_start_dataset(spatial_data_path, video_base_dir,
                             output_path="cold_start_data.json",
                             ns=5000, k=3, seed=42, num_frames=16):
    asyncio.run(_build_cold_start_async(
        spatial_data_path, video_base_dir, output_path, ns, k, seed, num_frames
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial_data_path", required=True)
    parser.add_argument("--video_base_dir", required=True)
    parser.add_argument("--output_path", default="cold_start_data.json")
    parser.add_argument("--ns", type=int, default=5000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("请设置: export DASHSCOPE_API_KEY=your_key")

    build_cold_start_dataset(
        args.spatial_data_path, args.video_base_dir, args.output_path,
        args.ns, args.k, args.seed, args.num_frames,
    )
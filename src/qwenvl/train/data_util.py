# ==============================================================================
# data_utils.py — 数据格式适配模块（三个阶段共用）
# ==============================================================================
# 你的原始 jsonl 字段：
#   video            → "scannet/videos/scene0000_00.mp4"
#   conversations    → [{"from":"human","value":"<video>\n...问题..."}, {"from":"gpt","value":"答案"}]
#   problem_type     → "verbal" / "multiple_choice" / "numerical"
#
# 本模块统一提供：
#   parse_item(item)          → 从原始条目中提取 question / answer / task_type / video_path
#   load_video_frames(...)    → 从 mp4 文件直接抽帧为 PIL Image 列表
# ==============================================================================

import os
import re
import numpy as np
from typing import Optional


# ===========================================================================
# 1. 字段解析
# ===========================================================================
def parse_item(item: dict) -> dict:
    """
    从你原始 jsonl 条目中解析出统一字段。
    
    Returns:
        {
            "question":   str,   # 问题文本（去掉了 <video> 标记）
            "answer":     str,   # 真实答案
            "task_type":  str,   # verbal / multiple_choice / numerical
            "video_path": str,   # 原始 video 路径，如 scannet/videos/scene0000_00.mp4
        }
    """
    # --- question: conversations[0].value 去掉 <video>\n 前缀 ---
    raw_question = item["conversations"][0]["value"]
    # 去掉 <video> 标记及其后面的换行
    question = re.sub(r"<video>\s*", "", raw_question).strip()

    # --- answer: conversations[1].value ---
    answer = item["conversations"][1]["value"].strip()

    # --- task_type: problem_type 字段 ---
    task_type = item.get("problem_type", "verbal")

    # --- video_path ---
    video_path = item.get("video", "")

    return {
        "question": question,
        "answer": answer,
        "task_type": task_type,
        "video_path": video_path,
    }


# ===========================================================================
# 2. 从 mp4 直接抽帧
# ===========================================================================
def load_video_frames(video_path: str, video_base_dir: str, num_frames: int = 16) -> list:
    """
    从 mp4 文件直接抽帧，返回 num_frames 帧的 PIL Image 列表。
    
    Args:
        video_path:     原始路径，如 "scannet/videos/scene0000_00.mp4"
        video_base_dir: 视频根目录。最终完整路径 = video_base_dir / video_path
        num_frames:     抽取帧数
    
    Returns:
        list of PIL.Image，长度 = num_frames
    
    依赖：优先用 decord（快），decord 装不上的话 fallback 到 opencv
    """
    full_path = os.path.join(video_base_dir, video_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Video not found: {full_path}")

    try:
        return _load_with_decord(full_path, num_frames)
    except ImportError:
        return _load_with_opencv(full_path, num_frames)


def _load_with_decord(video_path: str, num_frames: int) -> list:
    """用 decord 抽帧（推荐，速度快，GPU 加速可选）"""
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # 均匀采样 num_frames 帧的索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames_np = vr.batch_get_index(indices).asnumpy()  # (N, H, W, 3)

    return [Image.fromarray(f) for f in frames_np]


def _load_with_opencv(video_path: str, num_frames: int) -> list:
    """fallback：用 OpenCV 抽帧"""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            # BGR → RGB
            frames.append(Image.fromarray(frame[:, :, ::-1].copy()))
        if len(frames) == num_frames:
            break

    cap.release()

    if len(frames) < num_frames:
        raise RuntimeError(f"Only got {len(frames)} frames from {video_path}, expected {num_frames}")
    return frames
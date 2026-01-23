# ========== 导入必要的库 ==========
import ast
import base64
import copy
import itertools
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from decord import VideoReader  # 用于高效读取视频帧
from PIL import Image
from torch.utils.data import Dataset

from . import data_list
from .rope2d import get_rope_index_2, get_rope_index_25  # 2D RoPE位置编码

# ========== 全局常量定义 ==========
IGNORE_INDEX = -100  # 损失函数中忽略的标签索引（用于padding和非预测部分）
IMAGE_TOKEN_INDEX = 151655  # 图像token在词表中的索引
VIDEO_TOKEN_INDEX = 151656  # 视频token在词表中的索引
DEFAULT_IMAGE_TOKEN = "<image>"  # 图像占位符
DEFAULT_VIDEO_TOKEN = "<video>"  # 视频占位符


def read_jsonl(path):
    """
    读取JSONL格式文件（每行一个JSON对象）
    
    Args:
        path: JSONL文件路径
        
    Returns:
        包含所有JSON对象的列表
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    """
    预处理Qwen2视觉语言模型的对话数据
    
    核心功能：
    1. 将对话格式转换为模型输入格式
    2. 插入视觉token占位符
    3. 生成input_ids和对应的labels
    4. 正确设置labels的mask（只计算assistant回复的损失）
    
    Args:
        sources: 对话数据列表，格式如：
                [
                    [
                        {"from": "human", "value": "<image>这是什么？"},
                        {"from": "gpt", "value": "这是一只猫。"}
                    ]
                ]
        tokenizer: 分词器
        grid_thw: 视觉token的网格维度列表，每个元素表示 (time, height, width) 的patch数量
        visual_type: 视觉类型，"image"或"video"
        
    Returns:
        字典，包含：
        - input_ids: 输入token序列 [batch_size, seq_len]
        - labels: 标签序列 [batch_size, seq_len]
    """
    # 角色映射：将自定义角色名映射到Qwen模板要求的角色名
    roles = {"human": "user", "gpt": "assistant"}
    
    # 系统提示词
    system_message = "You are a helpful assistant."
    
    # 检查视觉类型是否合法
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")
    
    # 深拷贝tokenizer，避免修改原始对象
    tokenizer = copy.deepcopy(tokenizer)
    
    # 设置Qwen的对话模板
    # im_start和im_end是特殊标记，用于界定消息边界
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template
    
    # 用于追踪当前处理到第几个视觉输入（一个对话可能包含多张图/视频）
    visual_replicate_index = 0
    
    input_ids, targets = [], []
    
    # 遍历每个对话样本
    for i, source in enumerate(sources):
        # 确保对话以human开始，如果不是则跳过第一轮
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)
        
        input_id, target = [], []
        
        # ========== 添加系统消息 ==========
        # 系统消息不参与损失计算，所以target全部标记为IGNORE_INDEX
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)
        
        # ========== 处理每轮对话 ==========
        for conv in source:
            # 兼容不同的对话格式（role/content 或 from/value）
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            
            # 映射角色名
            role = roles.get(role, role)
            
            # ========== 处理用户消息中的视觉token ==========
            if role == "user":
                visual_tag = f"<{visual_type}>"  # <image> 或 <video>
                
                # 如果消息中包含视觉标记
                if visual_tag in content:
                    # 将内容按视觉标记分割
                    parts = content.split(visual_tag)
                    new_parts = []
                    
                    # 为每个视觉标记生成对应的vision tokens
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        
                        # 构造视觉token序列：
                        # <|vision_start|> + N个<|image_pad|>/<|video_pad|> + <|vision_end|>
                        # N由grid_thw决定，表示这个视觉输入需要多少个token
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    
                    # 添加最后一个分割部分
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
            
            # ========== 编码当前消息 ==========
            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            
            # ========== 设置标签mask ==========
            if role in ["user", "system"]:
                # 用户和系统消息不计算损失
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                # assistant的回复需要计算损失
                target_mask = encode_id.copy()
                # 但前3个token（<|im_start|>assistant\n）不计算损失
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask
        
        # 验证input_ids和labels长度一致
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        input_ids.append(input_id)
        targets.append(target)
    
    # 转换为tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """
    惰性加载的监督学习数据集
    
    "惰性"是指只在__getitem__时才真正加载和处理数据，而不是在初始化时加载所有数据。
    这样可以：
    1. 减少内存占用（不需要同时加载所有图像/视频）
    2. 支持大规模数据集
    3. 加快训练启动速度
    
    数据集格式：
    {
        "image": "path/to/image.jpg",  # 或 "video": "path/to/video.mp4"
        "conversations": [
            {"from": "human", "value": "<image>描述这张图"},
            {"from": "gpt", "value": "这是..."}
        ]
    }
    """
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        """
        初始化数据集
        
        Args:
            tokenizer: 分词器
            data_args: 数据配置参数，包含：
                - dataset_use: 使用的数据集名称（逗号分隔）
                - image_processor: 图像处理器
                - max_pixels: 图像最大像素数
                - min_pixels: 图像最小像素数
                - video_max_total_pixels: 视频总像素数上限
                - video_min_total_pixels: 视频总像素数下限
                等
        """
        super(LazySupervisedDataset, self).__init__()
        
        # 解析要使用的数据集列表（支持多个数据集混合训练）
        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        print(f"Loading datasets: {dataset_list}")
        
        # ========== 视频处理参数配置 ==========
        # 视频总像素数限制（所有帧加起来）
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        
        # ========== 根据模型类型选择RoPE索引生成函数 ==========
        # RoPE (Rotary Position Embedding) 是一种位置编码方式
        # Qwen2.5和Spatial-MLLM使用新版本，Qwen2使用旧版本
        self.model_type = data_args.model_type
        if "qwen2.5" in self.model_type.lower() or "spatial-mllm" in self.model_type.lower():
            self.get_rope_index = get_rope_index_25
        elif "qwen2" in self.model_type.lower() and not "qwen2.5" in self.model_type.lower():
            self.get_rope_index = get_rope_index_2
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported yet.")
        
        # ========== 加载所有数据集的标注文件 ==========
        list_data_dict = []
        
        # 设置随机种子，确保数据采样的可复现性
        random.seed(42)
        
        for data in dataset_list:
            # 根据文件扩展名判断格式
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r", encoding="utf-8"))
            
            # ========== 数据采样（可选）==========
            # 如果sampling_rate < 1.0，则只使用部分数据
            # 这在快速实验或数据不平衡时很有用
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                print(f"dataset name: {data}")
            
            # 为每个标注添加数据路径（图像/视频文件的根目录）
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            
            list_data_dict += annotations
        
        print(f"Total training samples: {len(list_data_dict)}")
        
        # 随机打乱数据，增加训练的随机性
        random.shuffle(list_data_dict)
        
        print("Formatting inputs...Skip in lazy mode")
        
        # ========== 保存配置 ==========
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
        # ========== 配置图像处理器的参数 ==========
        # 这些参数控制图像如何被resize和处理
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.list_data_dict)
    
    @property
    def lengths(self):
        """
        估算每个样本的长度（用于length-based采样）
        
        粗略计算：文本token数 + 视觉token数(固定为128)
        这个属性主要用于数据加载器的采样策略
        """
        length_list = []
        for sample in self.list_data_dict:
            # 如果有图像，假设需要128个token
            img_tokens = 128 if "image" in sample else 0
            # 计算对话中所有单词数（粗略估计）
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list
    
    @property
    def modality_lengths(self):
        """
        返回样本长度，带模态标记
        
        正数：包含视觉输入的样本
        负数：纯文本样本
        
        这个信息可用于构建平衡的batch（混合视觉和文本样本）
        """
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            # 有视觉输入的样本长度为正，纯文本样本为负
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list
    
    @property
    def pre_calculated_length(self):
        """
        返回预计算的token长度（如果有）
        
        某些数据集会预先tokenize并计算token数，保存在"num_tokens"字段
        这比实时计算更准确，也更快
        """
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            # 如果没有预计算长度，返回全1（表示所有样本长度相同）
            return np.array([1] * len(self.list_data_dict))
    
    def process_image_unified(self, image_file):
        """
        统一的图像预处理函数
        
        处理流程：
        1. 加载图像
        2. 使用image_processor处理（resize、normalize等）
        3. 返回处理后的tensor和元信息
        
        Args:
            image_file: 图像文件路径
            
        Returns:
            image_tensor: 处理后的图像tensor [num_patches, pixels_per_patch]
            grid_thw: 网格维度 [t, h, w]，表示图像被分成多少个patch
            image_tchw: 原始处理后的图像 [T, C, H, W]
        """
        # 深拷贝processor，避免影响其他数据
        processor = copy.deepcopy(self.data_args.image_processor)
        
        # 加载图像并转为RGB格式
        image = Image.open(image_file).convert("RGB")
        
        # 使用processor处理图像
        visual_processed = processor.preprocess(
            image, 
            return_tensors="pt",  # 返回PyTorch tensor
            return_processed_images=True  # 返回处理后的原始图像
        )
        
        # 提取处理结果
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        
        # grid_thw表示图像的网格维度 [time, height, width]
        # 对于单张图像，time=1
        grid_thw = visual_processed["image_grid_thw"][0]
        
        # 保存处理后的原始图像（可能用于某些特殊任务）
        image_tchw = visual_processed["processed_images"]
        
        return image_tensor, grid_thw, image_tchw
    
    def process_video(self, video_file):
        """
        视频预处理函数
        
        处理流程：
        1. 读取视频文件
        2. 根据视频长度和配置，智能采样帧
        3. 对采样的帧进行预处理
        
        采样策略：
        - 根据视频长度和interval计算目标帧数
        - 限制在[video_min_frames, video_max_frames]范围内
        - 均匀采样以覆盖整个视频
        
        Args:
            video_file: 视频文件路径
            
        Returns:
            video_tensor: 处理后的视频tensor [num_patches, pixels_per_patch]
            grid_thw: 网格维度 [t, h, w]
            second_per_grid_ts: 每个时间网格对应的秒数
            video_tchw: 原始处理后的视频帧 [T, C, H, W]
        """
        # 检查文件是否存在
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        
        # ========== 使用decord高效读取视频 ==========
        # decord比opencv和ffmpeg快很多，特别适合深度学习
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)  # 视频总帧数
        avg_fps = vr.get_avg_fps()  # 平均帧率
        video_length = total_frames / avg_fps  # 视频时长（秒）
        
        # ========== 计算需要采样多少帧 ==========
        # 基础采样间隔（秒）
        interval = getattr(self.data_args, "base_interval", 4)
        # 根据视频长度和间隔计算目标帧数
        num_frames_to_sample = round(video_length / interval)
        
        # 帧数限制
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        
        # 将目标帧数限制在合理范围内
        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        
        # ========== 均匀采样帧索引 ==========
        # 在整个视频长度上均匀采样target_frames个点
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        # 去除重复的索引（可能在短视频中出现）
        frame_idx = np.unique(frame_idx)
        
        # 批量读取选定的帧
        video = vr.get_batch(frame_idx).asnumpy()
        
        # ========== 计算有效帧率 ==========
        # 如果没有指定，则根据采样帧数和视频长度计算
        fps = getattr(self.data_args, "video_frame_fps", None)
        if fps is None:
            fps = len(frame_idx) / video_length
        
        # ========== 配置视频专用的图像处理器 ==========
        processor = copy.deepcopy(self.data_args.image_processor)
        # 视频帧可能需要不同的分辨率设置
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        
        # ========== 处理视频帧 ==========
        video_processed = processor.preprocess(
            images=None,  # 不处理单张图像
            videos=video,  # 处理视频帧序列
            return_tensors="pt",
        )
        
        # 提取处理结果
        # video_tensor的形状：[num_patches, pixels_per_patch]
        video_tensor = video_processed["pixel_values_videos"]
        
        # grid_thw: [time_patches, height_patches, width_patches]
        grid_thw = video_processed["video_grid_thw"][0]
        
        # 原始处理后的视频帧（如果需要）
        video_tchw = video_processed.get("processed_videos", None)
        
        # ========== 计算时间网格对应的真实时间 ==========
        # 每个时间patch对应多少秒
        # temporal_patch_size是时间维度上每个patch包含多少帧
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        
        return video_tensor, grid_thw, second_per_grid_ts, video_tchw
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        获取第i个样本
        
        实现了健壮的错误处理机制：
        1. 首先尝试加载当前样本（最多3次）
        2. 如果失败，尝试加载下一个样本（最多3次）
        3. 最后再尝试加载当前样本一次
        4. 如果仍然失败，抛出异常
        
        这种机制可以处理：
        - 网络存储的临时故障
        - 损坏的文件
        - 其他随机错误
        
        Args:
            i: 样本索引
            
        Returns:
            包含模型输入的字典
        """
        num_base_retries = 3  # 基础重试次数
        num_final_retries = 30  # 最终重试次数（未使用）
        
        # ========== 第一阶段：尝试加载当前样本 ==========
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # 如果失败，等待1秒（可能是网络问题）
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)
        
        # ========== 第二阶段：尝试加载其他样本 ==========
        # 如果当前样本反复失败，可能文件损坏，尝试下一个
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # 不需要等待，直接重试
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass
        
        # ========== 第三阶段：最后一次尝试 ==========
        # 如果还是失败，就让它抛出异常
        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        """
        实际加载和处理第i个样本的函数
        
        处理流程：
        1. 获取标注信息
        2. 根据模态类型（图像/视频/纯文本）进行相应处理
        3. 调用预处理函数tokenize对话
        4. 生成位置编码
        5. 组装返回字典
        
        Args:
            i: 样本索引
            
        Returns:
            包含以下键的字典：
            - input_ids: token序列
            - labels: 标签序列
            - position_ids: 位置编码
            - pixel_values: 图像像素值（如果有图像）
            - image_grid_thw: 图像网格维度（如果有图像）
            - pixel_values_videos: 视频像素值（如果有视频）
            - video_grid_thw: 视频网格维度（如果有视频）
        """
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        video = None
        
        # ========== 情况1：处理图像数据 ==========
        if "image" in sources[0]:
            # 获取图像文件路径
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            
            # 支持多图像输入
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    # 处理多张图像
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    # 并行处理所有图像
                    results = [self.process_image_unified(file) for file in image_file]
                    # 解包结果
                    image, grid_thw, image_tchw = zip(*results)
                    # 将所有图像拼接在一起
                    image_tchw = torch.cat(image_tchw, dim=0)
                else:
                    # 只有一张图像（但被包装在列表中）
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw, image_tchw = self.process_image_unified(image_file)
                    image = [image]
            else:
                # 单张图像
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw, image_tchw = self.process_image_unified(image_file)
                image = [image]
            
            # ========== 计算合并后的网格维度 ==========
            # grid_thw_merged用于确定需要插入多少个视觉token
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            
            # 计算每个图像合并后的token数
            # merge_size是合并因子，通常为2（即2x2的patch合并为1个token）
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            
            # ========== Tokenize对话 ==========
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )
            
            # ========== 生成位置编码 ==========
            # 2D RoPE需要知道图像的网格结构
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
        
        # ========== 情况2：处理视频数据 ==========
        elif "video" in sources[0]:
            # 获取视频文件路径
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            
            # 支持多视频输入
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    # 处理多个视频
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts, video_tchw = zip(*results)
                else:
                    # 单个视频（但在列表中）
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts, video_tchw = self.process_video(video_file)
                    video = [video]
            else:
                # 单个视频
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts, video_tchw = self.process_video(video_file)
                video = [video]
            
            # 计算合并后的网格维度
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            
            # Tokenize对话
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )
            
            # 生成位置编码（视频需要额外的时间信息）
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        
        # ========== 情况3：纯文本数据 ==========
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            
            # 纯文本使用简单的1D位置编码
            # 扩展为3维以保持与2D RoPE的兼容性
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)  # [3, 1, seq_len]
            )
        
        # ========== 解包batch维度（因为当前是单个样本）==========
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )
        
        # ========== 添加视觉数据到返回字典 ==========
        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            data_dict["image_tchw"] = image_tchw
        
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw
            data_dict["video_tchw"] = video_tchw
        
        return data_dict


def pad_and_cat(tensor_list):
    """
    将不同长度的tensor填充到相同长度并拼接
    
    用于处理批次中position_ids长度不一致的情况。
    
    处理流程：
    1. 找到最长的序列长度
    2. 将所有tensor右侧填充到该长度（填充值为1）
    3. 在batch维度上拼接
    
    Args:
        tensor_list: tensor列表，每个形状为 [3, 1, seq_len_i]
        
    Returns:
        拼接后的tensor，形状为 [3, batch_size, max_seq_len]
    """
    # 找到最长序列
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    
    padded_tensors = []
    for tensor in tensor_list:
        # 计算需要填充的长度
        pad_length = max_length - tensor.shape[2]
        # 在最后一个维度右侧填充（填充值为1）
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    
    # 在第二个维度（batch维度）上拼接
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    
    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    监督学习数据集的批处理整理器（Collator）
    
    功能：
    1. 将多个样本组合成一个batch
    2. 对不同长度的序列进行padding
    3. 创建attention_mask
    4. 整理多模态数据（图像/视频）
    
    DataCollator在PyTorch的DataLoader中使用，负责将__getitem__
    返回的单个样本整理成batch。
    """
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本整理成一个batch
        
        Args:
            instances: 样本列表，每个样本是__getitem__返回的字典
            
        Returns:
            batch字典，包含：
            - input_ids: [batch_size, max_seq_len]
            - labels: [batch_size, max_seq_len]
            - attention_mask: [batch_size, max_seq_len]
            - position_ids: [3, batch_size, max_seq_len]
            - pixel_values: [total_images, num_patches, pixels_per_patch]
            - image_grid_thw: [total_images, 3]
            - pixel_values_videos: [total_videos, num_patches, pixels_per_patch]
            - video_grid_thw: [total_videos, 3]
        """
        # ========== 提取所有样本的input_ids, labels, position_ids ==========
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        
        # ========== Padding input_ids ==========
        # 使用pad_sequence自动填充到最长序列
        # batch_first=True表示输出形状为[batch_size, seq_len]
        # padding_value使用tokenizer的pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # ========== Padding labels ==========
        # labels的padding使用IGNORE_INDEX，这样这些位置不会计算损失
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # ========== Padding position_ids ==========
        # position_ids的形状特殊，需要自定义padding函数
        position_ids = pad_and_cat(position_ids)
        
        # ========== 截断到最大长度 ==========
        # 如果序列超过模型最大长度，进行截断
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        
        # ========== 创建attention_mask ==========
        # attention_mask标记哪些位置是真实token（1），哪些是padding（0）
        # ne()是not equal的意思，返回input_ids != pad_token_id的mask
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # ========== 收集批次中的所有图像 ==========
        # 使用itertools.chain展平嵌套列表
        # 因为每个instance可能包含多张图像
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        
        # ========== 收集批次中的所有视频 ==========
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        
        # ========== 处理图像数据 ==========
        if len(images) != 0:
            # 将所有图像拼接成一个大tensor
            # 形状: [total_images, num_patches, pixels_per_patch]
            concat_images = torch.cat([image for image in images], dim=0)
            
            # 收集所有图像的网格维度信息
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )  # 列表，每个元素是[3]的tensor
            
            # 堆叠成一个tensor: [total_images, 3]
            grid_thw = torch.stack(grid_thw, dim=0)
            
            # 收集原始图像数据（可能用于某些特殊任务）
            image_tchw = [
                instance["image_tchw"] 
                for instance in instances 
                if "image_tchw" in instance
            ]  # 列表，每个元素是[T, C, H, W]
        else:
            # 如果batch中没有图像
            concat_images = None
            grid_thw = None
            image_tchw = None
        
        # ========== 处理视频数据（类似图像）==========
        if len(videos) != 0:
            # 拼接所有视频
            concat_videos = torch.cat([video for video in videos], dim=0)
            
            # 收集视频网格信息
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
            
            # 收集原始视频数据
            video_tchw = [
                instance["video_tchw"] 
                for instance in instances 
                if "video_tchw" in instance
            ]
        else:
            concat_videos = None
            video_grid_thw = None
            video_tchw = None
        
        # ========== 将多模态数据添加到batch ==========
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        # 添加原始图像/视频数据（如果有）
        if image_tchw is not None:
            batch["image_tchw"] = image_tchw
        if video_tchw is not None:
            batch["video_tchw"] = video_tchw
        
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """
    创建监督学习所需的数据模块
    
    这是一个工厂函数，用于创建训练所需的所有数据组件。
    
    Args:
        tokenizer: 分词器
        data_args: 数据配置参数
        
    Returns:
        包含以下键的字典：
        - train_dataset: 训练数据集（LazySupervisedDataset实例）
        - eval_dataset: 验证数据集（这里为None，因为没有实现验证）
        - data_collator: 数据整理器（DataCollatorForSupervisedDataset实例）
        
    这个字典会被直接传递给Trainer的构造函数。
    """
    # 创建训练数据集
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    
    # 创建数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,  # 没有实现验证集
        data_collator=data_collator
    )
# 版权声明：代码改编自FastChat项目，原始版权归Stanford Alpaca项目
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import os
import pathlib
import shutil
import sys
from pathlib import Path

# 将项目根目录添加到Python搜索路径，确保能正确导入项目模块
# add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
import transformers
from transformers import (
    AutoProcessor,          # 自动处理器，用于加载tokenizer和image processor
    AutoTokenizer,          # 自动分词器
    Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL模型
    Qwen2VLForConditionalGeneration,     # Qwen2-VL模型
    Qwen2VLImageProcessor,  # Qwen2-VL的图像处理器
    Trainer,                # HuggingFace的训练器
)

# 导入自定义模块
import src.qwenvl.train.trainer
from src.qwenvl.data.data_qwen import make_supervised_data_module  # 数据加载模块
from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration  # 空间增强型多模态模型
from src.qwenvl.preprocessor.image_processing_qwen2_vl import Qwen2VLImageProcessorModified  # 修改版图像处理器
from src.qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments  # 参数类
from src.qwenvl.train.trainer import replace_qwen2_vl_attention_class  # 注意力机制替换函数


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    安全地保存模型权重到磁盘
    
    功能说明：
    - 兼容DeepSpeed分布式训练和常规训练
    - DeepSpeed模式下直接调用trainer.save_model
    - 非DeepSpeed模式下将state_dict转到CPU再保存，避免显存溢出
    
    Args:
        trainer: HuggingFace的Trainer对象
        output_dir: 模型保存目录路径
    """
    # 如果使用DeepSpeed分布式训练
    if trainer.deepspeed:
        # 同步所有GPU，确保所有进程都完成当前操作
        torch.cuda.synchronize()
        # 直接使用trainer的保存方法（DeepSpeed有自己的保存逻辑）
        trainer.save_model(output_dir)
        return

    # 非DeepSpeed模式：手动保存state_dict
    # 获取模型的完整状态字典（包含所有参数）
    state_dict = trainer.model.state_dict()
    
    # 只在主进程（rank 0）保存模型
    if trainer.args.should_save:
        # 将所有tensor从GPU移到CPU，释放GPU显存
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        # 删除GPU上的state_dict，进一步释放显存
        del state_dict
        # 调用trainer的内部保存方法
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """
    根据训练配置设置模型各部分的可训练性（冻结或解冻参数）
    
    这个函数实现了细粒度的参数控制，可以选择性地训练模型的不同组件：
    1. 视觉编码器（visual encoder）：提取图像特征的ViT等backbone
    2. 视觉-语言连接器（mm_connector）：将视觉特征投影到语言空间
    3. 语言模型（LLM）：主要的文本生成模型
    4. 空间编码器（spatial_encoder）：额外的空间理解模块（仅Spatial-MLLM）
    
    Args:
        model_args: 模型配置参数，包含各个组件的训练开关
        model: 加载的模型实例
    """
    
    # ========== 1. 视觉编码器（Vision Encoder）配置 ==========
    # 决定是否训练视觉backbone（如ViT）
    if model_args.tune_mm_vision:
        # 解冻视觉编码器的所有参数，允许训练
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        # 冻结视觉编码器，不更新参数（通常预训练的视觉模型效果已经很好）
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # ========== 2. 多模态连接器（MM Connector）配置 ==========
    # 连接器负责将视觉特征对齐到语言模型的表示空间
    if model_args.tune_mm_connector:
        # 解冻连接器参数（通常这部分需要训练以适配新任务）
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        # 冻结连接器
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # ========== 3. 语言模型（LLM）配置 ==========
    # 控制主要的文本生成transformer和输出层
    if model_args.tune_mm_llm:
        # 解冻LLM的所有transformer层
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        # 解冻语言模型头（lm_head），用于生成词汇表上的概率分布
        model.lm_head.requires_grad = True
    else:
        # 冻结整个LLM（适用于只训练视觉部分的场景）
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # ========== 4. 空间编码器配置（仅Spatial-MLLM模型）==========
    # spatial_encoder是额外的空间理解模块，用于增强空间推理能力
    if hasattr(model, "spatial_encoder"):
        if model_args.tune_mm_spatial_encoder:
            # 解冻空间编码器
            for n, p in model.spatial_encoder.named_parameters():
                p.requires_grad = True
        else:
            # 冻结空间编码器
            for n, p in model.spatial_encoder.named_parameters():
                p.requires_grad = False

    # ========== 5. 额外的连接器配置（如果存在）==========
    # 某些模型架构可能有额外的connector模块
    if hasattr(model, "connector"):
        if model_args.tune_mm_connector:
            # 解冻额外的连接器
            for n, p in model.connector.named_parameters():
                p.requires_grad = True
        else:
            # 冻结额外的连接器
            for n, p in model.connector.named_parameters():
                p.requires_grad = False


def get_model(model_args, data_args, training_args, attn_implementation="flash_attention_2"):
    """
    根据配置加载相应的视觉语言模型
    
    支持三种模型架构：
    1. Spatial-MLLM：增强型空间理解模型，基于Qwen2-VL + 空间编码器
    2. Qwen2.5-VL：最新版本的Qwen视觉语言模型
    3. Qwen2-VL：标准版Qwen视觉语言模型
    
    Args:
        model_args: 模型配置参数
        data_args: 数据配置参数
        training_args: 训练配置参数
        attn_implementation: 注意力机制实现方式，默认使用flash_attention_2（更快更省显存）
    
    Returns:
        model: 加载的模型实例
        image_processor: 对应的图像处理器
    """
    
    # ========== 模型类型1：Spatial-MLLM（空间增强型多模态模型）==========
    if "spatial-mllm" in model_args.model_type.lower():
        # 创建Spatial-MLLM的配置对象
        spatial_mllm_config = SpatialMLLMConfig.from_pretrained(
            model_args.pretrained_model_name_or_path,  # 基础模型路径
            # 空间编码器配置
            spatial_config={
                "img_size": 518,        # 输入图像尺寸（518x518）
                "patch_size": 14,       # 图像patch大小（每个patch 14x14像素）
                "embed_dim": 1024,      # 空间特征的嵌入维度
            },
            # 连接器配置（用于将空间特征注入到LLM）
            connector_config={
                "connector_type": model_args.connector_type,  # 连接器类型（如MLP、Resampler等）
                "spatial_embeds_layer_idx": model_args.spatial_embeds_layer_idx,  # 在LLM的哪一层注入空间特征
            },
        )
        
        # 从预训练权重加载Spatial-MLLM模型
        model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            config=spatial_mllm_config,
            attn_implementation=attn_implementation,  # 使用Flash Attention 2
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),  # 混合精度训练
        )
        
        # 如果不是从头训练（"ct"代表from scratch），则加载预训练的空间编码器权重
        if "ct" not in model_args.model_type.lower():
            # 从指定路径加载Vision-GGT的预训练权重
            model.spatial_encoder.load_pretrained_weights(model_args.vggt_checkpoints_path)
            # 确保spatial_encoder与主模型在同一设备和数据类型上
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            model.spatial_encoder.to(device=device, dtype=dtype)

        # 使用修改版的图像处理器（可能包含特殊的预处理逻辑）
        image_processor = Qwen2VLImageProcessorModified.from_pretrained(
            model_args.pretrained_model_name_or_path,
        )
    
    # ========== 模型类型2：Qwen2.5-VL ==========
    elif "qwen2.5" in model_args.model_type.lower():
        # 加载Qwen2.5-VL模型（最新版本）
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,  # 模型缓存目录
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        
        # 使用AutoProcessor自动加载对应的图像处理器
        image_processor = AutoProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path,
        ).image_processor
    
    # ========== 模型类型3：Qwen2-VL（标准版）==========
    else:
        # 加载标准的Qwen2-VL模型
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        
        # 使用标准的Qwen2VL图像处理器
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path,
        )
    
    return model, image_processor


def train(attn_implementation="flash_attention_2"):
    """
    主训练函数
    
    完整的训练流程包括：
    1. 解析命令行参数
    2. 加载模型和处理器
    3. 配置训练策略（梯度检查点、参数冻结等）
    4. 加载数据集
    5. 开始训练
    6. 保存模型
    
    Args:
        attn_implementation: 注意力机制实现，默认使用Flash Attention 2
    """
    global local_rank  # 全局变量，记录当前进程的GPU编号

    # ========== 步骤1：解析命令行参数 ==========
    # HfArgumentParser可以从命令行、json文件或yaml文件解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 记录当前进程的GPU编号（分布式训练时使用）
    local_rank = training_args.local_rank
    
    # 创建输出目录（如果不存在）
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ========== 步骤2：加载模型和图像处理器 ==========
    model, image_processor = get_model(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        attn_implementation=attn_implementation,
    )
    
    # 将图像处理器和模型类型传递给数据参数（数据加载时需要用到）
    data_args.image_processor = image_processor
    data_args.model_type = model_args.model_type

    # ========== 步骤3：关闭KV缓存 ==========
    # 训练时不需要缓存past_key_values，关闭可以节省显存
    model.config.use_cache = False

    # ========== 步骤4：配置梯度检查点（Gradient Checkpointing）==========
    # 梯度检查点通过重计算来节省显存，适合训练大模型
    if training_args.gradient_checkpointing:
        # 优先使用模型自带的enable_input_require_grads方法
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 如果没有该方法，手动注册hook确保输入需要梯度
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            # 在embedding层注册forward hook
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ========== 步骤5：加载Tokenizer ==========
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,  # 最大序列长度
        padding_side="right",  # 在右侧填充（生成任务通常使用右填充）
        use_fast=False,        # 不使用Rust实现的fast tokenizer（某些情况下更稳定）
    )
    
    # ========== 步骤6：设置模型参数的可训练性 ==========
    set_model(model_args, model)

    # ========== 步骤7：打印模型各部分的参数统计信息 ==========
    # 这有助于了解模型规模和训练成本
    
    # 打印视觉编码器参数
    model.visual.print_trainable_parameters()
    # 打印语言模型参数
    model.model.print_trainable_parameters()
    
    # 如果有空间编码器，打印其参数
    if hasattr(model, "spatial_encoder"):
        model.spatial_encoder.print_trainable_parameters()
    
    # 如果有连接器，打印其参数
    if hasattr(model, "connector"):
        model.connector.print_trainable_parameters()
    
    # ========== 步骤8：详细的参数统计分析 ==========
    def count_parameters_detailed(model):
        """
        详细统计模型参数信息
        
        统计内容：
        - 每个模块的总参数量和可训练参数量
        - 全局总参数量和可训练参数量
        - 可训练参数比例
        - 估算训练所需显存（使用BF16 + Adam优化器）
        
        Returns:
            total_params: 总参数量
            trainable_params: 可训练参数量
            param_details: 各模块详细统计字典
        """
        total_params = 0        # 总参数量
        trainable_params = 0    # 可训练参数量
        param_details = {}      # 按模块统计的详细信息
        
        # 遍历模型的所有参数
        for name, param in model.named_parameters():
            num_params = param.numel()  # 获取参数数量（元素个数）
            total_params += num_params
            
            # 如果参数需要梯度，则计入可训练参数
            if param.requires_grad:
                trainable_params += num_params
            
            # 按模块分组统计（取参数名的第一部分作为模块名）
            module_name = name.split('.')[0]
            if module_name not in param_details:
                param_details[module_name] = {'total': 0, 'trainable': 0}
            
            param_details[module_name]['total'] += num_params
            if param.requires_grad:
                param_details[module_name]['trainable'] += num_params
        
        return total_params, trainable_params, param_details

    # 执行参数统计
    total, trainable, details = count_parameters_detailed(model)
    
    # ========== 步骤9：打印美观的参数统计报告 ==========
    print(f"\n{'='*50}")
    print(f"Parameter Statistics:")
    print(f"{'='*50}")
    
    # 打印每个模块的参数统计
    for module, counts in details.items():
        print(f"  {module}: {counts['total']/1e6:.2f}M total, {counts['trainable']/1e6:.2f}M trainable")
    
    print(f"{'='*50}")
    # 打印全局统计
    print(f"Total parameters: {total:,} ({total/1e9:.2f}B)")  # 总参数量（十亿为单位）
    print(f"Trainable parameters: {trainable:,} ({trainable/1e9:.2f}B)")  # 可训练参数量
    print(f"Trainable ratio: {100 * trainable / max(total, 1):.2f}%")  # 可训练参数比例
    
    # 估算训练所需显存
    # BF16每个参数2字节，Adam优化器需要额外的一阶和二阶动量（每个参数4字节）
    # 总计：2 + 4 + 4 + 2（梯度） = 12字节/参数
    print(f"Estimated training memory (BF16 + Adam): {trainable * 12 / 1024**3:.1f} GB")
    print(f"{'='*50}\n")

    # ========== 步骤10：创建数据模块 ==========
    # make_supervised_data_module返回包含train_dataset和data_collator的字典
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # ========== 步骤11：创建Trainer并开始训练 ==========
    trainer = Trainer(
        model=model,                    # 要训练的模型
        processing_class=tokenizer,     # tokenizer（HF新版本用processing_class替代tokenizer参数）
        args=training_args,             # 训练配置（学习率、batch size、保存策略等）
        **data_module                   # 解包数据模块（包含train_dataset和data_collator）
    )

    # 开始训练！
    trainer.train()

    # ========== 步骤12：保存训练状态 ==========
    # 保存optimizer、scheduler等状态，用于恢复训练
    trainer.save_state()

    # ========== 步骤13：复制chat_template.json到输出目录 ==========
    # chat_template.json定义了对话格式，推理时需要用到
    source_path = os.path.join(model_args.pretrained_model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    # ========== 步骤14：恢复KV缓存配置 ==========
    # 推理时需要启用KV缓存来加速生成
    model.config.use_cache = True

    # ========== 步骤15：保存最终模型 ==========
    # 使用safe_save_model_for_hf_trainer确保兼容DeepSpeed和普通训练
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


# ========== 程序入口 ==========
if __name__ == "__main__":
    # 启动训练，使用Flash Attention 2加速注意力计算
    train(attn_implementation="flash_attention_2")
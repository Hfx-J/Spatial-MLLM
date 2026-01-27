# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设这两个是从同级目录导入的辅助模块
# activate_head: 用于最后的输出激活（如深度估计常用的 Sigmoid 或 Log 变换）
# create_uv_grid, position_grid_to_embed: 用于生成位置编码
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT Head (Dense Prediction Transformer Head) 用于密集预测任务。

    实现遵循 "Vision Transformers for Dense Prediction" (https://arxiv.org/abs/2103.13413) 的架构。
    DPT Head 处理来自 Vision Transformer 主干的特征，并通过融合多尺度特征生成密集预测。

    参数:
        dim_in (int): 输入维度（通常是 Transformer 的 embed_dim）。
        patch_size (int, optional): Patch 大小，默认 14。
        output_dim (int, optional): 输出通道数（例如深度估计通常为 1）。默认 4。
        activation (str, optional): 激活函数类型。默认 "inv_log"。
        conf_activation (str, optional): 置信度激活类型。默认 "expp1"。
        features (int, optional): 中间表示的特征通道数。默认 256。
        out_channels (List[int], optional): 指定每个中间层映射到的输出通道数。
        intermediate_layer_idx (List[int], optional): 指定从主干网络的哪几层提取特征。
        pos_embed (bool, optional): 是否使用位置编码。默认 True。
        feature_only (bool, optional): 如果为 True，只返回特征图，不经过最后的预测头。默认 False。
        down_ratio (int, optional): 输出分辨率的下采样倍率。默认 1。
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # 1. 投影层 (Reassemble 阶段的一部分)
        # 将来自 Transformer 不同层的 Token 特征投影到指定的通道数 (out_channels)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # 2. 调整层 (Reassemble 阶段的一部分)
        # 将不同层级的特征图缩放，使它们的空间分辨率成倍数关系（通常是对齐到 1/4, 1/8, 1/16, 1/32）
        # 这里使用了转置卷积（上采样）和普通卷积（下采样或保持）
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d( # 第一层特征通常分辨率较低，这里上采样 4 倍
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d( # 第二层特征上采样 2 倍
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),      # 第三层保持原样
                nn.Conv2d(          # 第四层下采样 2 倍
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        # 3. 构建 Fusion 阶段的预处理层 (Scratch)
        # _make_scratch 会创建一组卷积层，将所有特征图的通道数统一调整为 self.features (例如 256)
        self.scratch = _make_scratch(out_channels, features, expand=False)

        # 4. 构建 Fusion 模块 (RefineNet)
        # 这里构建了一个类似 U-Net 解码器的结构，从深层（低分辨率）向浅层（高分辨率）逐级融合
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False) # 最深层通常不需要残差连接

        head_features_1 = features
        head_features_2 = 32

        # 5. 输出头 (Output Head)
        if feature_only:
            # 如果只需要特征，不进行通道压缩
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            # 标准模式：先压缩通道，再通过一系列卷积输出最终预测
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播，支持对视频帧进行分块处理以节省显存。

        Args:
            aggregated_tokens_list: 来自 Transformer 不同层的 Token 列表。
            images: 输入图像张量 [B, S, 3, H, W] (B: Batch, S: Sequence/Frames)。
            patch_start_idx: Patch token 在序列中的起始索引（跳过 CLS token 或 Register tokens）。
            frames_chunk_size: 每次处理的帧数。如果显存不足，调小此值。
        """
        B, S, _, H, W = images.shape

        # 如果没有指定 chunk 大小或 chunk 大于总帧数，一次性处理所有帧
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # 否则，分块处理帧
        assert frames_chunk_size > 0

        all_preds = []
        all_conf = []

        # 循环处理每个 Chunk
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # 将所有 Chunk 的结果在时间/序列维度 (dim=1) 拼接
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        DPT Head 的核心前向传播实现。
        """
        # 如果指定了帧索引，切片获取当前 Chunk 的图像
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        # 计算 Patch 的数量 (Grid Size)
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        # --- 阶段 1: Reassemble (重组) ---
        # 遍历指定的中间层索引 (intermediate_layer_idx)
        for layer_idx in self.intermediate_layer_idx:
            # 1. 提取 Token，去掉 CLS token 等前缀
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # 如果分块处理，只取当前 Chunk 对应的 Token
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            # 2. Reshape: [B, S, N_patches, Dim] -> [B*S, N_patches, Dim]
            x = x.reshape(B * S, -1, x.shape[-1])

            # 3. LayerNorm
            x = self.norm(x)

            # 4. 恢复空间结构: [B*S, H*W, C] -> [B*S, C, H, W]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # 5. 投影与尺寸调整
            x = self.projects[dpt_idx](x)  # 通道变换
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H) # 加上位置编码
            x = self.resize_layers[dpt_idx](x) # 分辨率调整

            out.append(x)
            dpt_idx += 1

        # --- 阶段 2: Fusion (融合) ---
        # 使用 RefineNet 融合多尺度特征
        out = self.scratch_forward(out)
        
        # --- 阶段 3: Upsample (上采样) ---
        # 双线性插值到目标分辨率
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        # --- 阶段 4: Head Prediction (预测) ---
        out = self.scratch.output_conv2(out) # 最后的卷积层
        
        # 使用激活函数生成预测值和置信度 (activate_head 是外部函数)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # 恢复 Batch 和 Sequence 维度
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        为特征图添加位置编码，这对密集预测任务（如深度）保持空间一致性很重要。
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        # 创建 UV 网格
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        # 将网格转换为嵌入向量
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        # 缩放并添加到特征 x 上
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        执行特征融合的前向传播。
        
        Args:
            features: 列表，包含从深层到浅层（或不同尺度）处理后的特征图。
        """
        layer_1, layer_2, layer_3, layer_4 = features

        # 先通过 RN (Residual Network) 层统一通道数
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 从最深层 layer_4 开始，逐级向上融合
        # refinenet4: 处理最底层特征
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4 # 节省显存

        # refinenet3: 融合上一级输出与 layer_3
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        # refinenet2: 融合上一级输出与 layer_2
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        # refinenet1: 融合上一级输出与 layer_1 (最高分辨率特征)
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        # 输出前的最后卷积
        out = self.scratch.output_conv1(out)
        return out


################################################################################
# 辅助模块 (Modules)
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    """
    工厂函数，用于创建 FeatureFusionBlock (RefineNet)。
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    """
    工厂函数，创建 Scratch 模块。
    Scratch 模块包含一组卷积层 (layer1_rn, layer2_rn...)，
    用于将输入特征的通道数映射到统一的 out_shape。
    """
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    # 定义各个层级的卷积层
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """
    残差卷积单元。
    结构: Input -> [Act -> Conv -> Norm -> Act -> Conv -> Norm] + Input -> Output
    """

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None
        # 这里虽然预留了 norm 的逻辑，但初始化为 None，实际没有使用 BN
        
        self.activation = activation
        # FloatFunctional 用于支持量化操作的加法
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    特征融合块 (RefineNet Block)。
    用于融合两个特征图：一个来自上一级 RefineNet 的输出（低分辨率），一个来自主干网络的同级特征（高分辨率）。
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        # resConfUnit1 处理来自 Backbone 的特征
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        # resConfUnit2 处理融合后的特征
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """
        Args:
            xs[0]: 上一级 RefineNet 的输出 (Coarse feature)。
            xs[1]: (可选) 当前层级 Backbone 的特征 (Fine feature)。
            size: 目标上采样尺寸。
        """
        output = xs[0]

        # 如果有残差（即有 xs[1]），将两者相加融合
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        # 再次经过残差卷积处理
        output = self.resConfUnit2(output)

        # 确定上采样参数
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        # 上采样并输出
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    自定义插值函数。
    
    原因：PyTorch 的 nn.functional.interpolate 在处理非常大的张量时（元素总数超过 INT_MAX，即 2^31-1），
    可能会因为底层 C++ 实现的索引溢出而崩溃。
    此函数通过将 Batch 维度切片（chunk）处理来规避此问题。
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736 # 安全阈值，略小于 2^31-1

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        # 如果张量过大，按 Batch 维度切分
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        # 正常情况直接调用官方实现
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
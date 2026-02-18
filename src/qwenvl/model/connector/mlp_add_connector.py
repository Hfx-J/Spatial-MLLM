from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from src.qwenvl.external.vggt.heads.dpt_head import _make_fusion_block

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

# 引用你提供的 DPT 组件
# 假设 ResidualConvUnit, FeatureFusionBlock, custom_interpolate 已定义
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

# 假设 FeatureFusionBlock, _make_fusion_block, custom_interpolate 已经从之前提供的 dpt_head.py 导入
# 如果在同一个文件，请确保它们在类定义之前
class DPTConnector(nn.Module):
    def __init__(
        self,
        vggt_dim: int = 2048,
        language_dim: int = 2048,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 2048],
        layer_indices: List[int] = [7, 11, 14, 23],
        visual_temporal_merge_size: int = 2,  # 与 QwenVL 一致
        visual_spatial_merge_size: int = 2,   # 与 QwenVL 一致
        dpt_nframes: int = 16,                # DPT/VGGT 实际处理的帧数
    ) -> None:
        super().__init__()
        self.layer_indices = layer_indices
        self.visual_temporal_merge_size = visual_temporal_merge_size
        self.visual_spatial_merge_size = visual_spatial_merge_size
        self.dpt_nframes = dpt_nframes

        # merge 后扩展的通道数：features * tm * sm^2
        self.merged_dim = (
            features
            * visual_temporal_merge_size
            * visual_spatial_merge_size ** 2
        )

        # ========== 输入归一化 ==========
        self.input_norms = nn.ModuleList([
            nn.LayerNorm(vggt_dim) for _ in out_channels
        ])

        # 1. 投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(vggt_dim, oc, kernel_size=1) for oc in out_channels
        ])

        # 2. 映射到 DPT 内部通道数
        self.rn_layers = nn.ModuleList([
            nn.Conv2d(oc, features, kernel_size=3, padding=1, bias=False) for oc in out_channels
        ])

        # 3. 级联融合模块
        self.refinenet1 = _make_fusion_block(features)
        self.refinenet2 = _make_fusion_block(features)
        self.refinenet3 = _make_fusion_block(features)
        self.refinenet4 = _make_fusion_block(features, has_residual=False)

        # 4. QwenVL 风格 merge 后的投影
        self.ln_q = Qwen2RMSNorm(self.merged_dim, eps=1e-6)
        self.output_proj = nn.Sequential(
            nn.Linear(self.merged_dim, self.merged_dim),
            nn.GELU(),
            nn.Linear(self.merged_dim, language_dim),
        )

        # ========== 输出归一化 ==========
        self.dpt_out_norm = nn.LayerNorm(language_dim)

        # 5. 门控融合
        self.fusion_proj = nn.Linear(language_dim * 2, language_dim)
        self.final_norm = Qwen2RMSNorm(language_dim, eps=1e-6)
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
        for proj in self.projects:
            nn.init.normal_(proj.weight, std=0.01)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        for rn in self.rn_layers:
            nn.init.normal_(rn.weight, std=0.01)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _frame_subsample(self, x: torch.Tensor, t_target: int) -> torch.Tensor:
        """
        对 DPT 输出的帧维度做跳帧采样，使其与 QwenVL 处理的帧数对齐。

        Args:
            x:        [T_dpt, C, H, W]，DPT refinenet 输出
            t_target: QwenVL 对应的帧数（来自 grid_thw 的 t_orig）

        Returns:
            x_sub:    [t_target, C, H, W]
        """
        T_dpt = x.shape[0]
        if T_dpt == t_target:
            return x  # 帧数已对齐，不需要跳帧

        assert T_dpt % t_target == 0, (
            f"DPT 帧数 {T_dpt} 不能被 QwenVL 帧数 {t_target} 整除，"
            f"无法做均匀跳帧。"
        )
        stride = T_dpt // t_target
        # 均匀采样：取第 0, stride, 2*stride, ... 帧
        indices = torch.arange(0, T_dpt, stride, device=x.device)
        return x[indices]   # [t_target, C, H, W]

    def _qwenvl_merge(self, x: torch.Tensor) -> torch.Tensor:
        """
        QwenVL 风格时空 merge（与 MLPAddConnector.preprocess_spatial_embeds 逻辑一致）。

        Args:
            x: [T, C, H, W]，跳帧后与 QwenVL 对齐的帧序列

        Returns:
            merged: [T//tm * H//sm * W//sm, C * tm * sm^2]
        """
        tm = self.visual_temporal_merge_size
        sm = self.visual_spatial_merge_size
        T, C, H, W = x.shape

        assert T % tm == 0, f"T={T} 不能被 temporal_merge_size={tm} 整除"
        assert H % sm == 0 and W % sm == 0, (
            f"H={H} 或 W={W} 不能被 spatial_merge_size={sm} 整除"
        )

        # [T, C, H, W] -> [T, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # 分解合并维度
        x = x.view(
            T // tm,        # 合并后时间步
            tm,             # 时间合并块
            H // sm,        # 合并后空间高
            sm,             # 空间合并块（高）
            W // sm,        # 合并后空间宽
            sm,             # 空间合并块（宽）
            C,
        )

        # 将合并块挪到 channel 末尾（与 QwenVL 完全一致）
        x = x.permute(0, 2, 4, 6, 1, 3, 5).contiguous()

        # 展平为 token 序列
        x = x.reshape(
            (T // tm) * (H // sm) * (W // sm),
            C * tm * sm * sm,
        )
        return x  # [N_tokens, merged_dim]

    def forward(
        self,
        video_embeds: torch.Tensor,
        spatial_embeds_list: List[torch.Tensor],
        grid_thw: torch.Tensor,
        patch_start_idx: Union[int, List[int], torch.Tensor],
    ) -> torch.Tensor:
        # A. 参数预处理
        p_start = (
            patch_start_idx[0]
            if isinstance(patch_start_idx, (list, torch.Tensor))
            else patch_start_idx
        )
        # t_orig：QwenVL 实际处理的帧数（进入 merge 前）
        t_orig, h_orig, w_orig = grid_thw[0].tolist()

        # B. 提取多尺度特征，还原空间结构
        #    注意：这里 x 的 batch 维度对应 DPT 的 dpt_nframes 帧
        layers_features = []
        for i, idx in enumerate(self.layer_indices):
            x = spatial_embeds_list[0][idx][:, p_start:]  # [T_dpt, P, D]
            x = self.input_norms[i](x)
            # 用 h_orig/w_orig 还原空间结构（DPT 和 QwenVL 空间分辨率相同）
            x = x.transpose(1, 2).reshape(-1, x.shape[-1], h_orig, w_orig)
            # x: [T_dpt, D, H, W]
            x = self.projects[i](x)
            x = self.rn_layers[i](x)
            layers_features.append(x)

        # C. 级联融合（DPT refinenet）
        l1, l2, l3, l4 = layers_features
        target_size = (h_orig, w_orig)

        path4 = self.refinenet4(l4, size=target_size)
        path3 = self.refinenet3(path4, l3, size=target_size)
        path2 = self.refinenet2(path3, l2, size=target_size)
        path1 = self.refinenet1(path2, l1, size=target_size)
        # path1: [T_dpt, features, H, W]，T_dpt=16

        # D. 【跳帧】将 DPT 的 T_dpt 帧降采样到 QwenVL 的 t_orig 帧
        #    stride = T_dpt / t_orig = 16 / 8 = 2，即每隔一帧取一帧
        path1 = self._frame_subsample(path1, t_target=int(t_orig))
        # path1: [t_orig, features, H, W]，t_orig=8

        # E. QwenVL 风格时空 merge
        dpt_out = self._qwenvl_merge(path1)
        # dpt_out: [t_orig//tm * H//sm * W//sm, features * tm * sm^2]
        # token 数量与 video_embeds 完全对齐 ✓

        # F. 归一化 + 投影到语言空间
        dpt_out = self.ln_q(dpt_out)
        dpt_out = self.output_proj(dpt_out)   # [N_tokens, language_dim]
        dpt_out = self.dpt_out_norm(dpt_out)

        # G. 门控残差融合
        combined = torch.cat([video_embeds, dpt_out], dim=-1)
        fused = self.fusion_proj(combined)
        return self.final_norm(video_embeds + torch.tanh(self.gate) * fused)

    def print_trainable_parameters(self) -> None:
        is_connector_trainable = any(param.requires_grad for param in self.parameters())
        print(f"DPTConnector 可训练状态: {is_connector_trainable}")
        print(f"当前 gate 值: {self.gate.item():.4f}, tanh(gate): {torch.tanh(self.gate).item():.4f}")
# class DPTConnector(nn.Module):
#     def __init__(
#         self,
#         vggt_dim: int = 2048,
#         language_dim: int = 2048,
#         features: int = 256,
#         out_channels: List[int] = [256, 512, 1024, 2048],
#         layer_indices: List[int] = [7, 11, 14, 23],
#     ) -> None:
#         super().__init__()
#         self.layer_indices = layer_indices
        
#         # ========== 新增：输入归一化 ==========
#         self.input_norms = nn.ModuleList([
#             nn.LayerNorm(vggt_dim) for _ in out_channels
#         ])
        
#         # 1. 投影层
#         self.projects = nn.ModuleList([
#             nn.Conv2d(vggt_dim, oc, kernel_size=1) for oc in out_channels
#         ])

#         # 2. 映射到 DPT 内部通道数
#         self.rn_layers = nn.ModuleList([
#             nn.Conv2d(oc, features, kernel_size=3, padding=1, bias=False) for oc in out_channels
#         ])

#         # 3. 级联融合模块
#         self.refinenet1 = _make_fusion_block(features)
#         self.refinenet2 = _make_fusion_block(features)
#         self.refinenet3 = _make_fusion_block(features)
#         self.refinenet4 = _make_fusion_block(features, has_residual=False)

#         # 4. 降采样模块
#         self.temporal_spatial_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

#         # 5. 投影到语言空间
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features // 2, language_dim, kernel_size=1)
#         )

#         # ========== 新增：输出归一化 ==========
#         self.dpt_out_norm = nn.LayerNorm(language_dim)

#         # 6. 拼接融合层
#         self.fusion_proj = nn.Linear(language_dim * 2, language_dim)
#         self.final_norm = Qwen2RMSNorm(language_dim, eps=1e-6)

#         # ========== 新增：可学习门控，初始为 0 ==========
#         self.gate = nn.Parameter(torch.zeros(1))
        
#         # ========== 关键：初始化 ==========
#         self._init_weights()

#     def _init_weights(self):
#         """稳定初始化，防止梯度爆炸"""
#         # 零初始化最终融合层，训练初期不破坏原有特征
#         nn.init.zeros_(self.fusion_proj.weight)
#         nn.init.zeros_(self.fusion_proj.bias)
        
#         # 小方差初始化投影层
#         for proj in self.projects:
#             nn.init.normal_(proj.weight, std=0.01)
#             if proj.bias is not None:
#                 nn.init.zeros_(proj.bias)
        
#         # 小方差初始化 rn_layers
#         for rn in self.rn_layers:
#             nn.init.normal_(rn.weight, std=0.01)
        
#         # output_conv 最后一层零初始化
#         nn.init.zeros_(self.output_conv[-1].weight)
#         nn.init.zeros_(self.output_conv[-1].bias)

#     def forward(
#         self,
#         video_embeds: torch.Tensor,
#         spatial_embeds_list: List[torch.Tensor], 
#         grid_thw: torch.Tensor,
#         patch_start_idx: Union[int, List[int], torch.Tensor]
#     ) -> torch.Tensor:
#         # A. 参数预处理
#         p_start = patch_start_idx[0] if isinstance(patch_start_idx, (list, torch.Tensor)) else patch_start_idx
#         t_orig, h_orig, w_orig = grid_thw[0].tolist()
        
#         # B. 提取多尺度特征并还原空间结构
#         layers_features = []
#         for i, idx in enumerate(self.layer_indices):
#             x = spatial_embeds_list[0][idx][:, p_start:]
            
#             # ========== 新增：先归一化 ==========
#             x = self.input_norms[i](x)
            
#             x = x.transpose(1, 2).reshape(-1, x.shape[-1], h_orig, w_orig)
#             x = self.projects[i](x)
#             x = self.rn_layers[i](x)
#             layers_features.append(x)

#         # C. 级联融合
#         l1, l2, l3, l4 = layers_features
#         target_size = (h_orig, w_orig)
        
#         path4 = self.refinenet4(l4, size=target_size)
#         path3 = self.refinenet3(path4, l3, size=target_size)
#         path2 = self.refinenet2(path3, l2, size=target_size)
#         path1 = self.refinenet1(path2, l1, size=target_size)

#         # D. 时空降采样
#         fused_3d = path1.view(-1, *path1.shape[1:]).transpose(0, 1)
#         fused_3d = self.temporal_spatial_pool(fused_3d)
        
#         # E. 映射到语言空间
#         c_f, t_new, h_new, w_new = fused_3d.shape
#         fused_2d = fused_3d.transpose(1, 2).reshape(-1, c_f, h_new, w_new)
#         dpt_out = self.output_conv(fused_2d)
#         dpt_out = dpt_out.flatten(2).transpose(1, 2).reshape(-1, dpt_out.shape[1])
        
#         # ========== 新增：输出归一化 ==========
#         dpt_out = self.dpt_out_norm(dpt_out)
        
#         # F. 最终融合
#         combined = torch.cat([video_embeds, dpt_out], dim=-1)
#         fused = self.fusion_proj(combined)
        
#         # ========== 关键修改：门控残差连接 ==========
#         # tanh(gate) 初始接近 0，训练过程中逐渐学习合适的缩放
#         return self.final_norm(video_embeds + torch.tanh(self.gate) * fused)

#     def print_trainable_parameters(self) -> None:
#         is_connector_trainable = any(param.requires_grad for param in self.parameters())
#         print(f"DPTConnector 可训练状态: {is_connector_trainable}")
#         print(f"当前 gate 值: {self.gate.item():.4f}, tanh(gate): {torch.tanh(self.gate).item():.4f}")

        
class MLPAddConnector(nn.Module):

    """
    MLPAddConnector: 将视觉空间嵌入进行下采样(Merge)并映射到语言模型空间的连接器。
    它不仅改变维度，还通过合并相邻的 token 来减少 token 的数量。
    """
    def __init__(
        self, vggt_dim, language_dim, spatial_embeds_layer_idx, visual_temporal_merge_size, visual_spatial_merge_size
    ) -> None:
        super().__init__()
        self.vggt_dim = vggt_dim        # 原始视觉特征维度
        self.language_dim = language_dim  # 目标语言模型维度 (如 4096)

        # 指定从视觉模型输出的哪一层提取空间特征
        self.spatial_embeds_layer_idx = spatial_embeds_layer_idx
        print(f"Using spatial_embeds_layer_idx: {self.spatial_embeds_layer_idx}")

        # 时间和空间维度的合并倍数 (例如 merge_size=2，则 2x2 的空间块合并为 1 个 token)
        self.visual_temporal_merge_size = visual_temporal_merge_size
        self.visual_spatial_merge_size = visual_spatial_merge_size

        # 计算合并后的总维度：(原始维度*2) * 时间倍数 * 空间倍数的平方
        # 这里的 *2 暗示输入可能是某种拼接后的特征
        self.merged_dim = (self.vggt_dim * 2) * self.visual_temporal_merge_size * self.visual_spatial_merge_size**2
        
        # 使用 Qwen2 的 RMSNorm 进行归一化
        self.ln_q = Qwen2RMSNorm(self.merged_dim, eps=1e-6)
        
        # 投影层：将合并后的高维特征映射到语言模型维度
        self.mlp = nn.Sequential(
            nn.Linear(self.merged_dim, self.merged_dim),
            nn.GELU(),
            nn.Linear(self.merged_dim, self.language_dim),
        )

    def preprocess_spatial_embeds(
        self,
        spatial_embeds_list: List[List[torch.Tensor]], # 视觉特征列表
        patch_start_idx: List[int],                    # 真正的图像 patch 开始的索引（避开 cls_token 等）
        grid_thw: torch.Tensor,                        # 包含视频/图像的 [T, H, W] 结构信息
    ) -> torch.Tensor:
        """
        预处理函数：负责将一维的 token 序列还原为 3D (T,H,W) 结构，执行合并，再拉平。
        """
        all_spatial_embeds = []
        grid_idx = 0

        for i, spatial_embeds_item in enumerate(spatial_embeds_list):
            # 1. 提取指定层的特征并增加 batch 维度
            spatial_embeds = spatial_embeds_item[self.spatial_embeds_layer_idx].unsqueeze(0)
            
            # 2. 裁剪掉非 patch token (如 cls_token, register tokens 等)
            spatial_embeds = spatial_embeds[:, :, patch_start_idx[i]:]

            B, S, P, DD = spatial_embeds.shape # Batch, Sequence(Time), Patch(H*W), Dimension
            assert B == 1, "目前仅支持 batch size 为 1 的处理"

            # 3. 根据 grid_thw 确定视频/图像的实际 T, H, W 结构
            accumulated_t = 0
            if grid_idx >= len(grid_thw):
                raise ValueError(f"grid_thw 行数不足以匹配 spatial_embeds {i}")

            # 循环计算，直到覆盖当前样本的所有帧 S
            while accumulated_t * self.visual_temporal_merge_size < S:
                t, h, w = grid_thw[grid_idx].tolist()
                if accumulated_t == 0:
                    npatch_h, npatch_w = h, w # 记录第一帧的高宽作为基准
                else:
                    assert h == npatch_h and w == npatch_w, "视频内部的空间尺寸必须一致"
                accumulated_t += t
                grid_idx += 1

            npatch_t = accumulated_t # 实际的总帧数

            # 校验：确保 patch 数量和帧数能对齐
            assert P == npatch_h * npatch_w, "patch 数量不匹配"
            assert npatch_t == S // self.visual_temporal_merge_size, "时间轴合并比例不匹配"

            # 4. 核心重组逻辑 (Merging)
            # a. 还原为 2D 空间网格: [B, S, H, W, D] -> [B, S, D, H, W]
            spatial_embeds = (
                spatial_embeds.view(B, S, npatch_h, npatch_w, DD).permute(0, 1, 4, 2, 3).contiguous()
            )

            # b. 利用 view 分解出要合并的维度，并通过 permute 将相邻 patch 的特征凑在一起
            # 这一步非常精妙：它将空间和时间维度拆解，准备在维度末尾进行 Concatenate
            print(f"Before merging, spatial_embeds shape: {spatial_embeds.shape}")
            spatial_embeds = (
                spatial_embeds.view(
                    B,
                    npatch_t,
                    self.visual_temporal_merge_size,
                    DD,
                    npatch_h // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                    npatch_w // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                )
                .permute(0, 1, 4, 6, 5, 7, 3, 2)
                .contiguous()
            )
            print(f"After merging view and permute, spatial_embeds shape: {spatial_embeds.shape}")
            # c. 将合并的维度拉平到通道维度 (Dimension)，实现 token 数量减少，通道数增加
            spatial_embeds = spatial_embeds.reshape(
                B * npatch_t * (npatch_h // self.visual_spatial_merge_size) * (npatch_w // self.visual_spatial_merge_size), 
                -1
            )
            print(f"After merging reshape, spatial_embeds shape: {spatial_embeds.shape}")
            # 确保维度符合 merged_dim
            spatial_embeds = spatial_embeds.view(
                -1, DD * self.visual_temporal_merge_size * self.visual_spatial_merge_size**2
            )
            print(f"After final view, spatial_embeds shape: {spatial_embeds.shape}")
            all_spatial_embeds.append(spatial_embeds)

        # 合并所有样本的特征
        all_spatial_embeds_concated = torch.cat(all_spatial_embeds, dim=0)
        return all_spatial_embeds_concated

    def forward(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        spatial_embeds_list: List[List[torch.Tensor]] = None,
        patch_start_idx: List[int] = None,
        grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播：融合主路径嵌入（video_embeds）和空间路径嵌入（spatial_embeds）。
        """
        assert video_embeds is not None or image_embeds is not None, "必须提供视频或图像嵌入。"

        # 1. 预处理并合并空间 token
        spatial_embeds = self.preprocess_spatial_embeds(spatial_embeds_list, patch_start_idx, grid_thw)

        # 2. 归一化处理并投影到语言空间维度
        spatial_embeds = self.ln_q(spatial_embeds)
        spatial_embeds = self.mlp(spatial_embeds)

        # 3. 残差相加：将处理后的空间嵌入加到原始的主路径嵌入上
        # 这种做法允许模型在原有全局特征基础上，通过加法补充更细致的空间信息
        if image_embeds is not None:
            return image_embeds + spatial_embeds
        else:
            return video_embeds + spatial_embeds

    def print_trainable_parameters(self) -> None:
        """
        打印连接器各部分的训练状态
        """
        is_connector_trainable = any(param.requires_grad for param in self.parameters())
        print(f"MLPAddConnector 可训练状态: {is_connector_trainable}")
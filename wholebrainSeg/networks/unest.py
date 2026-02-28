# limitations under the License.

"""
UNesT: 基于层次化Transformer的医学图像分割网络

本模块实现了UNesT (U-Net with hierarchical Transformer) 模型，
用于3D医学图像分割任务，特别是全脑分割(132/133个脑区)。

主要特点:
- 结合U-Net架构与嵌套Transformer (NesT) 编码器
- 支持3D MRI图像的体素级分割
- 可扩展支持TICV(颅内总体积)和PFV(后颅窝体积)估计

参考论文:
    Xin Yu et al. "UNesT: local spatial representation learning with hierarchical 
    transformer for efficient medical segmentation." Medical Image Analysis, 2023.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

# MONAI库的网络组件
from monai.networks.blocks.dynunet_block import UnetOutBlock  # U-Net输出块
from monai.networks.blocks import Convolution  # 通用卷积块

# 本地模块
from networks.unest_block import UNesTConvBlock, UNestUpBlock, UNesTBlock
from networks.nest_transformer_3D import NestTransformer3D


class UNesT(nn.Module):
    """
    UNesT模型 - 用于全脑分割的Transformer-U-Net混合架构
    
    网络结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         输入: (B, 1, 96, 96, 96)                 │
    │                              ↓                                   │
    │  ┌──────────────┐    ┌──────────────────────────────────┐       │
    │  │   encoder1   │    │         NestTransformer3D          │       │
    │  │  (CNN卷积)    │    │   (层次化Vision Transformer编码器)  │       │
    │  │  1→32通道    │    │   输出多尺度特征: 128/256/512通道    │       │
    │  └──────┬───────┘    └────────────────┬─────────────────┘       │
    │         │ enc0                        │ x, hidden_states        │
    │         ↓                             ↓                          │
    │  ┌──────────────┐            ┌────────────────┐                 │
    │  │   encoder2   │←─x1────────│   encoder10    │ ← 最深特征       │
    │  │  (上采样块)   │            │  (额外下采样)   │                 │
    │  │ 128→64通道   │            │ 512→1024通道   │                 │
    │  └──────┬───────┘            └───────┬────────┘                 │
    │         │ enc1                       │ dec4                     │
    │         ↓                            ↓                          │
    │  ┌──────────────┐            ┌────────────────┐                 │
    │  │   encoder3   │←─x2        │    decoder5    │←─x4 (跳跃连接)   │
    │  │  (CNN卷积)    │            │  (解码块+上采样) │                 │
    │  │ 128→128通道  │            │1024→512通道    │                 │
    │  └──────┬───────┘            └───────┬────────┘                 │
    │         │ enc2                       │ dec3                     │
    │         ↓                            ↓                          │
    │  ┌──────────────┐            ┌────────────────┐                 │
    │  │   encoder4   │←─x3        │    decoder4    │←─x3             │
    │  │  (CNN卷积)    │            │  (解码块+上采样) │                 │
    │  │ 256→256通道  │            │ 512→256通道    │                 │
    │  └──────┬───────┘            └───────┬────────┘                 │
    │         │ enc3                       │ dec2                     │
    │         └──────────────┬─────────────┘                          │
    │                        ↓                                        │
    │                 ┌────────────────┐                              │
    │                 │    decoder3    │←─x2                          │
    │                 │ 256→128通道    │                              │
    │                 └───────┬────────┘                              │
    │                         │ dec1                                  │
    │                         ↓                                        │
    │                 ┌────────────────┐                              │
    │                 │    decoder2    │←─enc1                        │
    │                 │ 128→64通道     │                              │
    │                 └───────┬────────┘                              │
    │                         │ dec0                                  │
    │                         ↓                                        │
    │                 ┌────────────────┐                              │
    │                 │    decoder1    │←─enc0                        │
    │                 │  64→32通道     │                              │
    │                 └───────┬────────┘                              │
    │                         │ out                                    │
    │                         ↓                                        │
    │                 ┌────────────────┐                              │
    │                 │   UnetOutBlock  │                             │
    │                 │  32→133通道     │ (133个脑区分类)              │
    │                 └────────────────┘                              │
    │                         ↓                                        │
    │                   输出: (B, 133, 96, 96, 96)                     │
    └─────────────────────────────────────────────────────────────────┘
    
    跳跃连接: 编码器特征与解码器特征进行concat或add操作
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [96, 96, 96],
        feature_size: int = 16,
        patch_size: int = 4,
        depths: Tuple[int, int, int] = [2, 2, 8],
        num_heads: Tuple[int, int, int] = [4, 8, 16],
        embed_dim: Tuple[int, int, int] = [128, 256, 512],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        初始化UNesT模型
        
        Args:
            in_channels: 输入通道数，MRI通常为1（单通道灰度图）
            out_channels: 输出通道数，即分割类别数（133类脑区+背景）
            img_size: 输入图像尺寸，默认(96, 96, 96)用于滑动窗口推理
            feature_size: 基础特征通道数，决定了网络的宽度
                          实际通道数为 feature_size * N (N=2,4,8,16,32,64)
            patch_size: Transformer的patch大小，将图像分割为patch进行编码
            depths: 每个Transformer层级的深度，即每个层级的block数量
                    [2, 2, 8] 表示3个层级分别有2、2、8个transformer block
            num_heads: 多头注意力机制的头数
                       [4, 8, 16] 头数逐层增加，捕获更丰富的特征
            embed_dim: 嵌入维度，每个层级的特征通道数
                       [128, 256, 512] 通道数逐层增加
            window_size: 窗口注意力的窗口大小（当前版本未使用）
            norm_name: 归一化方式，"instance" 或 "batch"
            conv_block: 是否使用卷积块（当前版本默认False）
            res_block: 是否使用残差连接，推荐True以改善梯度流
            dropout_rate: Dropout概率，范围[0, 1]
        
        Examples::
            # 单通道输入，133类输出，基础特征尺寸16
            >>> net = UNesT(in_channels=1, out_channels=133, img_size=(96,96,96))
            >>> output = net(torch.randn(2, 1, 96, 96, 96))
            >>> print(output.shape)  # torch.Size([2, 133, 96, 96, 96])
        """

        super().__init__()

        # 参数校验：dropout率必须在0-1之间
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
    
        # 保存嵌入维度，用于后续层通道数配置
        self.embed_dim = embed_dim

        # ==================== 编码器部分 ====================
        
        # NestTransformer3D: 层次化Vision Transformer编码器
        # 将输入图像分割为patch，通过transformer提取多尺度特征
        # 输出: x (最终特征), hidden_states_out (各层级特征)
        self.nestViT = NestTransformer3D(
            img_size=96,              # 输入图像尺寸
            in_chans=1,               # 输入通道数(MRI单通道)
            patch_size=patch_size,    # Patch大小，决定特征图下采样率
            num_levels=3,             # 层级数，产生3种尺度特征
            embed_dims=embed_dim,     # 各层级嵌入维度 [128, 256, 512]
            num_heads=num_heads,      # 各层级注意力头数 [4, 8, 16]
            depths=depths,            # 各层级transformer深度 [2, 2, 8]
            num_classes=1000,         # 分类数(仅用于预训练，分割任务不直接使用)
            mlp_ratio=4.,             # MLP扩展比例
            qkv_bias=True,            # QKV投影是否使用偏置
            drop_rate=0.,             # Dropout率
            attn_drop_rate=0.,        # 注意力Dropout率
            drop_path_rate=0.5,       # DropPath率(随机深度正则化)
            norm_layer=None,          # 归一化层(使用默认LayerNorm)
            act_layer=None,           # 激活函数(使用默认GELU)
            pad_type='',              # 填充类型
            weight_init='',           # 权重初始化方式
            global_pool='avg',        # 全局池化方式
        )

        # encoder1: 第一层卷积编码器 (CNN路径)
        # 处理原始输入，提取底层特征
        # 输入: (B, 1, 96, 96, 96) → 输出: (B, 32, 96, 96, 96)
        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,  # 16*2=32通道
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # encoder2: 上采样编码块
        # 将Transformer第1层特征上采样，与CNN路径融合
        # 输入: (B, 128, 24, 24, 24) → 输出: (B, 64, 48, 48, 48)
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],  # 128
            out_channels=feature_size * 4,   # 64
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,          # 2倍上采样
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )
    
        # encoder3: 卷积编码块
        # 处理Transformer第1层特征
        # 输入: (B, 128, 24, 24, 24) → 输出: (B, 128, 24, 24, 24)
        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],  # 128
            out_channels=8 * feature_size,   # 128
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # encoder4: 卷积编码块
        # 处理Transformer第2层特征
        # 输入: (B, 256, 12, 12, 12) → 输出: (B, 256, 12, 12, 12)
        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],  # 256
            out_channels=16 * feature_size,  # 256
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # ==================== 解码器部分 ====================
        
        # decoder5: 最深层解码块
        # 接收最深层特征，进行上采样和跳跃连接
        # 输入: (B, 1024, 3, 3, 3) + skip: (B, 512, 6, 6, 6) 
        # → 输出: (B, 512, 6, 6, 6)
        self.decoder5 = UNesTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],  # 1024 (concat后)
            out_channels=feature_size * 32,    # 512
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # decoder4: 第4层解码块
        # 输入: (B, 512, 6, 6, 6) + skip: (B, 256, 12, 12, 12)
        # → 输出: (B, 256, 12, 12, 12)
        self.decoder4 = UNesTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],     # 512
            out_channels=feature_size * 16,     # 256
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # decoder3: 第3层解码块
        # 输入: (B, 256, 12, 12, 12) + skip: (B, 128, 24, 24, 24)
        # → 输出: (B, 128, 24, 24, 24)
        self.decoder3 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,     # 256
            out_channels=feature_size * 8,      # 128
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # decoder2: 第2层解码块
        # 输入: (B, 128, 24, 24, 24) + skip: (B, 64, 48, 48, 48)
        # → 输出: (B, 64, 48, 48, 48)
        self.decoder2 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,      # 128
            out_channels=feature_size * 4,      # 64
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # decoder1: 第1层解码块（最终解码）
        # 输入: (B, 64, 48, 48, 48) + skip: (B, 32, 96, 96, 96)
        # → 输出: (B, 32, 96, 96, 96)
        self.decoder1 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,      # 64
            out_channels=feature_size * 2,      # 32
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # encoder10: 额外的下采样层
        # 进一步提取深层特征
        # 输入: (B, 512, 6, 6, 6) → 输出: (B, 1024, 3, 3, 3)
        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=32*feature_size,       # 512
            out_channels=64*feature_size,       # 1024
            strides=2,                          # 下采样2倍
            adn_ordering="ADN",                 # Activation->Dropout->Norm
            dropout=0.0,
        )

        # 输出层: 将特征映射到分割类别
        # 输入: (B, 32, 96, 96, 96) → 输出: (B, 133, 96, 96, 96)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        """
        特征投影/重排
        
        将展平的transformer输出重塑为3D特征图格式
        
        Args:
            x: 展平的特征向量 (B, N, hidden_size)
            hidden_size: 隐藏层维度
            feat_size: 特征图空间尺寸 (D, H, W)
        
        Returns:
            重排后的特征图 (B, hidden_size, D, H, W)
        """
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, H, W, C) -> (B, C, D, H, W)
        return x


    def forward(self, x_in):
        """
        前向传播
        
        数据流:
        1. 输入通过NestTransformer提取多尺度特征
        2. 同时通过CNN编码器提取底层特征
        3. 特征逐层融合并上采样
        4. 输出分割预测
        
        Args:
            x_in: 输入图像张量 (B, 1, 96, 96, 96)
        
        Returns:
            logits: 分割预测 (B, 133, 96, 96, 96)
                    每个体素预测133个类别的概率分数
        """
        # ============ 编码阶段 ============
        # NestTransformer前向传播
        # x: 最终层特征 (B, 512, 6, 6, 6)
        # hidden_states_out: 各层级特征列表 [x1, x2, x3, x4]
        x, hidden_states_out = self.nestViT(x_in) 
        
        # CNN编码路径 - 提取底层特征
        enc0 = self.encoder1(x_in)           # (B, 32, 96, 96, 96)
        
        # Transformer各层级特征
        x1 = hidden_states_out[0]            # (B, 128, 24, 24, 24) - 第1层级
        enc1 = self.encoder2(x1)             # (B, 64, 48, 48, 48)  - 上采样后
        
        x2 = hidden_states_out[1]            # (B, 128, 24, 24, 24) - 第1层级(另一分支)
        enc2 = self.encoder3(x2)             # (B, 128, 24, 24, 24)
        
        x3 = hidden_states_out[2]            # (B, 256, 12, 12, 12) - 第2层级
        enc3 = self.encoder4(x3)             # (B, 256, 12, 12, 12)
        
        x4 = hidden_states_out[3]            # (B, 512, 6, 6, 6)    - 第3层级
        enc4 = x4                            # 跳跃连接
        
        # ============ 解码阶段 ============
        # 最深层特征处理
        dec4 = x                             # (B, 512, 6, 6, 6)
        dec4 = self.encoder10(dec4)          # (B, 1024, 3, 3, 3) - 额外下采样
        
        # 逐层上采样并融合跳跃连接
        dec3 = self.decoder5(dec4, enc4)     # (B, 512, 6, 6, 6)   - 与enc4融合
        dec2 = self.decoder4(dec3, enc3)     # (B, 256, 12, 12, 12) - 与enc3融合
        dec1 = self.decoder3(dec2, enc2)     # (B, 128, 24, 24, 24) - 与enc2融合
        dec0 = self.decoder2(dec1, enc1)     # (B, 64, 48, 48, 48)  - 与enc1融合
        out = self.decoder1(dec0, enc0)      # (B, 32, 96, 96, 96)  - 与enc0融合
        
        # 输出层: 生成分割预测
        logits = self.out(out)               # (B, 133, 96, 96, 96)
        
        return logits


class UNesT_ticv(nn.Module):
    """
    UNesT-TICV模型 - 带颅内体积估计的全脑分割模型
    
    继承UNesT架构，额外添加两个输出头:
    1. TICV (Total Intracranial Volume): 颅内总体积
    2. PFV (Posterior Fossa Volume): 后颅窝体积
    
    这两个体积指标在神经影像学研究中具有重要临床意义:
    - TICV: 用于脑萎缩评估的标准化参考
    - PFV: 后颅窝相关疾病的诊断指标
    
    输出结构:
    - logits: (B, 133, 96, 96, 96) - 133类脑区分割
    - logits_ticv: (B, 1, 96, 96, 96) - TICV分割
    - logits_pfv: (B, 1, 96, 96, 96) - PFV分割
    - 最终输出: 三者在通道维度拼接 (B, 135, 96, 96, 96)
    
    网络结构与UNesT类似，区别在于输出层有3个分支。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = [96, 96, 96],
        feature_size: int = 16,
        patch_size: int = 2,
        depths: Tuple[int, int, int, int] = [2, 2, 2, 2],
        num_heads: Tuple[int, int, int, int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        初始化UNesT-TICV模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 分割输出通道数(不含TICV/PFV)
            img_size: 输入图像尺寸
            feature_size: 基础特征通道数
            patch_size: Transformer的patch大小
            depths: Transformer各层级深度
            num_heads: 各层级注意力头数
            window_size: 窗口注意力大小
            norm_name: 归一化方式
            conv_block: 是否使用卷积块
            res_block: 是否使用残差连接
            dropout_rate: Dropout概率
        
        注意: 当前实现使用固定的NestTransformer参数，
              部分传入参数(如depths, num_heads)可能未生效。
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        
        # 固定嵌入维度
        self.embed_dim = [128, 256, 512]

        # NestTransformer编码器 (参数与UNesT相同)
        self.nestViT = NestTransformer3D(
            img_size=96, 
            in_chans=1, 
            patch_size=4, 
            num_levels=3, 
            embed_dims=(128, 256, 512),                 
            num_heads=(4, 8, 16), 
            depths=(2, 2, 8), 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,                
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='', 
            weight_init='', 
            global_pool='avg',
        )

        # ============ 编码器 (与UNesT相同) ============
        self.encoder1 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UNestUpBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )

        self.encoder3 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[0],
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UNesTConvBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[1],
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # ============ 解码器 (与UNesT相同) ============
        self.decoder5 = UNesTBlock(
            spatial_dims=3,
            in_channels=2*self.embed_dim[2],
            out_channels=feature_size * 32,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UNesTBlock(
            spatial_dims=3,
            in_channels=self.embed_dim[2],
            out_channels=feature_size * 16,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UNesTBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 额外下采样层
        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=32*feature_size,
            out_channels=64*feature_size,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        # ============ 输出层 (多任务输出) ============
        # 主输出: 脑区分割 (133类)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
        # TICV输出: 颅内总体积 (二值分割)
        self.out_ticv = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=1)
        # PFV输出: 后颅窝体积 (二值分割)
        self.out_pfv = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=1)

    def proj_feat(self, x, hidden_size, feat_size):
        """
        特征投影/重排
        
        Args:
            x: 展平的特征向量
            hidden_size: 隐藏层维度
            feat_size: 特征图空间尺寸
        
        Returns:
            重排后的特征图
        """
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
            
    def forward(self, x_in):
        """
        前向传播
        
        Args:
            x_in: 输入图像 (B, 1, 96, 96, 96)
        
        Returns:
            logits_out: 拼接的输出张量 (B, 133+1+1, 96, 96, 96)
                       - [:, :133, ...]: 脑区分割预测
                       - [:, 133, ...]: TICV预测
                       - [:, 134, ...]: PFV预测
        """
        # ============ 编码阶段 ============
        x, hidden_states_out = self.nestViT(x_in) 
        
        enc0 = self.encoder1(x_in)           # (B, 32, 96, 96, 96)
        x1 = hidden_states_out[0]            # (B, 128, 24, 24, 24)
        enc1 = self.encoder2(x1)             # (B, 64, 48, 48, 48)
        x2 = hidden_states_out[1]            # (B, 128, 24, 24, 24)
        enc2 = self.encoder3(x2)             # (B, 128, 24, 24, 24)
        x3 = hidden_states_out[2]            # (B, 256, 12, 12, 12)
        enc3 = self.encoder4(x3)             # (B, 256, 12, 12, 12)
        x4 = hidden_states_out[3]
        enc4 = x4                            # (B, 512, 6, 6, 6)
        
        # ============ 解码阶段 ============
        dec4 = x                             # (B, 512, 6, 6, 6)
        dec4 = self.encoder10(dec4)          # (B, 1024, 3, 3, 3)
        dec3 = self.decoder5(dec4, enc4)     # (B, 512, 6, 6, 6)
        dec2 = self.decoder4(dec3, enc3)     # (B, 256, 12, 12, 12)
        dec1 = self.decoder3(dec2, enc2)     # (B, 128, 24, 24, 24)
        dec0 = self.decoder2(dec1, enc1)     # (B, 64, 48, 48, 48)
        out = self.decoder1(dec0, enc0)      # (B, 32, 96, 96, 96)
        
        # ============ 多任务输出 ============
        logits = self.out(out)               # (B, 133, 96, 96, 96) - 脑区分割
        logits_ticv = self.out_ticv(out)     # (B, 1, 96, 96, 96)   - TICV
        logits_pfv = self.out_pfv(out)       # (B, 1, 96, 96, 96)   - PFV
        
        # 在通道维度拼接所有输出
        logits_out = torch.cat((logits, logits_ticv, logits_pfv), 1)  # (B, 135, 96, 96, 96)
        
        return logits_out

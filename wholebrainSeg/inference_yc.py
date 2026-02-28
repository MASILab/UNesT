"""
UNesT模型完整推理脚本（带预处理和后处理）

本脚本实现了完整的全脑分割推理流程，包括：
1. 颅骨剥离 (Skull Stripping) - 使用ROBEX
2. 强度归一化 - 使用FCM (Fuzzy C-Means)
3. 图像预处理 - 重采样、方向校正、裁剪
4. 模型推理 - 滑动窗口推理
5. 后处理 - 最大连通域提取

依赖库:
    - torch: PyTorch深度学习框架
    - monai: 医学图像分析框架
    - nibabel: NIfTI文件读写
    - ants: ANTs医学图像处理
    - pyrobex: ROBEX颅骨剥离
    - intensity_normalization: 强度归一化
    - skimage: 图像处理

使用示例:
    python inference_yc.py \
        --data_dir /path/to/test/images \
        --model_path /path/to/model \
        --results_dir /path/to/results \
        --overlap 0.2 \
        --device 0

输出文件:
    - {原文件名}: 原始图像副本
    - strip_{原文件名}: 颅骨剥离后的图像
    - norm_{原文件名}: 归一化后的图像
    - label_{原文件名}: 最终分割结果

作者: AI Assistant
"""

import torch
import numpy as np
import os
from monai.inferers import sliding_window_inference
from monai import transforms
import nibabel as nib
import argparse
import sys
import ants  # ANTs医学图像处理库
from pyrobex.robex import robex  # ROBEX颅骨剥离工具
from intensity_normalization.normalize.fcm import FCMNormalize  # FCM强度归一化
from intensity_normalization.typing import Modality, TissueType
from skimage import measure  # 图像处理工具

# 禁止Python写入.pyc字节码文件
sys.dont_write_bytecode = True


# ==================== 辅助函数 ====================

def create_nonzero_mask(data):
    """
    创建非零区域掩码
    
    通过检测所有通道中的非零体素，创建一个二值掩码，
    标识图像中的有效数据区域。
    
    Args:
        data: numpy.ndarray, 形状为(C, H, W, D)的多通道图像
    
    Returns:
        numpy.ndarray: 形状为(H, W, D)的布尔掩码，True表示有效区域
    
    Example:
        >>> data = np.random.rand(1, 100, 100, 100)
        >>> mask = create_nonzero_mask(data)
        >>> mask.shape
        (100, 100, 100)
    """
    from scipy.ndimage import binary_fill_holes
    
    # 初始化全零掩码
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    
    # 遍历所有通道，合并非零区域
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    
    # 填充内部的空洞（确保掩码是连通的）
    nonzero_mask = binary_fill_holes(nonzero_mask)
    
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    """
    从掩码中提取边界框（Bounding Box）
    
    找到掩码中非零区域的最小外接矩形，
    用于裁剪图像以减少计算量。
    
    Args:
        mask: numpy.ndarray, 3D二值掩码
        outside_value: int, 表示背景的值（默认0）
    
    Returns:
        list: 三个维度的边界索引 [[z_min, z_max], [x_min, x_max], [y_min, y_max]]
    
    Example:
        >>> mask = np.zeros((50, 50, 50))
        >>> mask[10:30, 10:30, 10:30] = 1
        >>> bbox = get_bbox_from_mask(mask)
        >>> print(bbox)
        [[10, 30], [10, 30], [10, 30]]
    """
    # 获取非零体素的坐标
    mask_voxel_coords = np.where(mask != outside_value)
    
    # 计算各维度的最小和最大索引
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description='全脑分割推理脚本')
parser.add_argument('--data_dir', type=str, default='data',
                    help='测试图像目录路径')
parser.add_argument('--model_path', type=str, default='model',
                    help='训练模型目录路径')
parser.add_argument('--results_dir', type=str, default='result',
                    help='结果保存目录路径')
parser.add_argument('--overlap', type=float, default=0.2,
                    help='滑动窗口推理的重叠率 (默认: 0.2)')
parser.add_argument('--device', type=int, default=0,
                    help='GPU设备ID (默认: 0)')

args = parser.parse_args()

# ==================== 设备配置 ====================
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# ==================== 图像预处理Pipeline ====================
# 定义推理时的完整预处理流程
img_transform = transforms.Compose(
    [
        # 1. 加载NIfTI图像（保留元数据）
        transforms.LoadImaged(keys=['image'], image_only=False),
        
        # 2. 确保通道优先格式: (H,W,D) -> (1,H,W,D)
        transforms.EnsureChannelFirstd(keys=['image']),
        
        # 3. 重采样到1mm各向同性体素
        #    目标spacing: (1, 1, 1) mm
        transforms.Spacingd(keys=['image'], pixdim=[1, 1, 1], mode=("bilinear")),
        
        # 4. 统一方向为RAS (Right-Anterior-Superior)
        #    确保所有图像方向一致
        transforms.Orientationd(keys=['image'], axcodes="RAS"),
        
        # 5. 裁剪前景区域（去除背景）
        #    减少计算量，提高推理效率
        transforms.CropForegroundd(keys=["image"], source_key="image"),
        
        # 6. 转换为PyTorch张量
        transforms.ToTensord(keys=["image"], dtype=torch.float32)
    ]
)

# ==================== 推理参数配置 ====================
# 滑动窗口的ROI大小（与训练时一致）
roi_size = (96, 96, 96)

# FCM强度归一化器
# 基于白质(WM)的模糊C均值聚类进行归一化
# 这种方法可以使不同扫描的强度分布一致
fcm_norm = FCMNormalize(tissue_type=TissueType.WM)

# 获取所有测试文件
dataAll = os.listdir(args.data_dir)

# ==================== 模型加载 ====================
# 加载完整的PyTorch模型（包含结构和权重）
model = torch.load(os.path.join(args.model_path, 'new_norm_rstrip_matexp_dicom_model.pth'))
model.to(device)
model.eval()  # 设置为评估模式

print(f"开始推理，共 {len(dataAll)} 个文件")
print(f"设备: {device}")
print(f"滑动窗口重叠率: {args.overlap}")

# ==================== 推理循环 ====================
with torch.no_grad():  # 禁用梯度计算以节省内存
    for idx, ele in enumerate(dataAll):
        print(f"\n处理 [{idx+1}/{len(dataAll)}]: {ele}")
        
        # ============ 步骤1: 保存原始图像副本 ============
        # 加载原始NIfTI图像
        predata = nib.load(os.path.join(args.data_dir, ele))
        
        # 保存到结果目录（创建副本）
        nib.Nifti1Image(predata.get_fdata(), predata.affine).to_filename(
            os.path.join(args.results_dir, ele)
        )
        
        # ============ 步骤2: 颅骨剥离 (Skull Stripping) ============
        # 使用ROBEX进行自动颅骨剥离
        # ROBEX是一种基于学习的颅骨剥离方法，适用于T1加权MRI
        print("  - 颅骨剥离中...")
        preStripData = nib.load(os.path.join(args.results_dir, ele))
        stripData, _ = robex(preStripData)  # stripData为剥离后的脑组织
        
        # 保存颅骨剥离结果
        nib.Nifti1Image(stripData.get_fdata(), stripData.affine).to_filename(
            os.path.join(args.results_dir, 'strip_' + ele)
        )
        
        # ============ 步骤3: 图像预处理 ============
        print("  - 图像预处理中...")
        
        # 构建数据字典用于MONAI transforms
        data_dict = [{'image': os.path.join(args.results_dir, 'strip_' + ele)}]
        dict1 = img_transform(data_dict[0])
        
        # 提取预处理后的图像张量
        data_img = dict1['image'].as_tensor()
        data_img = np.squeeze(data_img.numpy())  # 移除单维度: (1,H,W,D) -> (H,W,D)
        
        # 处理异常值
        data_img[np.isnan(data_img)] = 0  # NaN替换为0
        data_img[data_img < 0] = 0        # 负值替换为0
        
        # 获取affine矩阵（用于保存NIfTI）
        data_affine = dict1['image'].affine
        
        # ============ 步骤4: FCM强度归一化 ============
        # 使用模糊C均值聚类基于白质进行归一化
        # 这一步对于跨扫描仪、跨协议的数据一致性很重要
        print("  - 强度归一化中...")
        data_img = fcm_norm(data_img)
        
        # 保存归一化后的图像
        nib.Nifti1Image(data_img, data_affine).to_filename(
            os.path.join(args.results_dir, 'norm_' + ele)
        )
        
        # ============ 步骤5: 模型推理 ============
        print("  - 模型推理中...")
        
        # 添加批次和通道维度: (H,W,D) -> (1,1,H,W,D)
        data_img = np.expand_dims(data_img, axis=0)
        data_img = torch.unsqueeze(torch.tensor(data_img), dim=0).to(device)
        
        # 滑动窗口推理
        # 对于大体积3D图像，避免GPU显存溢出
        # 参数:
        #   - inputs: 输入张量
        #   - roi_size: 滑动窗口大小
        #   - sw_batch_size: 每次处理的窗口数
        #   - predictor: 分割模型
        #   - overlap: 窗口重叠率
        #   - device: 推理设备
        data_img = sliding_window_inference(
            data_img, 
            roi_size, 
            1,           # sw_batch_size
            model, 
            overlap=args.overlap, 
            device=device
        )
        
        # Softmax获取概率分布，然后取最大值作为预测类别
        # 形状变化: (1, 133, H, W, D) -> (1, 1, H, W, D) -> (H, W, D)
        data_img = torch.softmax(data_img, 1).detach().cpu().numpy()
        data_img = np.squeeze(np.argmax(data_img, axis=1).astype(np.uint8))
        
        # ============ 步骤6: 后处理 - 最大连通域提取 ============
        # 目的: 去除分割结果中的孤立噪点和小区域
        # 只保留最大的连通区域
        print("  - 后处理中...")
        
        # 创建二值掩码（所有分割区域）
        img_tmp = data_img.copy()
        img_tmp[img_tmp > 0] = 1
        
        # 使用skimage进行连通域标记
        # labels: 标记图像，每个连通域有唯一标签
        # num: 连通域数量
        labels, num = measure.label(img_tmp, background=0, return_num=True)
        
        # 计算每个连通域的属性（面积等）
        props = measure.regionprops(labels)
        areas = [props[i].area for i in range(len(props))]
        
        # 只保留最大的连通域
        # np.argmax(areas) + 1 因为标签从1开始
        data_img[labels != np.argmax(areas) + 1] = 0
        
        # ============ 步骤7: 保存最终结果 ============
        output_path = os.path.join(args.results_dir, 'label_' + ele)
        nib.Nifti1Image(data_img, data_affine).to_filename(output_path)
        
        print(f"  - 结果已保存: {output_path}")

print("\n" + "=" * 50)
print("推理完成!")
print("=" * 50)

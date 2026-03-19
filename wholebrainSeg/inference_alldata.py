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
import gc
import ants  # ANTs医学图像处理库
from pyrobex.robex import robex  # ROBEX颅骨剥离工具
from networks.unest import UNesT
from intensity_normalization.normalizers.individual.fcm import FCMNormalizer
from intensity_normalization.domain.models import TissueType
from intensity_normalization.adapters.images import NumpyImageAdapter
from skimage import measure  # 图像处理工具
import time
from numba import njit
from scipy.ndimage import binary_dilation, binary_erosion

# 禁止Python写入.pyc字节码文件
sys.dont_write_bytecode = True

# 添加commen_utils到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'commen_utils'))
try:
    from HD_BET.run import run_hd_bet
    import HD_BET
    HAS_HD_BET = True
except ImportError:
    HAS_HD_BET = False
    print("警告: HD-BET未安装，去颅骨功能不可用")
    print("安装方法: pip install HD-BET 或将commen_utils添加到PYTHONPATH")


# ==================== 辅助函数 ====================

def allFilter(in_arr):
    arr_tmp = in_arr.copy()
    arr_tmp[arr_tmp>0] = 1
    labels,num = measure.label(arr_tmp,background= 0,connectivity=1,return_num= True)
    props = measure.regionprops(labels)
    areas = [props[i].area for i in range(len(props))]
    sorted_id = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
    bool_arr = labels == (sorted_id[0] + 1)
    if len(areas) > 1:
        if 0.1 > areas[sorted_id[1]]/areas[sorted_id[0]] > 0.008:
            print('area')
            bool_arr2 = labels == (sorted_id[1] + 1)
            bool_arr = bool_arr + bool_arr2
    in_arr[~bool_arr] = 0
    return in_arr

def labelFilter(in_arr):
    bool_arr = np.zeros(in_arr.shape)
    for label in range(1, 97):
        label_arr = np.zeros(in_arr.shape)
        label_arr[in_arr == label] = 1
        if label_arr.any():
            labels, num = measure.label(label_arr, background=0, connectivity=1, return_num=True)
            props = measure.regionprops(labels)
            areas = [props[i].area for i in range(len(props))]
            sorted_id = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
            label_bool_arr = labels == (sorted_id[0] + 1)
            bool_arr = bool_arr + label_arr - label_bool_arr
    coo_arr = np.where(bool_arr == 1)
    return coo_arr

@njit
def medianFilter(in_arr, coo_arr,s = 15):
    edge = int((s - 1) / 2)
    new_arr = in_arr.copy()
    new_tmp_arr = in_arr.copy()
    for i in range(len(coo_arr[0])):
        x1 = coo_arr[0][i] - edge
        x2 = coo_arr[0][i] + edge + 1
        y1 = coo_arr[1][i] - edge
        y2 = coo_arr[1][i] + edge + 1
        z1 = coo_arr[2][i] - edge
        z2 = coo_arr[2][i] + edge + 1
        if x1 < 0:
            x1 = 0
        if x2 > in_arr.shape[0]:
            x2 = in_arr.shape[0]
        if y1 < 0:
            y1 = 0
        if y2 > in_arr.shape[1]:
            y2 = in_arr.shape[1]
        if z1 < 0:
            z1 = 0
        if z2 > in_arr.shape[2]:
            z2 = in_arr.shape[2]
        # new_arr[coo_arr[0][i], coo_arr[1][i], coo_arr[2][i]] = np.nanmedian(new_tmp_arr[x1:x2, y1:y2, z1:z2])
        tmp_arry = new_tmp_arr[x1:x2, y1:y2, z1:z2].flatten()
        tmp_arry=tmp_arry[tmp_arry!=0]
        new_arr[coo_arr[0][i], coo_arr[1][i], coo_arr[2][i]] = np.median(tmp_arry)
    return new_arr
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
def run_bet( input_img_path: str, output_bet_path: str, 
                device: int = 0, keep_mask: bool = True,
                max_retries: int = 3, fallback_to_cpu: bool = True) -> bool:
        """
        使用HD-BET进行去颅骨处理（带重试和GPU/CPU降级机制）
        
        Args:
            input_img_path: 输入图像路径
            output_bet_path: 输出去颅骨图像路径
            device: GPU设备ID，-1表示CPU
            keep_mask: 是否保存脑mask
            max_retries: 最大重试次数
            fallback_to_cpu: GPU失败时是否降级到CPU
            
        Returns:
            bool: 是否成功
        """
        if not HAS_HD_BET:
            print("错误: HD-BET未安装")
            return False
        
        devices_to_try = [device]  # 先尝试指定的设备
        if fallback_to_cpu and device >= 0:
            devices_to_try.append(-1)  # GPU失败后降级到CPU
        
        for current_device in devices_to_try:
            device_name = f"GPU:{current_device}" if current_device >= 0 else "CPU"
            
            for attempt in range(max_retries):
                try:
                    # 清理内存
                    gc.collect()
                    
                    print(f"    HD-BET去颅骨: {os.path.basename(input_img_path)}")
                    print(f"    使用设备: {device_name} (尝试 {attempt + 1}/{max_retries})")
                    
                    run_hd_bet(
                        mri_fnames=input_img_path,
                        output_fnames=output_bet_path,
                        mode="accurate",
                        config_file=os.path.join(HD_BET.__path__[0], "config.py"),
                        device=current_device,
                        postprocess=False,
                        do_tta=False,
                        keep_mask=keep_mask,
                        overwrite=True,
                        bet=True,  # 输出去颅骨后的图像
                    )
                    
                    print(f"    去颅骨完成: {output_bet_path}")
                    
                    # 清理内存
                    gc.collect()
                    
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # 检查是否是GPU内存不足错误
                    is_gpu_error = any(keyword in error_msg.lower() for keyword in 
                                       ['cuda', 'gpu', 'memory', 'out of memory'])
                    
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"    [重试 {attempt + 1}/{max_retries}] 去颅骨失败: {e}")
                        print(f"    等待 {wait_time} 秒后重试...")
                        gc.collect()
                        time.sleep(wait_time)
                    elif is_gpu_error and current_device >= 0 and fallback_to_cpu:
                        print(f"    [降级] GPU失败，尝试使用CPU...")
                        break  # 跳出重试循环，尝试下一个设备（CPU）
                    else:
                        print(f"    [失败] 去颅骨已达最大重试次数")
                        import traceback
                        traceback.print_exc()
                        
        return False
def normalize_intensity(data: np.ndarray):
        """
        强度归一化（Z-score后平移到非负范围）
        
        Args:
            data: 输入图像数据
            
        Returns:
            归一化后的图像数据
        """
        # 仅对前景区域归一化（使用阈值过滤背景）
        mask = data > 0
        if mask.sum() > 0:
            # Z-score归一化
            mean = data[mask].mean()
            std = data[mask].std()
            if std > 0:
                data[mask] = (data[mask] - mean) / std
            
            # 平移到非负范围
            min_val = data[mask].min()
            data[mask] = data[mask] - min_val
            
            # 背景设为0
            data[~mask] = 0
            
        return data
    
def crop_foreground( data: np.ndarray, affine: np.ndarray, return_bounds: bool = False):
    """
    裁剪前景区域（减少背景）
    
    Args:
        data: 输入图像数据
        affine: 仿射矩阵
        return_bounds: 是否返回裁剪边界
        
    Returns:
        裁剪后的数据、更新后的affine，以及可选的裁剪边界
    """
    # 使用Otsu阈值或简单阈值
    mask = data > (data.mean() * 0.1)
    
    # 找到前景边界
    coords = np.where(mask)
    if len(coords[0]) == 0:
        if return_bounds:
            return data, affine, None
        return data, affine
    
    # 添加边界padding
    padding = 10
    min_coords = [max(0, c.min() - padding) for c in coords]
    max_coords = [min(s, c.max() + padding) for c, s in zip(coords, data.shape)]
    
    # 裁剪
    cropped = data[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]
    
    # 正确更新affine：将原点移动到裁剪后的起始位置
    new_affine = affine.copy()
    # affine矩阵将voxel坐标转换为世界坐标，所以需要将裁剪起始点作为新原点
    new_origin = affine @ np.array([min_coords[0], min_coords[1], min_coords[2], 1.0])
    new_affine[:3, 3] = new_origin[:3]
    
    if return_bounds:
        return cropped, new_affine, (min_coords, max_coords)
    return cropped, new_affine



# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description='全脑分割推理脚本')
parser.add_argument('--data_dir', type=str, default='data',
                    help='测试图像目录路径')
parser.add_argument('--model_path', type=str, default='model',
                    help='训练模型目录路径')
parser.add_argument('--results_dir', type=str, default='result',
                    help='结果保存目录路径')
parser.add_argument('--overlap', type=float, default=0.5,
                    help='滑动窗口推理的重叠率 (默认: 0.5)')
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

# FCM强度归一化器（已弃用，改用Z-score标准化）
# 基于白质(WM)的模糊C均值聚类进行归一化
# 这种方法可以使不同扫描的强度分布一致

# 获取所有测试文件
dataAll = os.listdir(args.data_dir)

# ==================== 模型加载 ====================
# 加载完整的PyTorch模型（包含结构和权重）
# 初始化UNesT模型 (使用large配置以匹配预训练权重)
model = UNesT(
    in_channels=1,      # 单通道MRI图像
    out_channels=97,   # 97类输出（96个脑区+背景）
    patch_size=4,
    depths=[2, 2, 8],  # 与训练配置一致
    num_heads=[4, 8, 16],
    embed_dim=[128, 256, 512]
)
ckpt = torch.load(os.path.join(args.model_path, 'new_norm_rstrip_matexp_dicom_model.pt'))
model.load_state_dict(ckpt['state_dict'], strict=True)
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
        original_img_path = os.path.join(args.results_dir, ele)
        
        # 保存到结果目录（创建副本）
        nib.Nifti1Image(predata.get_fdata(), predata.affine).to_filename(
            os.path.join(args.results_dir, ele)
        )
        
        # =====================================================
        # 步骤2: HD-BET去颅骨 - 对original_from_dicom进行去颅骨
        # =====================================================
        # 如果指定了保存路径，保存到指定位置；否则使用临时目录
              
            
        print("  - 颅骨剥离中...")
        preStripData = nib.load(os.path.join(args.results_dir, ele))
        original_bet_path = os.path.join(args.results_dir, 'strip_' + ele)
    
        if HAS_HD_BET:
            print("\n  [去颅骨] original_from_dicom -> original_bet")
            print(f"    输出路径: {original_bet_path}")
            if not run_bet(original_img_path, original_bet_path, keep_mask=False):
                print("  警告: 去颅骨失败，使用原始图像进行配准")
                original_bet_path = original_img_path
        else:
            print("  警告: HD-BET不可用，使用原始图像进行配准")
            original_bet_path = original_img_path       


 
        
        # # 保存颅骨剥离结果
        # nib.Nifti1Image(stripData.get_fdata(), stripData.affine).to_filename(
        #     os.path.join(args.results_dir, 'strip_' + ele)
        # )
        
        # ============ 步骤3: 图像预处理 ============
        print("  - 图像预处理中...")
        
        # 加载去颅骨图像（或原始图像，如果去颅骨失败）
        nii = nib.load(original_bet_path)
        data = nii.get_fdata()
        affine = nii.affine
        
        # 获取原始spacing
        original_spacing = nii.header.get_zooms()[:3]
        target_spacing = [1, 1, 1]
        
        # 计算重采样比例
        zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
        import scipy.ndimage as ndimage
        if not np.allclose(zoom_factors, 1.0):
            # 重采样
            order =  3  # 标签使用最近邻插值
            data = ndimage.zoom(data, zoom_factors, order=order)
            
            # 正确更新affine矩阵
            # affine的方向矩阵部分表示每个体素在世界坐标系中的向量
            # 当体素数量增加(zoom > 1)时，每个体素变小，affine缩放因子应为 1/zoom
            scale_factors = [1.0 / z for z in zoom_factors]
            scale_matrix = np.diag(scale_factors + [1.0])
            affine = affine @ scale_matrix
        
    
        # 图像归一化（Z-score）
        data = normalize_intensity(data)
        
        # 裁剪前景（减少背景区域）
        data, affine, crop_bounds = crop_foreground(data, affine, return_bounds=True)
    
        
        # 创建新的header，确保正确的spacing
        header = nii.header.copy()
        # 更新header中的pixdim信息
        header['pixdim'][1:4] = target_spacing
        
        # 保存处理后的图像（可选，用于调试）
        nii_output_path = os.path.join(args.results_dir, 'processed_' + ele)
        nii_out = nib.Nifti1Image(data.astype(np.float32), affine, header)
        nib.save(nii_out, nii_output_path)

        # 保存用于后处理的掩码（背景区域）
        tmp_mask = (data > 0).astype(np.uint8)  # 前景掩码
        
        # ============ 步骤5: 模型推理 ============
        print("  - 模型推理中...")
        
        # 添加批次和通道维度: (H,W,D) -> (1,1,H,W,D)
        data_img = np.expand_dims(data, axis=0)
        data_img = torch.unsqueeze(torch.tensor(data_img), dim=0).float().to(device)
        
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
        
        time_start = time.time()  # 记录开始时间
        data_img[tmp_mask == 0] = 0  # 使用前景掩码过滤背景
        data_img = allFilter(data_img)
        
 
        
        coo_arr = labelFilter(data_img)
        data_img = medianFilter(data_img,coo_arr)
        
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(f"  后处理耗时: {time_sum:.2f}秒")
        # ============ 步骤7: 保存最终结果 ============
        output_path = os.path.join(args.results_dir, 'label_' + ele)
        nib.Nifti1Image(data_img, affine).to_filename(output_path)
        
        print(f"  - 结果已保存: {output_path}")

print("\n" + "=" * 50)
print("推理完成!")
print("=" * 50)

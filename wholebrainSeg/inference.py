"""
UNesT模型推理脚本

本脚本用于对测试集图像进行推理，输出分割概率图（.npy格式），
供后续ensemble集成使用。

工作流程:
1. 加载预训练的UNesT模型
2. 遍历测试图像目录
3. 使用滑动窗口推理进行预测
4. 保存每个case的概率预测结果

输出格式:
    每个case保存为一个.npy文件，形状为(1, 133, H, W, D)
    包含133个类别的概率分数

使用示例:
    python inference.py \
        --imagesTs_path /path/to/test/images \
        --saved_model_path /path/to/model.pt \
        --base_dir ./outputs \
        --fold 0 \
        --overlap 0.7 \
        --device 0

作者: MONAI Consortium
"""

import os
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label
from networks.unest import UNesT
from monai.config import DtypeLike, KeysCollection

# 解决NIfTI文件四元数精度问题
nib.Nifti1Header.quaternion_threshold = -1e-06


# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description='UNesT Testing')
parser.add_argument('--imagesTs_path', type=str, 
                    help='测试图像目录路径')
parser.add_argument('--saved_model_path', type=str,
                    help='预训练模型权重路径(.pt文件)')
parser.add_argument('--base_dir', type=str, 
                    help='结果保存根目录')
parser.add_argument('--fold', default=0, type=int,
                    help='交叉验证折号，用于区分不同fold的结果')
parser.add_argument('--sw_batch_size', default=1, type=int,
                    help='滑动窗口推理的批次大小')
parser.add_argument('--overlap', default=0.7, type=float,
                    help='滑动窗口重叠率，越大结果越精确但速度越慢')
parser.add_argument('--device', default=0, type=int,
                    help='GPU设备ID')
args = parser.parse_args()

# 启用cuDNN benchmark模式，加速推理
torch.backends.cudnn.benchmark = True

# ==================== 设备配置 ====================
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# ==================== 输出目录配置 ====================
# 结果保存路径: base_dir/pred_{overlap}/fold{N}/
base_save_pred_dir = os.path.join(args.base_dir, 'pred_{}/'.format(args.overlap))
model_name = 'fold{}'.format(args.fold)
checkpoint_dir = args.saved_model_path
path = args.imagesTs_path

# 创建输出目录
results_folder = base_save_pred_dir + model_name
checkpoints = [checkpoint_dir]

if not os.path.exists(base_save_pred_dir):
    os.makedirs(base_save_pred_dir)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# ==================== 构建测试文件列表 ====================
# 解析测试图像目录，提取case ID
ids = []
validation_files = []
files = os.listdir(path)

for file in files:
    if not file.startswith('.'):  # 跳过隐藏文件
        # 从文件名提取case ID
        # 假设文件名格式: {case_id}_xxx.nii.gz 或 {case_id}.nii.gz
        img_id = file.split('_')[0].split('.nii.gz')[0]
        if img_id not in ids:
            ids.append(img_id)
            validation_files.append({
                'label': '',
                'image': [os.path.join(path, file)]
            })

# ==================== 数据预处理Pipeline ====================
# 推理时的预处理变换（与训练时保持一致，但不含数据增强）
val_transforms = transforms.Compose(
    [
        # 加载NIfTI图像
        transforms.LoadImaged(keys=["image"]),
        # 添加通道维度: (H,W,D) -> (1,H,W,D)
        transforms.AddChanneld(keys=["image"]),
        # 强度归一化（仅对非零区域归一化）
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # 转换为PyTorch张量
        transforms.ToTensord(keys=["image"]),
    ]
)

# ==================== 模型加载 ====================
# 滑动窗口推理的ROI大小
img_size = (96, 96, 96)

# 初始化UNesT模型
model = UNesT(
    in_channels=1,      # 单通道MRI图像
    out_channels=133,   # 133类输出（132个脑区+背景）
)

# 加载预训练权重
ckpt = torch.load(checkpoint_dir, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.to(device)

# 设置为评估模式（关闭Dropout等）
model.eval()

# ==================== 创建数据加载器 ====================
val_ds = data.Dataset(data=validation_files, transform=val_transforms)
val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, sampler=None)

# 推理参数
overlap_ratio = args.overlap      # 滑动窗口重叠率
sw_batch_size = args.sw_batch_size  # 滑动窗口批次大小

# ==================== 推理循环 ====================
with torch.no_grad():
    i = 0
    for idx, batch_data in enumerate(val_loader):
        # 获取case名称（从文件路径提取）
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]
        
        print('##############  Inference case {}-{}  ##############'.format(idx, case_name))
        
        # 获取输入图像
        image = batch_data['image'].to(device)
        
        # 保存原始affine矩阵（用于后续保存NIfTI文件）
        affine = batch_data['image_meta_dict']['original_affine'][0].numpy()
        
        # 初始化输出
        infer_outputs = 0.0
        
        # ==================== 滑动窗口推理 ====================
        # 对于大体积3D图像，使用滑动窗口推理避免显存溢出
        # 参数说明:
        #   - image: 输入图像 (B, C, H, W, D)
        #   - img_size: 滑动窗口ROI大小
        #   - sw_batch_size: 每次处理的窗口数
        #   - model: 分割模型
        #   - overlap: 窗口重叠率，影响边界融合质量
        #   - device: 推理设备（CPU用于节省GPU显存）
        #   - mode: 边界融合模式，'gaussian'使用高斯加权
        pred = sliding_window_inference(
            image, 
            img_size, 
            sw_batch_size, 
            model, 
            overlap=overlap_ratio, 
            device=torch.device('cpu'),  # 在CPU上执行推理以节省GPU显存
            mode='gaussian'              # 高斯加权融合边界区域
        )
        
        # 应用Softmax获取概率分布
        # 输出形状: (B, 133, H, W, D)，每个体素133类的概率
        infer_outputs += torch.nn.Softmax(dim=1)(pred)
        
        # 转移到CPU并转换为numpy数组
        infer_outputs = infer_outputs.cpu().numpy()
        
        # ==================== 保存结果 ====================
        # 再次获取case名称
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]
        
        # 创建case文件夹
        subject_folder = os.path.join(results_folder, case_name)
        
        # 保存概率预测结果为.npy文件
        # 文件名格式: {case_id}.npy
        outNUMPY = results_folder + '/' + case_name + '.npy'
        np.save(outNUMPY, infer_outputs)
        
        print(f'Saved prediction to: {outNUMPY}')
        print(f'Shape: {infer_outputs.shape}')

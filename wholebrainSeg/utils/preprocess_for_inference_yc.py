"""
训练数据预处理脚本 - 匹配 inference_yc.py 推理流程

本脚本确保训练数据的预处理与推理时完全一致，包括：
1. 颅骨剥离 (Skull Stripping) - 使用ROBEX
2. 重采样到1mm各向同性体素
3. 方向校正为RAS
4. 前景裁剪
5. FCM强度归一化（基于白质）

数据结构说明:
原始数据:
    /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
    ├── 76384925062202/
    │   ├── dcm/                      # DICOM序列文件
    │   └── c_results/                # 标注结果
    │       └── cleanup_labelmap96_src.nii.gz

处理后数据:
    /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
    ├── train/
    │   ├── images/                   # 颅骨剥离+归一化后的图像
    │   └── labels/                   # 对应的分割标签（已重采样）
    ├── val/
    └── test/

依赖安装:
    pip install pyrobex
    pip install intensity-normalization
    pip install antspyx
    pip install monai
    pip install nibabel

作者: AI Assistant
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import torch

# MONAI transforms
from monai import transforms

# 颅骨剥离和强度归一化
try:
    from pyrobex.robex import robex
    HAS_ROBEX = True
except ImportError:
    HAS_ROBEX = False
    print("警告: pyrobex未安装，将跳过颅骨剥离步骤")
    print("安装方法: pip install pyrobex")

try:
    from intensity_normalization.normalizers.individual.fcm import FCMNormalizer
    from intensity_normalization.domain.models import TissueType
    HAS_FCM = True
except ImportError:
    HAS_FCM = False
    print("警告: intensity-normalization未安装，将使用Z-score归一化")
    print("安装方法: pip install intensity-normalization")

# 设置NIfTI精度阈值
nib.Nifti1Header.quaternion_threshold = -1e-06


class PreprocessForInferenceYC:
    """
    匹配 inference_yc.py 的训练数据预处理器
    
    确保训练数据的处理流程与推理时完全一致：
    1. 颅骨剥离 (ROBEX)
    2. 重采样 (1mm³)
    3. 方向校正 (RAS)
    4. 前景裁剪
    5. FCM强度归一化
    """
    
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 skip_skull_strip: bool = False,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        初始化预处理器
        
        Args:
            source_dir: 原始数据根目录
            output_dir: 处理后数据输出目录
            skip_skull_strip: 是否跳过颅骨剥离（如果数据已去颅骨）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.skip_skull_strip = skip_skull_strip
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 创建输出目录
        self._create_output_dirs()
        
        # FCM归一化器
        if HAS_FCM:
            self.fcm_norm = FCMNormalizer(tissue_type=TissueType.WM)
        else:
            self.fcm_norm = None
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'json'), exist_ok=True)
    
    def skull_strip(self, image_path: str, output_path: str):
        """
        颅骨剥离 - 使用ROBEX
        
        Args:
            image_path: 输入NIfTI图像路径
            output_path: 输出颅骨剥离后图像路径
            
        Returns:
            bool: 是否成功
        """
        if not HAS_ROBEX:
            print("    pyrobex不可用，复制原始图像...")
            import shutil
            shutil.copy(image_path, output_path)
            return True
        
        try:
            print("    颅骨剥离中...")
            nii_img = nib.load(image_path)
            stripped, _ = robex(nii_img)
            
            # 保存结果
            nib.Nifti1Image(stripped.get_fdata(), stripped.affine).to_filename(output_path)
            return True
            
        except Exception as e:
            print(f"    颅骨剥离失败: {e}")
            # 失败时复制原始图像
            import shutil
            shutil.copy(image_path, output_path)
            return False
    
    def preprocess_image(self, image_path: str, output_path: str):
        """
        图像预处理 - 完全匹配 inference_yc.py 流程
        
        步骤:
        1. 加载图像
        2. 重采样到1mm各向同性
        3. 方向校正为RAS
        4. 前景裁剪
        5. FCM强度归一化
        
        Args:
            image_path: 输入图像路径（已去颅骨）
            output_path: 输出图像路径
        """
        # 定义预处理pipeline - 与 inference_yc.py 完全一致
        img_transform = transforms.Compose([
            # 加载NIfTI图像
            transforms.LoadImage(image_only=False),
            
            # 确保通道优先格式
            transforms.EnsureChannelFirst(),
            
            # 重采样到1mm各向同性体素
            transforms.Spacing(pixdim=[1, 1, 1], mode="bilinear"),
            
            # 统一方向为RAS
            transforms.Orientation(axcodes="RAS"),
            
            # 裁剪前景区域
            transforms.CropForeground(),
            
            # 转换为张量
            transforms.ToTensor(dtype=torch.float32)
        ])
        
        # 执行预处理
        data_dict = img_transform(image_path)
        data_img = data_dict.as_tensor() if hasattr(data_dict, 'as_tensor') else data_dict['image']
        data_img = np.squeeze(data_img.numpy() if hasattr(data_img, 'numpy') else data_img.cpu().numpy())
        
        # 处理异常值
        data_img[np.isnan(data_img)] = 0
        data_img[data_img < 0] = 0
        
        # 获取affine
        data_affine = data_dict.affine if hasattr(data_dict, 'affine') else np.eye(4)
        
        # FCM强度归一化
        if self.fcm_norm is not None:
            try:
                print("    FCM归一化中...")
                data_img = self.fcm_norm(data_img)
            except Exception as e:
                print(f"    FCM归一化失败，使用Z-score: {e}")
                data_img = self._zscore_normalize(data_img)
        else:
            data_img = self._zscore_normalize(data_img)
        
        # 保存结果
        nib.Nifti1Image(data_img, data_affine).to_filename(output_path)
    
    def preprocess_label(self, label_path: str, output_path: str):
        """
        标签预处理
        
        对标签应用与图像相同的几何变换（重采样、方向校正、裁剪）
        但使用最近邻插值保持标签值不变
        
        Args:
            label_path: 输入标签路径
            output_path: 输出标签路径
        """
        # 定义标签预处理pipeline
        # 使用最近邻插值，不进行强度变换
        label_transform = transforms.Compose([
            transforms.LoadImage(image_only=False),
            transforms.EnsureChannelFirst(),
            
            # 使用最近邻插值重采样
            transforms.Spacing(pixdim=[1, 1, 1], mode="nearest"),
            
            # 统一方向为RAS
            transforms.Orientation(axcodes="RAS"),
            
            # 裁剪前景（基于标签本身）
            transforms.CropForeground(),
            
            transforms.ToTensor(dtype=torch.int32)
        ])
        
        # 执行预处理
        data_dict = label_transform(label_path)
        data_label = data_dict.as_tensor() if hasattr(data_dict, 'as_tensor') else data_dict['image']
        data_label = np.squeeze(data_label.numpy() if hasattr(data_label, 'numpy') else data_label.cpu().numpy())
        
        # 获取affine
        data_affine = data_dict.affine if hasattr(data_dict, 'affine') else np.eye(4)
        
        # 保存结果
        nib.Nifti1Image(data_label.astype(np.int32), data_affine).to_filename(output_path)
    
    def _zscore_normalize(self, data: np.ndarray):
        """
        Z-score归一化（备用方法）
        
        Args:
            data: 输入图像数据
            
        Returns:
            归一化后的图像数据
        """
        mask = data > 0
        if mask.sum() > 0:
            mean = data[mask].mean()
            std = data[mask].std()
            if std > 0:
                data = data.copy()
                data[mask] = (data[mask] - mean) / std
        return data
    
    def find_image_and_label(self, case_dir: str):
        """
        查找case的图像和标签文件
        
        Args:
            case_dir: case目录路径
            
        Returns:
            tuple: (image_path, label_path) 或 None
        """
        # 查找图像文件
        image_candidates = [
            # 标注结果目录中的预处理图像
            os.path.join(case_dir, 'c_results', 'brain_preproc_img.nii.gz'),
            os.path.join(case_dir, 'c_results', 'cropped_img.nii.gz'),
            os.path.join(case_dir, 'c_results', '1000_image_0000.nii.gz'),
            # NIfTI格式
            os.path.join(case_dir, 'c_results', 'image.nii.gz'),
            # 根目录
            os.path.join(case_dir, 'image.nii.gz'),
        ]
        
        image_path = None
        for candidate in image_candidates:
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        # 如果没有NIfTI，尝试从DICOM转换
        if image_path is None:
            dcm_dir = os.path.join(case_dir, 'dcm')
            if os.path.exists(dcm_dir):
                temp_nii = os.path.join(case_dir, 'temp_from_dicom.nii.gz')
                if self._dicom_to_nifti(dcm_dir, temp_nii):
                    image_path = temp_nii
        
        # 查找标签文件
        label_candidates = [
            os.path.join(case_dir, 'c_results', 'cleanup_labelmap96_src.nii.gz'),
            os.path.join(case_dir, 'c_results', 'cimage.nii.gz'),
            os.path.join(case_dir, 'c_results', 'label.nii.gz'),
            os.path.join(case_dir, 'label.nii.gz'),
        ]
        
        label_path = None
        for candidate in label_candidates:
            if os.path.exists(candidate):
                label_path = candidate
                break
        
        return image_path, label_path
    
    def _dicom_to_nifti(self, dicom_dir: str, output_path: str):
        """
        DICOM序列转NIfTI
        """
        try:
            import SimpleITK as sitk
            
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
            
            if not dicom_files:
                return False
            
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            sitk.WriteImage(image, output_path)
            return True
            
        except ImportError:
            print("SimpleITK未安装，无法转换DICOM")
            return False
    
    def process_case(self, case_id: str, split: str):
        """
        处理单个case
        
        Args:
            case_id: case标识符
            split: 数据集划分 ('train', 'val', 'test')
            
        Returns:
            bool: 是否成功处理
        """
        case_dir = os.path.join(self.source_dir, case_id)
        
        # 查找文件
        image_path, label_path = self.find_image_and_label(case_dir)
        
        if image_path is None:
            print(f"  跳过 {case_id}: 未找到图像文件")
            return False
        
        if label_path is None:
            print(f"  跳过 {case_id}: 未找到标签文件")
            return False
        
        print(f"  处理 {case_id}...")
        
        # 输出路径
        out_image = os.path.join(self.output_dir, split, 'images', f'{case_id}.nii.gz')
        out_label = os.path.join(self.output_dir, split, 'labels', f'{case_id}_seg.nii.gz')
        
        # 中间文件路径（颅骨剥离后）
        temp_strip = os.path.join(self.output_dir, split, 'images', f'{case_id}_stripped.nii.gz')
        
        try:
            # 步骤1: 颅骨剥离（如果需要）
            if self.skip_skull_strip:
                print("    跳过颅骨剥离（数据已去颅骨）")
                stripped_image = image_path
            else:
                self.skull_strip(image_path, temp_strip)
                stripped_image = temp_strip
            
            # 步骤2: 图像预处理（重采样、方向、裁剪、归一化）
            self.preprocess_image(stripped_image, out_image)
            
            # 步骤3: 标签预处理（重采样、方向、裁剪）
            self.preprocess_label(label_path, out_label)
            
            # 清理中间文件
            if os.path.exists(temp_strip) and not self.skip_skull_strip:
                os.remove(temp_strip)
            
            return True
            
        except Exception as e:
            print(f"  处理失败 {case_id}: {e}")
            return False
    
    def run(self):
        """
        执行完整的预处理流程
        """
        print("=" * 70)
        print("训练数据预处理 - 匹配 inference_yc.py 推理流程")
        print("=" * 70)
        print(f"源目录: {self.source_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"颅骨剥离: {'跳过' if self.skip_skull_strip else '执行'}")
        print()
        
        # 检查依赖
        if not HAS_ROBEX and not self.skip_skull_strip:
            print("错误: 需要安装 pyrobex 进行颅骨剥离")
            print("安装: pip install pyrobex")
            print("或使用 --skip_skull_strip 参数跳过（如果数据已去颅骨）")
            return
        
        # 获取所有case
        all_cases = []
        for item in os.listdir(self.source_dir):
            item_path = os.path.join(self.source_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                if item not in ['train', 'val', 'test', 'json' ]:
                    all_cases.append(item)
        
        print(f"发现 {len(all_cases)} 个case")
        
        if len(all_cases) == 0:
            print("没有找到case，退出")
            return
        
        # 划分数据集
        np.random.seed(self.random_seed)
        np.random.shuffle(all_cases)
        
        n_total = len(all_cases)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_cases = all_cases[:n_train]
        val_cases = all_cases[n_train:n_train + n_val]
        test_cases = all_cases[n_train + n_val:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_cases)}")
        print(f"  验证集: {len(val_cases)}")
        print(f"  测试集: {len(test_cases)}")
        
        # 处理数据
        splits = {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }
        
        success_count = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, cases in splits.items():
            print(f"\n{'='*60}")
            print(f"处理 {split_name} 集 ({len(cases)} cases)")
            print('='*60)
            
            for case_id in tqdm(cases, desc=split_name):
                if self.process_case(case_id, split_name):
                    success_count[split_name] += 1
        
        # 生成JSON
        self._create_json_files()
        
        # 总结
        print("\n" + "=" * 70)
        print("预处理完成!")
        print("=" * 70)
        print(f"训练集: {success_count['train']}/{len(train_cases)}")
        print(f"验证集: {success_count['val']}/{len(val_cases)}")
        print(f"测试集: {success_count['test']}/{len(test_cases)}")
        print(f"\nJSON文件: {self.output_dir}/json/fold0.json")
    
    def _create_json_files(self):
        """生成JSON数据列表"""
        datadict = {'training': [], 'validation': []}
        
        for split_name, key in [('train', 'training'), ('val', 'validation')]:
            images_dir = os.path.join(self.output_dir, split_name, 'images')
            if os.path.exists(images_dir):
                for f in sorted(os.listdir(images_dir)):
                    if f.endswith('.nii.gz') and not f.startswith('temp'):
                        image_rel = f"{split_name}/images/{f}"
                        label_rel = f"{split_name}/labels/{f.replace('.nii.gz', '_seg.nii.gz')}"
                        
                        datadict[key].append({
                            'image': image_rel,
                            'label': label_rel
                        })
        
        json_path = os.path.join(self.output_dir, 'json', 'fold0.json')
        with open(json_path, 'w') as f:
            json.dump(datadict, f, indent=4)
        
        print(f"\n已生成: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='训练数据预处理 - 匹配 inference_yc.py 推理流程'
    )
    parser.add_argument('--source_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164',
                        help='原始数据根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset',
                        help='处理后数据输出目录')
    parser.add_argument('--skip_skull_strip', action='store_true',
                        help='跳过颅骨剥离（如果数据已去颅骨）')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    preprocessor = PreprocessForInferenceYC(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        skip_skull_strip=args.skip_skull_strip,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    preprocessor.run()


if __name__ == '__main__':
    main()

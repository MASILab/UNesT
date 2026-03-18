"""
全脑分割数据集预处理脚本

本脚本用于将原始DICOM影像和标注结果转换为UNesT训练所需的格式。

数据结构说明:
原始数据:
    /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
    ├── 76384925062202/
    │   ├── dcm/                      # DICOM序列文件
    │   │   ├── 1_xxx.dcm
    │   │   ├── 2_xxx.dcm
    │   │   └── ...
    │   ├── c_results/                # 中间文件
    │   ├── original_bet.nii.gz       #  nii.gz文件
    │   ├── label_original_space.nii.gz # nii.gz文件对应的分割标签(96类)
处理后数据:
    /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
    ├── train/
    │   ├── images/                   # 训练图像
    │   │   ├── 76384925062202.nii.gz（76384925062202/original_bet.nii.gz ）
    │   │   └── ...
    │   └── labels/                   # 训练标签（
    │       ├── 76384925062202_seg.nii.gz（76384925062202/label_original_space.nii.gz）
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

作者: AI Assistant
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import pydicom
from pydicom.dataset import Dataset
from collections import defaultdict

# 设置NIfTI精度阈值
nib.Nifti1Header.quaternion_threshold = -1e-06


class WholeBrainDatasetPreprocessor:
    """
    全脑分割数据集预处理器
    
    功能:
    1. DICOM序列转NIfTI格式
    2. 图像预处理（重采样、归一化、裁剪）
    3. 标签处理（重采样、标签映射）
    4. 数据集划分（训练/验证/测试）
    5. 生成JSON数据列表
    """
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 target_spacing: tuple = (1.0, 1.0, 1.0),
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        初始化预处理器
        
        Args:
            source_dir: 原始数据根目录
            output_dir: 处理后数据输出目录
            target_spacing: 目标体素间距(mm)
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_spacing = target_spacing
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        

        # 创建输出目录结构
        self._create_output_dirs()
        
    def _create_output_dirs(self):
        """创建输出目录结构"""
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)
    
    
    def process_case(self, case_id: str):
        """
        处理单个case
        
        Args:
            case_id: case标识符
            
        Returns:
            tuple: (image_path, label_path) 或 None
        """
        case_dir = os.path.join(self.source_dir, case_id)
        
        if not os.path.isdir(case_dir):
            print(f"跳过: {case_dir} 不是目录")
            return None
        
        dcm_dir = os.path.join(case_dir, 'dcm')
        # results_dir = os.path.join(case_dir, 'c_results')
        results_dir = case_dir
        
        # 检查必要的文件
        if not os.path.exists(results_dir):
            print(f"跳过: {case_id} 缺少标注结果")
            return None
        
        # 查找图像文件
        # 优先使用预处理好的图像
        image_candidates = [
            os.path.join(results_dir, 'original_bet.nii.gz')
        ]
        
        image_path = None
        for candidate in image_candidates:
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        # 如果没有预处理图像，从DICOM转换
        if image_path is None and os.path.exists(dcm_dir):
            temp_nii_path = os.path.join(results_dir, 'from_dicom.nii.gz')
            if self.dicom_to_nifti(dcm_dir, temp_nii_path):
                image_path = temp_nii_path
        
        if image_path is None:
            print(f"跳过: {case_id} 未找到图像文件")
            return None
        
        # 查找标签文件
        label_candidates = [
            os.path.join(case_dir, 'label_original_space.nii.gz')
        ]
        
        label_path = None
        for candidate in label_candidates:
            if os.path.exists(candidate):
                label_path = candidate
                break
        
        if label_path is None:
            print(f"跳过: {case_id} 未找到标签文件")
            return None
        
        return image_path, label_path
    
    def _load_manufacturer_cases(self) -> dict:
        """
        从Excel文件加载各机型的case列表
        
        Returns:
            dict: {manufacturer: set(case_ids)}
        """
        import pandas as pd
        
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
        manufacturer_files = {
            'Lowin': 'Lowin_.xlsx',
            'Philips': 'Philips_.xlsx',
            'SIEMENS': 'SIEMENS_.xlsx',
            'TOSHIBA': 'TOSHIBA_.xlsx',
            'UIH134': 'UIH134_.xlsx',
            'UIH164': 'UIH164_.xlsx'
        }
        
        manufacturer_cases = {}
        for manufacturer, filename in manufacturer_files.items():
            filepath = os.path.join(dataset_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_excel(filepath)
                    # 假设第一列或'case_id'列包含case ID
                    case_ids = set()
                    for col in df.columns:
                        case_ids.update(df[col].astype(str).dropna().tolist())
                    manufacturer_cases[manufacturer] = case_ids
                    print(f"加载 {manufacturer}: {len(case_ids)} cases")
                except Exception as e:
                    print(f"加载 {filename} 失败: {e}")
            else:
                print(f"文件不存在: {filepath}")
        
        return manufacturer_cases
    
    def _get_manufacturer(self, case_id: str, manufacturer_cases: dict) -> str:
        """
        获取case所属机型
        
        Args:
            case_id: case标识符
            manufacturer_cases: 机型到case集合的映射
            
        Returns:
            str: 机型名称，未知返回'Unknown'
        """
        for manufacturer, case_set in manufacturer_cases.items():
            if case_id in case_set:
                return manufacturer
        return 'Unknown'
    
    def _split_by_manufacturer(self, valid_cases: list, manufacturer_cases: dict) -> tuple:
        """
        按机型分别划分数据集，确保每个机型的数据按固定比例分配
        
        Args:
            valid_cases: 有效case列表
            manufacturer_cases: 机型到case集合的映射
            
        Returns:
            tuple: (train_cases, val_cases, test_cases)
        """
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        train_cases = []
        val_cases = []
        test_cases = []
        
        # 按机型分组
        cases_by_manufacturer = {}
        for case_id in valid_cases:
            manufacturer = self._get_manufacturer(case_id, manufacturer_cases)
            if manufacturer not in cases_by_manufacturer:
                cases_by_manufacturer[manufacturer] = []
            cases_by_manufacturer[manufacturer].append(case_id)
        
        # 对每个机型分别划分
        for manufacturer, cases in cases_by_manufacturer.items():
            if len(cases) == 0:
                continue
                
            # 打乱顺序
            np.random.shuffle(cases)
            
            n_total = len(cases)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            train_cases.extend(cases[:n_train])
            val_cases.extend(cases[n_train:n_train + n_val])
            test_cases.extend(cases[n_train + n_val:])
            
            print(f"  {manufacturer}: train={n_train}, val={n_val}, test={n_total - n_train - n_val}")
        
        return train_cases, val_cases, test_cases
    


    def preprocess_image(self, image_path: str, output_path: str, 
                         is_label: bool = False, crop_bounds: tuple = None):
        """
        图像预处理
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            is_label: 是否为标签图像
            crop_bounds: 裁剪边界 (min_coords, max_coords)，如果提供则使用此边界裁剪
        """
        import scipy.ndimage as ndimage
        
        # 加载图像
        nii = nib.load(image_path)
        data = nii.get_fdata()
        affine = nii.affine
        
        # 获取原始spacing
        original_spacing = nii.header.get_zooms()[:3]
        
        # 计算重采样比例
        zoom_factors = [o / t for o, t in zip(original_spacing, self.target_spacing)]
        
        if not np.allclose(zoom_factors, 1.0):
            # 重采样
            order = 0 if is_label else 3  # 标签使用最近邻插值
            data = ndimage.zoom(data, zoom_factors, order=order)
            
            # 正确更新affine矩阵
            # affine的方向矩阵部分表示每个体素在世界坐标系中的向量
            # 当体素数量增加(zoom > 1)时，每个体素变小，affine缩放因子应为 1/zoom
            scale_factors = [1.0 / z for z in zoom_factors]
            scale_matrix = np.diag(scale_factors + [1.0])
            affine = affine @ scale_matrix
        
        if not is_label:
            # 图像归一化（Z-score）
            data = self._normalize_intensity(data)
            
            # 裁剪前景（减少背景区域）
            data, affine, crop_bounds = self._crop_foreground(data, affine, return_bounds=True)
        else:
            # 标签：使用传入的裁剪边界
            if crop_bounds is not None:
                min_coords, max_coords = crop_bounds
                data = data[
                    min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2]
                ]
                # 更新affine原点
                new_affine = affine.copy()
                new_origin = affine @ np.array([min_coords[0], min_coords[1], min_coords[2], 1.0])
                new_affine[:3, 3] = new_origin[:3]
                affine = new_affine
        
        # 创建新的header，确保正确的spacing
        header = nii.header.copy()
        # 更新header中的pixdim信息
        header['pixdim'][1:4] = self.target_spacing
        
        # 保存处理后的图像
        nii_out = nib.Nifti1Image(data.astype(np.float32 if not is_label else np.int32), affine, header)
        nib.save(nii_out, output_path)
        
        return crop_bounds
    
    def _normalize_intensity(self, data: np.ndarray):
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
    
    def _crop_foreground(self, data: np.ndarray, affine: np.ndarray, return_bounds: bool = False):
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
    

    def dicom_to_nifti(self, dicom_dir: str, output_path: str):
        """
        将DICOM序列转换为NIfTI格式
        
        Args:
            dicom_dir: DICOM文件目录
            output_path: 输出NIfTI文件路径
        """
        try:
            import SimpleITK as sitk
            
            # 读取DICOM序列
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
            
            if not dicom_files:
                print(f"警告: 未找到DICOM序列: {dicom_dir}")
                return False
            
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            # 保存为NIfTI
            sitk.WriteImage(image, output_path)
            return True
            
        except ImportError:
            print("SimpleITK未安装，使用pydicom读取...")
            return self._dicom_to_nifti_pydicom(dicom_dir, output_path)
    
    def _dicom_to_nifti_pydicom(self, dicom_dir: str, output_path: str):
        """
        使用pydicom将DICOM序列转换为NIfTI格式（备用方法）
        """
        import pydicom
        from pydicom.dataset import Dataset
        
        # 获取所有DICOM文件
        dcm_files = sorted(glob(os.path.join(dicom_dir, '*.dcm')))
        
        if not dcm_files:
            print(f"警告: 未找到DICOM文件: {dicom_dir}")
            return False
        
        # 读取DICOM序列
        slices = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f)
                if hasattr(ds, 'PixelData'):
                    slices.append(ds)
            except Exception as e:
                print(f"跳过文件 {f}: {e}")
                continue
        
        if not slices:
            return False
        
        # 按InstanceNumber排序
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # 提取像素数据
        pixel_arrays = []
        for s in slices:
            pixel_data = s.pixel_array.astype(np.float32)
            # 应用DICOM窗口设置
            if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
                pixel_data = pixel_data * s.RescaleSlope + s.RescaleIntercept
            pixel_arrays.append(pixel_data)
        
        # 堆叠为3D数组
        volume = np.stack(pixel_arrays, axis=-1)
        
        # 获取空间信息
        pixel_spacing = slices[0].PixelSpacing
        slice_spacing = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0
        affine = np.eye(4)
        affine[0, 0] = float(pixel_spacing[0])
        affine[1, 1] = float(pixel_spacing[1])
        affine[2, 2] = slice_spacing
        
        # 保存为NIfTI
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, output_path)
        return True
    
    def run(self):
        """
        执行完整的预处理流程
        """
        print("=" * 60)
        print("全脑分割数据集预处理")
        print("=" * 60)
        
        # 获取所有case
        all_cases = []
        for item in os.listdir(self.source_dir):
            item_path = os.path.join(self.source_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 排除已经划分好的目录
                if item not in ['train', 'val', 'test']:
                    all_cases.append(item)
        
        print(f"发现 {len(all_cases)} 个case")
        
        # 过滤有效的case
        #根据wholebrainSeg/datasetn文件目录下机型对应的Excel文件获取数据集划分信息，Lowin_.xlsx代表朗润机型，Philips_.xlsx代表Philips机型 ， 
        # SIEMENS_.xlsx代表Siemens机型 ，TOSHIBA_.xlsx代表东芝机型，UIH134_.xlsx代表UIH134机型，UIH164_.xlsx代表UIH164机型
        #设置机型对应valid_cases
        valid_cases = []
        case_files = {}
        
        # 加载机型信息
        manufacturer_cases = self._load_manufacturer_cases()
        
        for case_id in tqdm(all_cases, desc="检查数据"):
            result = self.process_case(case_id)
            if result:
                valid_cases.append(case_id)
                case_files[case_id] = result
        
        print(f"有效case: {len(valid_cases)}")
        
        if len(valid_cases) == 0:
            print("没有有效的case，退出")
            return
        
        # 按机型划分数据集，确保每个机型的数据按固定比例分配
        train_cases, val_cases, test_cases = self._split_by_manufacturer(
            valid_cases, manufacturer_cases
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_cases)}")
        print(f"  验证集: {len(val_cases)}")
        print(f"  测试集: {len(test_cases)}")
        
        # 打印各机型分布
        print("\n各机型数据分布:")
        for split_name, cases in [('train', train_cases), ('val', val_cases), ('test', test_cases)]:
            manufacturer_counts = {}
            for case_id in cases:
                manufacturer = self._get_manufacturer(case_id, manufacturer_cases)
                manufacturer_counts[manufacturer] = manufacturer_counts.get(manufacturer, 0) + 1
            print(f"  {split_name}: {manufacturer_counts}")
        
        # 处理并保存数据
        splits = {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }
        
        for split_name, cases in splits.items():
            print(f"\n处理 {split_name} 集 ({len(cases)} cases)...")
            
            for case_id in tqdm(cases, desc=split_name):
                image_path, label_path = case_files[case_id]
                
                # 输出路径
                out_image = os.path.join(self.output_dir, split_name, 'images', f'{case_id}.nii.gz')
                out_label = os.path.join(self.output_dir, split_name, 'labels', f'{case_id}_seg.nii.gz')
                
                try:
                    # 预处理图像，获取裁剪边界
                    crop_bounds = self.preprocess_image(image_path, out_image, is_label=False)
                    
                    # 预处理标签，使用相同的裁剪边界
                    self.preprocess_image(label_path, out_label, is_label=True, crop_bounds=crop_bounds)
                    
                except Exception as e:
                    print(f"处理失败 {case_id}: {e}")
                    continue
        
        # 生成JSON数据列表
        self._create_json_files(splits)
        
        print("\n" + "=" * 60)
        print("预处理完成!")
        print("=" * 60)
    
    def _create_json_files(self, splits: dict):
        """
        生成JSON数据列表文件
        
        Args:
            splits: 数据集划分字典
        """
        json_dir = os.path.join(self.output_dir, 'json')
        os.makedirs(json_dir, exist_ok=True)
        
        # 生成fold0.json（单折训练）
        datadict = {
            'training': [],
            'validation': []
        }
        
        for split_name in ['train', 'val']:
            key = 'training' if split_name == 'train' else 'validation'
            split_dir = split_name
            
            images_dir = os.path.join(self.output_dir, split_dir, 'images')
            if os.path.exists(images_dir):
                for f in os.listdir(images_dir):
                    if f.endswith('.nii.gz'):
                        image_rel = f"{split_dir}/images/{f}"
                        label_rel = f"{split_dir}/labels/{f.replace('.nii.gz', '_seg.nii.gz')}"
                        
                        datadict[key].append({
                            'image': image_rel,
                            'label': label_rel
                        })
        
        json_path = os.path.join(json_dir, 'fold0.json')
        with open(json_path, 'w') as f:
            json.dump(datadict, f, indent=4)
        
        print(f"已生成: {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全脑分割数据集预处理')
    parser.add_argument('--source_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/all_data',
                        help='原始数据根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset',
                        help='处理后数据输出目录')
    parser.add_argument('--target_spacing', type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        help='目标体素间距(mm)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = WholeBrainDatasetPreprocessor(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_spacing=tuple(args.target_spacing),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # 执行预处理
    preprocessor.run()


if __name__ == '__main__':
    main()

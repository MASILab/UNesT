"""
NIfTI操作工具 - 标签空间映射

功能：
1. 将原始DICOM影像转换为NIfTI格式
2. 计算原始DICOM空间与预处理影像空间之间的映射关系
3. 将标签从预处理空间映射回原始影像空间

数据结构：
    /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164/
    ├── 76384925062202/                  # 患者编号
    │   ├── dcm/                         # 原始DICOM影像
    │   ├── original_from_dicom.nii.gz   # DICOM转换后的NIfTI (输出)
    │   └── c_results/
    │       ├── brain_preproc_img.nii.gz # 预处理后的影像（去颅骨，mini空间）
    │       ├── cleanup_labelmap96_src.nii.gz # 96空间下的标签结果
    │       └── label_original_space.nii.gz  # 映射到原始空间的标签 (输出)

处理流程：
    1. DICOM -> NIfTI转换
    2. 计算原始空间与预处理空间的映射矩阵
    3. 将标签重采样到原始空间

依赖：
    pip install nibabel SimpleITK numpy
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("警告: SimpleITK未安装，DICOM转换功能不可用")
    print("安装方法: pip install SimpleITK")


class NiftiOperator:
    """NIfTI空间映射操作器"""
    
    def __init__(self, source_dir: str):
        """
        初始化
        
        Args:
            source_dir: 数据根目录，如 /path/to/UIH164
        """
        self.source_dir = source_dir
    
    def dicom_to_nifti(self, dicom_dir: str, output_path: str) -> bool:
        """
        将DICOM序列转换为NIfTI格式
        
        Args:
            dicom_dir: DICOM文件目录
            output_path: 输出NIfTI文件路径
            
        Returns:
            bool: 是否成功
        """
        if not HAS_SITK:
            print("错误: 需要安装SimpleITK")
            return False
        
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
            
            if not dicom_files:
                print(f"错误: 未在 {dicom_dir} 找到DICOM序列")
                return False
            
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            sitk.WriteImage(image, output_path)
            print(f"  DICOM -> NIfTI: {output_path}")
            return True
            
        except Exception as e:
            print(f"  DICOM转换失败: {e}")
            return False
    
    def compute_affine_transform(self, source_img: nib.Nifti1Image, 
                                  target_img: nib.Nifti1Image) -> np.ndarray:
        """
        计算从source空间到target空间的仿射变换矩阵
        
        Args:
            source_img: 源图像（如原始DICOM转后的图像）
            target_img: 目标图像（如预处理后的brain_preproc_img）
            
        Returns:
            np.ndarray: 4x4仿射变换矩阵，用于将target空间的点映射到source空间
        """
        # source_affine: 原始空间的affine矩阵
        # target_affine: 预处理空间的affine矩阵
        # 
        # 要将target空间的坐标映射到source空间:
        # x_source = source_affine @ inv(target_affine) @ x_target
        # 
        # 即: transform = source_affine @ inv(target_affine)
        
        source_affine = source_img.affine
        target_affine = target_img.affine
        
        # 计算映射矩阵: target -> source
        transform = source_affine @ np.linalg.inv(target_affine)
        
        return transform
    
    def resample_label_to_original_space(self, 
                                          label_path: str, 
                                          original_img_path: str,
                                          preproc_img_path: str,
                                          output_path: str,
                                          interpolation: str = 'nearest') -> bool:
        """
        将标签从预处理空间重采样到原始影像空间
        
        通过配准计算brain_preproc_img和original_from_dicom之间的变换，
        然后用该变换将标签映射到原始空间
        
        Args:
            label_path: 预处理空间的标签文件路径
            original_img_path: 原始影像路径（DICOM转换后的或原始的）
            preproc_img_path: 预处理后的影像路径（用于计算映射关系）
            output_path: 输出标签路径
            interpolation: 插值方式 ('nearest', 'linear')
            
        Returns:
            bool: 是否成功
        """
        try:
            from scipy.ndimage import map_coordinates
            
            # 加载图像
            original_img = nib.load(original_img_path)
            preproc_img = nib.load(preproc_img_path)
            label_img = nib.load(label_path)
            
            # 获取数据
            label_data = label_img.get_fdata()
            original_shape = original_img.shape[:3]
            
            # 方法：使用配准计算变换矩阵
            # 将 brain_preproc_img 配准到 original_from_dicom
            # 得到 original -> preproc 的变换
            print("  正在计算空间变换（刚性配准）...")
            
            if HAS_SITK:
                # 使用SimpleITK进行刚性配准
                registration_transform = self._register_images(
                    original_img_path, preproc_img_path, use_affine=False
                )
                if registration_transform is not None:
                    # 配准返回的是 preproc -> original 的变换
                    # 我们需要 original_voxel -> label_voxel 的变换
                    # 变换链: original_voxel -> world -> preproc_voxel -> label_voxel
                    # label_voxel = inv(label_affine) @ (preproc -> world) @ orig_affine @ orig_voxel
                    
                    # 但配准给的变换是在物理空间中的，需要转换为voxel空间
                    # 简化：直接使用配准得到的变换矩阵
                    
                    # 配准变换: preproc 物理坐标 -> original 物理坐标
                    # 我们需要: original voxel -> label voxel
                    
                    # orig_voxel -> world_orig (orig_affine)
                    # world_orig -> world_preproc (inv(registration_transform)) 
                    # world_preproc -> label_voxel (inv(label_affine))
                    
                    # 但registration_transform可能是 preproc -> original
                    # 需要测试方向
                    
                    voxel_transform = registration_transform
                else:
                    # 配准失败，使用affine矩阵
                    print("  配准失败，使用affine矩阵计算...")
                    orig_affine = original_img.affine
                    preproc_affine = preproc_img.affine
                    voxel_transform = np.linalg.inv(preproc_affine) @ orig_affine
            else:
                # 没有SimpleITK，使用affine矩阵
                orig_affine = original_img.affine
                preproc_affine = preproc_img.affine
                voxel_transform = np.linalg.inv(preproc_affine) @ orig_affine
            
            # 生成原始图像空间的所有体素坐标
            coords = np.mgrid[0:original_shape[0], 
                              0:original_shape[1], 
                              0:original_shape[2]].reshape(3, -1).astype(np.float64)
            
            # 添加齐次坐标
            coords_h = np.vstack([coords, np.ones((1, coords.shape[1]))])
            
            # 变换到标签空间的体素坐标
            label_coords = voxel_transform @ coords_h
            label_coords = label_coords[:3]  # 只取前3行
            
            # 使用map_coordinates进行插值
            order = 0 if interpolation == 'nearest' else 1
            resampled = map_coordinates(label_data, label_coords, 
                                        order=order, 
                                        mode='constant', 
                                        cval=0.0)
            resampled = resampled.reshape(original_shape)
            
            # 保存结果
            orig_affine = original_img.affine
            nib.Nifti1Image(resampled.astype(np.int32), orig_affine).to_filename(output_path)
            
            print(f"  标签映射完成: {output_path}")
            return True
            
        except Exception as e:
            print(f"  标签映射失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _register_images(self, fixed_path: str, moving_path: str, 
                         use_affine: bool = False) -> np.ndarray:
        """
        使用SimpleITK进行图像配准
        
        Args:
            fixed_path: 固定图像路径（目标空间，这里是original_from_dicom）
            moving_path: 移动图像路径（源空间，这里是brain_preproc_img）
            use_affine: 是否使用仿射变换，False则使用刚性变换
            
        Returns:
            np.ndarray: 4x4变换矩阵，将moving空间的点映射到fixed空间
        """
        try:
            # 读取图像
            fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
            
            # 初始化配准方法
            registration_method = sitk.ImageRegistrationMethod()
            
            # 设置相似性度量（互信息）
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.1)
            
            # 设置插值器
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # 设置优化器
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # 初始变换：基于图像中心对齐
            initial_transform = sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.Euler3DTransform() if not use_affine else sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # 多分辨率策略
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # 执行配准
            final_transform = registration_method.Execute(fixed, moving)
            
            # 获取变换参数
            params = final_transform.GetParameters()
            
            if not use_affine:
                # 刚性变换：3个旋转 + 3个平移
                # Euler3DTransform: rx, ry, rz, tx, ty, tz
                rx, ry, rz, tx, ty, tz = params
                
                # 构建旋转矩阵
                cx, sx = np.cos(rx), np.sin(rx)
                cy, sy = np.cos(ry), np.sin(ry)
                cz, sz = np.cos(rz), np.sin(rz)
                
                # Rx * Ry * Rz
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                R = Rx @ Ry @ Rz
                
                # 构建4x4矩阵
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R
                transform_matrix[:3, 3] = [tx, ty, tz]
            else:
                # 仿射变换：12个参数
                transform_matrix = np.eye(4)
                transform_matrix[:3, :] = np.array(params).reshape(3, 4)
            
            print(f"  配准完成，变换矩阵:")
            print(f"    平移: [{transform_matrix[0,3]:.2f}, {transform_matrix[1,3]:.2f}, {transform_matrix[2,3]:.2f}]")
            
            # 将SimpleITK的变换转换为nibabel/RAS坐标系
            # SimpleITK使用LPS坐标系，nibabel使用RAS
            ras_to_lps = np.diag([-1, -1, 1, 1])
            transform_ras = ras_to_lps @ transform_matrix @ ras_to_lps
            
            # 现在需要转换为voxel空间的变换
            # transform_ras: moving 物理坐标 -> fixed 物理坐标
            # 我们需要: fixed_voxel -> moving_voxel
            
            fixed_img = nib.load(fixed_path)
            moving_img = nib.load(moving_path)
            fixed_affine = fixed_img.affine
            moving_affine = moving_img.affine
            
            # fixed_voxel -> world_fixed (fixed_affine)
            # world_fixed -> world_moving (我们需要的是这个，但transform_ras是反的)
            # world_moving -> moving_voxel (inv(moving_affine))
            
            # transform_ras 是 moving -> fixed
            # 所以 inv(transform_ras) 是 fixed -> moving
            # 在物理空间中: world_moving = inv(transform_ras) @ world_fixed
            
            # voxel变换:
            # moving_voxel = inv(moving_affine) @ world_moving
            #             = inv(moving_affine) @ inv(transform_ras) @ world_fixed
            #             = inv(moving_affine) @ inv(transform_ras) @ fixed_affine @ fixed_voxel
            
            voxel_transform = np.linalg.inv(moving_affine) @ np.linalg.inv(transform_ras) @ fixed_affine
            
            return voxel_transform
            
        except Exception as e:
            print(f"  配准失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _resample_with_sitk(self, label_path: str, ref_img_path: str,
                            transform: np.ndarray, output_path: str,
                            interpolation: str = 'nearest'):
        """
        使用SimpleITK进行重采样 (已弃用，保留备用)
        """
        pass
    
    def process_case(self, case_id: str) -> dict:
        """
        处理单个患者case
        
        Args:
            case_id: 患者编号，如 '76384925062202'
            
        Returns:
            dict: 处理结果
        """
        case_dir = os.path.join(self.source_dir, case_id)
        result = {
            'case_id': case_id,
            'success': False,
            'message': ''
        }
        
        # 检查必要目录
        dcm_dir = os.path.join(case_dir, 'dcm')
        c_results_dir = os.path.join(case_dir, 'c_results')
        
        if not os.path.exists(dcm_dir):
            result['message'] = 'DICOM目录不存在'
            return result
        
        if not os.path.exists(c_results_dir):
            result['message'] = 'c_results目录不存在'
            return result
        
        # 文件路径
        original_nii_path = os.path.join(case_dir, 'original_from_dicom.nii.gz')
        preproc_img_path = os.path.join(c_results_dir, 'brain_preproc_img.nii.gz')
        label_path = os.path.join(c_results_dir, 'cleanup_labelmap96_src.nii.gz')
        # 输出到患者目录的c_results下
        output_label_path = os.path.join(case_dir, 'label_original_space.nii.gz')
        
        # 检查预处理文件是否存在
        if not os.path.exists(preproc_img_path):
            result['message'] = f'预处理影像不存在: {preproc_img_path}'
            return result
        
        if not os.path.exists(label_path):
            result['message'] = f'标签文件不存在: {label_path}'
            return result
        
        print(f"\n处理 {case_id}...")
        
        # 步骤1: DICOM -> NIfTI
        if not self.dicom_to_nifti(dcm_dir, original_nii_path):
            result['message'] = 'DICOM转换失败'
            return result
        
        # 步骤2: 将标签映射到原始空间
        if not self.resample_label_to_original_space(
            label_path, original_nii_path, preproc_img_path, output_label_path
        ):
            result['message'] = '标签映射失败'
            return result
        
        # 验证结果
        try:
            original_img = nib.load(original_nii_path)
            mapped_label = nib.load(output_label_path)
            
            print(f"  原始影像形状: {original_img.shape}")
            print(f"  映射后标签形状: {mapped_label.shape}")
            print(f"  标签值范围: {mapped_label.get_fdata().min():.0f} - {mapped_label.get_fdata().max():.0f}")
            
            result['success'] = True
            result['message'] = '处理成功'
            result['original_nii'] = original_nii_path
            result['mapped_label'] = output_label_path
            
        except Exception as e:
            result['message'] = f'验证失败: {e}'
        
        return result
    
    def run(self, case_ids: list = None):
        """
        执行批量处理
        
        Args:
            case_ids: 指定处理的case列表，None则处理所有
        """
        print("=" * 70)
        print("NIfTI空间映射工具")
        print("=" * 70)
        print(f"源目录: {self.source_dir}")
        
        # 获取所有case
        if case_ids is None:
            case_ids = []
            for item in os.listdir(self.source_dir):
                item_path = os.path.join(self.source_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否有dcm和c_results目录
                    if os.path.exists(os.path.join(item_path, 'dcm')) or \
                       os.path.exists(os.path.join(item_path, 'c_results')):
                        case_ids.append(item)
        
        print(f"发现 {len(case_ids)} 个case")
        
        if len(case_ids) == 0:
            print("没有找到可处理的case")
            return
        
        # 处理每个case
        results = []
        for case_id in tqdm(case_ids, desc="处理进度"):
            result = self.process_case(case_id)
            results.append(result)
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        print("\n" + "=" * 70)
        print("处理完成!")
        print("=" * 70)
        print(f"成功: {success_count}/{len(results)}")
        
        # 显示失败原因
        failed = [r for r in results if not r['success']]
        if failed:
            print("\n失败列表:")
            for r in failed:
                print(f"  {r['case_id']}: {r['message']}")


def main():
    parser = argparse.ArgumentParser(
        description='NIfTI空间映射工具 - 将标签从预处理空间映射到原始DICOM空间'
    )
    parser.add_argument('--source_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164',
                        help='数据根目录')
    parser.add_argument('--case_ids', type=str, nargs='+',
                        default=None,
                        help='指定处理的case ID列表')
    
    args = parser.parse_args()
    
    operator = NiftiOperator(source_dir=args.source_dir)
    operator.run(case_ids=args.case_ids)


if __name__ == '__main__':
    main()

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
    2. HD-BET去颅骨：original_from_dicom -> original_bet (提高配准精度)
    3. ANTs刚性配准（以MNI152为中间空间）：
       - brain_preproc_img -> MNI152_T1_1mm_brain.nii.gz
       - original_bet -> MNI152_T1_1mm_brain.nii.gz
    4. 组合2次刚性变换将标签重采样到原始空间

依赖：
    pip install nibabel SimpleITK numpy antspyx
    HD-BET: https://github.com/MIC-DKFZ/hd-bet
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import tempfile
import shutil

# 添加commen_utils到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'commen_utils'))

try:
    from HD_BET.run import run_hd_bet
    import HD_BET
    HAS_HD_BET = True
except ImportError:
    HAS_HD_BET = False
    print("警告: HD-BET未安装，去颅骨功能不可用")
    print("安装方法: pip install HD-BET 或将commen_utils添加到PYTHONPATH")

try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False
    print("警告: antspyx未安装，配准功能不可用")
    print("安装方法: pip install antspyx")

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("警告: SimpleITK未安装，DICOM转换功能不可用")
    print("安装方法: pip install SimpleITK")


# MNI152模板路径（默认位置，可通过参数覆盖）
DEFAULT_MNI152_PATH = '/home/tenoke4090/B_WorkPath/mrqs/UNesT/commen_utils/minispace/model/mni_space/MNI152_T1_1mm_brain.nii.gz'


class NiftiOperator:
    """NIfTI空间映射操作器 - 使用ANTs进行刚性配准（MNI152中间空间）"""
    
    def __init__(self, source_dir: str, mni152_path: str = None):
        """
        初始化
        
        Args:
            source_dir: 数据根目录，如 /path/to/UIH164
            mni152_path: MNI152模板路径，默认使用内置路径
        """
        self.source_dir = source_dir
        self.mni152_path = mni152_path or DEFAULT_MNI152_PATH
        self.temp_dirs = []  # 跟踪临时目录
    
    def __del__(self):
        """清理临时目录"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
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
    
    def _ants_rigid_registration(self, fixed_path: str, moving_path: str, 
                                   temp_dir: str, prefix: str = 'rigid_reg_') -> dict:
        """
        使用ANTs进行刚性配准
        
        配准方向：moving -> fixed
        
        Args:
            fixed_path: 固定图像路径（目标空间）
            moving_path: 移动图像路径（源空间）
            temp_dir: 临时文件目录
            prefix: 输出文件前缀
            
        Returns:
            dict: ANTs配准结果，包含：
                - fwdtransforms: 变换文件列表（moving -> fixed）
                - invtransforms: 逆变换文件列表（fixed -> moving）
                - warpedmovout: 变换后的移动图像
        """
        if not HAS_ANTS:
            raise RuntimeError("ANTs未安装，无法进行配准")
        
        # 使用ANTs读取图像
        fixed_img = ants.image_read(fixed_path)
        moving_img = ants.image_read(moving_path)
        
        # 输出前缀
        outprefix = os.path.join(temp_dir, prefix)
        
        print("    正在进行ANTs刚性配准...")
        print(f"    Fixed (目标): {os.path.basename(fixed_path)}")
        print(f"    Moving (源): {os.path.basename(moving_path)}")
        
        # 执行刚性配准
        registration_result = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='Rigid',  # 刚性变换（平移+旋转）
            outprefix=outprefix,
            verbose=False
        )
        
        return registration_result
    
    def _ants_resample_label(self, label_path: str, fixed_path: str,
                              transforms: list, output_path: str,
                              interpolation: str = 'nearest') -> bool:
        """
        使用ANTs将标签重采样到固定图像空间
        
        Args:
            label_path: 标签文件路径（在moving空间）
            fixed_path: 参考图像路径（目标空间）
            transforms: 变换文件列表（from ants.registration）
            output_path: 输出路径
            interpolation: 插值方式 ('nearest', 'linear', 'genericLabel')
            
        Returns:
            bool: 是否成功
        """
        try:
            # 读取标签和参考图像
            label_img = ants.image_read(label_path)
            fixed_img = ants.image_read(fixed_path)
            
            # 对于标签图像，使用最近邻插值或genericLabel
            if interpolation == 'nearest':
                interp_method = 'nearestNeighbor'
            elif interpolation == 'genericLabel':
                interp_method = 'genericLabel'
            else:
                interp_method = 'linear'
            
            print(f"    正在重采样标签到原始空间...")
            
            # 应用变换
            resampled_label = ants.apply_transforms(
                fixed=fixed_img,
                moving=label_img,
                transformlist=transforms,
                interpolator=interp_method
            )
            
            # 保存结果
            ants.image_write(resampled_label, output_path)
            print(f"    标签重采样完成: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"    标签重采样失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def run_bet(self, input_img_path: str, output_bet_path: str, 
                device: int = 0, keep_mask: bool = True) -> bool:
        """
        使用HD-BET进行去颅骨处理
        
        Args:
            input_img_path: 输入图像路径
            output_bet_path: 输出去颅骨图像路径
            device: GPU设备ID，-1表示CPU
            keep_mask: 是否保存脑mask
            
        Returns:
            bool: 是否成功
        """
        if not HAS_HD_BET:
            print("错误: HD-BET未安装")
            return False
        
        try:
            print(f"    HD-BET去颅骨: {os.path.basename(input_img_path)}")
            
            run_hd_bet(
                mri_fnames=input_img_path,
                output_fnames=output_bet_path,
                mode="accurate",
                config_file=os.path.join(HD_BET.__path__[0], "config.py"),
                device=device,
                postprocess=False,
                do_tta=False,
                keep_mask=keep_mask,
                overwrite=True,
                bet=True,  # 输出去颅骨后的图像
            )
            
            print(f"    去颅骨完成: {output_bet_path}")
            return True
            
        except Exception as e:
            print(f"    去颅骨失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def resample_label_to_original_space(self, 
                                          label_path: str, 
                                          original_img_path: str,
                                          preproc_img_path: str,
                                          output_path: str,
                                          interpolation: str = 'genericLabel') -> bool:
        """
        将标签从预处理空间重采样到原始影像空间
        
        处理流程（以MNI152为中间空间，使用HD-BET去颅骨提高配准精度）：
        1. HD-BET去颅骨: original_from_dicom -> original_bet
        2. 配准1: brain_preproc_img -> MNI152
           得到 fwdtransforms1: preproc -> MNI
        3. 配准2: original_bet -> MNI152
           得到 invtransforms2: MNI -> original (逆变换)
           注：original_bet与original_from_dicom在同一空间，变换可直接应用
        4. 组合变换: label -> (fwdtransforms1) -> MNI -> (invtransforms2) -> original
        
        Args:
            label_path: 预处理空间的标签文件路径
            original_img_path: 原始影像路径（DICOM转换后的）
            preproc_img_path: 预处理后的影像路径
            output_path: 输出标签路径
            interpolation: 插值方式 ('nearest', 'linear', 'genericLabel')
            
        Returns:
            bool: 是否成功
        """
        if not HAS_ANTS:
            print("错误: 需要安装ANTs (antspyx)")
            return False
        
        # 检查MNI152模板是否存在
        if not os.path.exists(self.mni152_path):
            print(f"错误: MNI152模板不存在: {self.mni152_path}")
            return False
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='ants_reg_mni_')
        self.temp_dirs.append(temp_dir)
        
        try:
            print("  正在进行空间映射（MNI152中间空间 + HD-BET去颅骨）...")
            print(f"  MNI152模板: {self.mni152_path}")
            
            # =====================================================
            # 步骤1: HD-BET去颅骨 - 对original_from_dicom进行去颅骨
            # =====================================================
            original_bet_path = os.path.join(temp_dir, 'original_bet.nii.gz')
            
            if HAS_HD_BET:
                print("\n  [去颅骨] original_from_dicom -> original_bet")
                if not self.run_bet(original_img_path, original_bet_path, keep_mask=False):
                    print("  警告: 去颅骨失败，使用原始图像进行配准")
                    original_bet_path = original_img_path
            else:
                print("  警告: HD-BET不可用，使用原始图像进行配准")
                original_bet_path = original_img_path
            
            # =====================================================
            # 步骤2: 配准1 - brain_preproc_img -> MNI152
            # =====================================================
            print("\n  [配准1] brain_preproc_img -> MNI152")
            reg1_result = self._ants_rigid_registration(
                fixed_path=self.mni152_path,
                moving_path=preproc_img_path,
                temp_dir=temp_dir,
                prefix='reg1_preproc_to_mni_'
            )
            
            # fwdtransforms1: preproc -> MNI
            fwdtransforms1 = reg1_result['fwdtransforms']
            if not fwdtransforms1:
                print("  错误: 配准1未产生变换文件")
                return False
            print(f"    获取变换: preproc -> MNI ({len(fwdtransforms1)} 个文件)")
            
            # =====================================================
            # 步骤3: 配准2 - original_bet -> MNI152
            # 使用去颅骨后的图像进行配准，提高配准精度
            # =====================================================
            print("\n  [配准2] original_bet -> MNI152")
            reg2_result = self._ants_rigid_registration(
                fixed_path=self.mni152_path,
                moving_path=original_bet_path,
                temp_dir=temp_dir,
                prefix='reg2_original_to_mni_'
            )
            
            # invtransforms2: MNI -> original (逆变换)
            # 注：original_bet与original_from_dicom在同一空间
            # 所以这个变换可以直接用于将标签映射到原始空间
            invtransforms2 = reg2_result['invtransforms']
            if not invtransforms2:
                print("  错误: 配准2未产生逆变换文件")
                return False
            print(f"    获取变换: MNI -> original ({len(invtransforms2)} 个文件)")
            
            # =====================================================
            # 步骤4: 组合变换 - preproc -> MNI -> original
            # 变换列表顺序: 先应用 fwdtransforms1, 再应用 invtransforms2
            # =====================================================
            print("\n  组合变换链: label -> MNI152 -> original_space")
            combined_transforms = fwdtransforms1 + invtransforms2
            print(f"    总变换文件数: {len(combined_transforms)}")
            
            # =====================================================
            # 步骤5: 使用组合变换重采样标签
            # =====================================================
            success = self._ants_resample_label(
                label_path=label_path,
                fixed_path=original_img_path,
                transforms=combined_transforms,
                output_path=output_path,
                interpolation=interpolation
            )
            
            if success:
                print(f"\n  标签映射完成: {output_path}")
            
            return success
            
        except Exception as e:
            print(f"  标签映射失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                if temp_dir in self.temp_dirs:
                    self.temp_dirs.remove(temp_dir)
    
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
        description='NIfTI空间映射工具 - 将标签从预处理空间映射到原始DICOM空间（MNI152中间空间）'
    )
    parser.add_argument('--source_dir', type=str,
                        default='/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164',
                        help='数据根目录')
    parser.add_argument('--mni152_path', type=str,
                        default=DEFAULT_MNI152_PATH,
                        help='MNI152模板路径')
    parser.add_argument('--case_ids', type=str, nargs='+',
                        default=None,
                        help='指定处理的case ID列表')
    
    args = parser.parse_args()
    
    operator = NiftiOperator(source_dir=args.source_dir, mni152_path=args.mni152_path)
    operator.run(case_ids=args.case_ids)


if __name__ == '__main__':
    main()

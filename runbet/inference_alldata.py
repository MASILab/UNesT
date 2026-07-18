"""
去颅骨（Skull Stripping）推理脚本

使用HD-BET对MRI图像进行颅骨剥离处理。
 

依赖库:
    - torch: PyTorch深度学习框架
    - nibabel: NIfTI文件读写
    - HD-BET: 颅骨剥离工具

使用示例:
python inference_alldata.py  --data_dir "/home/tenoke4090/B_WorkPath/mrqs/tst_data/"  --results_dir "/home/tenoke4090/B_WorkPath/mrqs/tst_data" 

输出文件:
    - {原文件名}: 原始图像副本
    - strip_{原文件名}: 颅骨剥离后的图像
"""

import os
import sys
import gc
import time
import argparse

import torch
import nibabel as nib

# 禁止Python写入.pyc字节码文件
sys.dont_write_bytecode = True

# 添加commen_utils到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'commen_utils'))
try:
    from commen_utils.HD_BET.run import run_hd_bet
    import HD_BET
    HAS_HD_BET = True
except Exception as e:
    HAS_HD_BET = False
    print(f"警告: HD-BET导入失败，去颅骨功能不可用")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()
    print("安装方法: pip install HD-BET 或将commen_utils添加到PYTHONPATH")


# ==================== 去颅骨函数 ====================

def run_bet(input_img_path: str, output_bet_path: str,
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


# ==================== 命令行参数解析 ====================
time_start = time.time()  # 记录开始时间
parser = argparse.ArgumentParser(description='去颅骨推理脚本')
parser.add_argument('--data_dir', type=str, default='data',
                    help='测试图像目录路径')
parser.add_argument('--results_dir', type=str, default='result',
                    help='结果保存目录路径')
parser.add_argument('--device', type=int, default=0,
                    help='GPU设备ID (默认: 0)')
# 以下参数为兼容旧命令保留，去颅骨功能不使用
parser.add_argument('--model_path', type=str, default='model',
                    help='(已弃用) 模型目录路径，去颅骨不需要')
parser.add_argument('--overlap', type=float, default=0.5,
                    help='(已弃用) 滑动窗口重叠率，去颅骨不需要')

args = parser.parse_args()

# ==================== 设备配置 ====================
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# 创建结果目录
os.makedirs(args.results_dir, exist_ok=True)

# 获取所有测试文件（过滤掉已有的去颅骨结果文件strip_和非nii文件）
dataAll = [f for f in os.listdir(args.data_dir)
           if f.endswith(('.nii', '.nii.gz')) and not f.startswith('strip_')]

print(f"开始去颅骨处理，共 {len(dataAll)} 个文件")
print(f"设备: {device}")
print("=" * 50)

# ==================== 去颅骨处理循环 ====================
for idx, ele in enumerate(dataAll):
    print(f"\n处理 [{idx+1}/{len(dataAll)}]: {ele}")

    # ============ 步骤1: 保存原始图像副本 ============
    predata = nib.load(os.path.join(args.data_dir, ele))
    original_img_path = os.path.join(args.results_dir, ele)
    nib.Nifti1Image(predata.get_fdata(), predata.affine).to_filename(original_img_path)

    # ============ 步骤2: HD-BET去颅骨 ============
    print("  - 颅骨剥离中...")
    original_bet_path = os.path.join(args.results_dir, 'strip_' + ele)

    if HAS_HD_BET:
        print("\n  [去颅骨] original -> bet")
        print(f"    输出路径: {original_bet_path}")
        if not run_bet(original_img_path, original_bet_path, keep_mask=False):
            print("  警告: 去颅骨失败，使用原始图像")
            original_bet_path = original_img_path
    else:
        print("  警告: HD-BET不可用，跳过去颅骨")
        original_bet_path = original_img_path

time_end = time.time()
print("\n" + "=" * 50)
print(f"去颅骨处理完成，耗时: {time_end - time_start:.2f} 秒")

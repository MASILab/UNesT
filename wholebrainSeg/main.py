# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from turtle import circle
import nibabel as nb
from PIL import Image
import scipy.ndimage as ndimage
import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete,Activations,Compose
from tqdm import tqdm
from utils.data_utils import get_loader
import gc  # 垃圾回收模块
import datetime

from optimizers.lr_scheduler import WarmupCosineSchedule


import yaml

def count_parameters(model):
    """
    计算模型的可训练参数数量（以百万为单位）。
    
    Args:
        model: PyTorch模型实例
        
    Returns:
        float: 可训练参数数量（单位：百万）
    
    Example:
        >>> model = UNesT(in_channels=1, out_channels=133)
        >>> num_params = count_parameters(model)
        >>> print(f"模型参数量: {num_params:.2f}M")
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def Dice(x, y):
    """
    计算两个3D二值掩码之间的Dice系数（Dice Similarity Coefficient）。
    
    Dice系数用于衡量两个集合的相似度，常用于评估医学图像分割结果。
    计算公式: DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        x: numpy.ndarray, 第一个3D二值掩码，形状为(H, W, D)
        y: numpy.ndarray, 第二个3D二值掩码，形状为(H, W, D)
        
    Returns:
        float: Dice系数，取值范围[0, 1]
               - 1.0 表示完全重合
               - 0.0 表示无重叠或y为空
    
    Note:
        当y掩码全为0时，返回0.0以避免除零错误
    """
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def resample(img, target_size):
    """
    将3D图像重采样到目标尺寸（使用最近邻插值）。
    
    该函数使用scipy的zoom函数进行图像重采样，适用于分割标签图像，
    因为最近邻插值(order=0)不会产生标签值之间的插值。
    
    Args:
        img: numpy.ndarray, 输入的3D图像，形状为(imx, imy, imz)
        target_size: tuple, 目标尺寸(tx, ty, tz)
        
    Returns:
        numpy.ndarray: 重采样后的3D图像，形状为target_size
    
    Example:
        >>> img = np.random.rand(100, 100, 100)
        >>> resampled = resample(img, (50, 50, 50))
        >>> resampled.shape
        (50, 50, 50)
    """
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def main(cfig, device):
    """
    UNesT模型训练的主函数。
    
    该函数负责初始化模型、优化器、损失函数和学习率调度器，
    并执行完整的训练循环，包括定期验证和模型检查点保存。
    
    Args:
        cfig: dict, 配置字典，包含以下关键字段：
            - logdir: str, 日志和模型保存目录
            - data_dir: str, 数据根目录
            - jsondir: str, JSON数据列表目录
            - use_pretrained: str, 预训练模型路径（可选）
            - fold: int, 交叉验证折数
            - num_classes: int, 分割类别数（默认133）
            - model_type: str, 模型类型（'base'/'small'/'large'）
            - patch_size: int, Patch大小
            - depth: list, Transformer各层深度
            - num_heads: list, 各层的注意力头数
            - embed_dims: list, 各层的嵌入维度
            - num_steps: int, 总训练步数
            - lr: float, 学习率
            - decay: float, 权重衰减
            - batch_size: int, 批次大小
            - loss_type: str, 损失函数类型
            - eval_num: int, 验证间隔步数
            - opt: str, 优化器类型（'adam'/'adamw'/'sgd'）
            - lrdecay: bool, 是否使用学习率衰减
            - roi_x/y/z: int, 训练ROI尺寸
            - sw_batch_size: int, 滑动窗口推理批次大小 
        device: torch.device, 训练设备（CPU或CUDA）
    
    Returns:
        None
    
    Note:
        训练过程中会在logdir目录下保存：
        - model.pt: 验证集上表现最好的模型
        - model_final_epoch.pt: 最终训练完成的模型
    """
    def save_ckp(state, checkpoint_dir):
        """
        保存模型检查点。
        
        Args:
            state: dict, 包含以下键的字典：
                - global_step: int, 当前训练步数
                - state_dict: OrderedDict, 模型权重
                - optimizer: dict, 优化器状态
            checkpoint_dir: str, 检查点保存路径
        """
        torch.save(state, checkpoint_dir)

    def train(global_step,train_loader,dice_val_best, val_shape_dict):
        """
        执行单个epoch的训练循环。
        
        Args:
            global_step: int, 当前全局训练步数
            train_loader: DataLoader, 训练数据加载器
            dice_val_best: float, 历史最佳验证Dice分数
            val_shape_dict: dict, 验证集图像形状信息
            
        Returns:
            tuple: (global_step, dice_val_best)
                - global_step: 更新后的全局训练步数
                - dice_val_best: 更新后的最佳验证Dice分数
        
        Note:
            每40步计算并打印训练Dice分数
            每eval_num步执行验证并保存最佳模型
        """
        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",dynamic_ncols=True)
        
        # 混合精度训练
        use_amp = cfig.get('amp', False)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].to(device), batch["label"].to(device))
            
            # 使用混合精度前向传播
            with torch.amp.autocast('cuda', enabled=use_amp):
                logit_map = model(x)
                
                # 调试信息：检查维度和标签值范围
                if global_step == 0:
                    print(f'\n=== 调试信息 ===')
                    print(f'输入 x shape: {x.shape}')
                    print(f'标签 y shape: {y.shape}')
                    print(f'标签 y min: {y.min().item()}, max: {y.max().item()}')
                    print(f'模型输出 logit_map shape: {logit_map.shape}')
                    print(f'num_classes: {cfig["num_classes"]}')
                    print(f'混合精度训练: {use_amp}')
                    print(f'===============\n')

                try:
                    loss = loss_function(logit_map, y)
                except Exception as e:
                    print(f"损失函数计算错误: {e}")
                    print(f"尝试使用备用标签格式: y[:,0].long()")
                    loss = loss_function(logit_map, y[:,0].long())

            # training Dice 
            if global_step % 40 == 0:
                print(f'GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
                with torch.no_grad():
                    train_pred = torch.softmax(logit_map.float(), 1).detach().cpu().numpy()
                    train_pred = np.argmax(train_pred, axis = 1).astype(np.uint8)
                    train_label = y.detach().cpu().numpy()[:,0,:,:,:]
                    
                    dice_list_sub = []
                    for i in range(1, cfig['num_classes']):
                        organ_Dice = Dice(train_pred[0] == i, train_label[0] == i)
                        dice_list_sub.append(organ_Dice)
                    print('Train DSC: {} ,Current Time:{}'.format(np.mean(dice_list_sub),datetime.datetime.now()))
                    writer.add_scalar("train/DSC_sample", scalar_value=np.mean(dice_list_sub), global_step=global_step)
                    del train_pred, train_label, dice_list_sub

            # 使用混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            if cfig['lrdecay']:
                scheduler.step()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, cfig['num_steps'], loss.item()))
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)

            # 释放计算图和临时变量
            del x, y, logit_map, loss
            
            global_step += 1
            
            # 定期清理GPU缓存（增加频率以提高稳定性）
            if global_step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if global_step % cfig['eval_num'] == 0:
                # 验证前准备：刷新缓冲、清理缓存、同步CUDA
                sys.stdout.flush()
                torch.cuda.synchronize()  # 确保所有CUDA操作完成
                torch.cuda.empty_cache()
                gc.collect()
                print(f'\n===== 开始验证 (Step {global_step}) =====')
                
                epoch_iterator_val = tqdm(test_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True, leave=True)

                mean_list = validation(epoch_iterator_val, val_shape_dict)
                
                # 验证后清理缓存并切回训练模式
                torch.cuda.synchronize()  # 确保验证操作全部完成
                torch.cuda.empty_cache()
                gc.collect()
                model.train()
                print(f'===== 验证完成，继续训练 =====\n')

                writer.add_scalar("ValAvgDice/Dice_avg", scalar_value=np.mean(mean_list), global_step=global_step)
                for lbl_i in range(cfig['num_classes']-1):
                    writer.add_scalar("Validation/Dice_{}".format(lbl_i+1), scalar_value=mean_list[lbl_i], global_step=global_step)

                dice_val = np.mean(mean_list)
                if dice_val > dice_val_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model.pt')
                    dice_val_best = dice_val
                    print('Model Was Saved ! Current Best Dice: {},  Current Dice: {}'.format(dice_val_best, np.mean(mean_list)))
                else:
                    print('Model Was NOT Saved ! Current Best Dice: {} Current Dice: {}'.format(dice_val_best, dice_val))
                del mean_list, dice_val
        return global_step, dice_val_best


    def validation(epoch_iterator_val, val_shape_dict):
        """
        在验证集上执行模型验证。
        
        使用滑动窗口推理(sliding window inference)对验证图像进行预测，
        计算每个类别的Dice分数。
        
        Args:
            epoch_iterator_val: iterator, 验证数据迭代器
            val_shape_dict: dict, 验证集图像形状字典（用于记录）
            
        Returns:
            list: 每个类别（除背景外）的平均Dice分数，长度为num_classes-1
        
        Note:
            - 使用overlap=0.2的滑动窗口推理
            - 模型在CPU上进行推理以节省GPU显存
        """
        model.eval()
        metric_values = []
        roi_size = (cfig['roi_x'], cfig['roi_y'], cfig['roi_z'])
        sw_batch_size = cfig['sw_batch_size']
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                name = batch["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
                # # 使用GPU推理，更快；如果显存不足可改回 device=torch.device('cpu')
                # val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.5, device=device)
                # 使用CPU推理
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.5, device=torch.device('cpu'))
                val_outputs = torch.softmax(val_outputs, 1).detach().cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis = 1).astype(np.uint8)
                val_labels = val_labels.detach().cpu().numpy()[:,0,:,:,:]
                print(f'验证样本: {name}')

                dice_list_sub = []
                for i in range(1, cfig['num_classes']):
                    organ_Dice = Dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)

                dice_mean = np.mean(dice_list_sub)
                metric_values.append(dice_list_sub)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice_mean=%2.5f)" % (global_step, 5.0, dice_mean))

                # 每个验证样本处理完后清理内存，防止累积导致段错误
                del val_outputs, val_inputs, val_labels
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

            mean_list = np.mean(metric_values, axis=0)

        return mean_list

    # ==================== GPU加速配置 ====================
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法以加速训练
    cfig['n_gpu'] = torch.cuda.device_count()
    cfig['device'] = device

    print(torch.version.cuda)
    
    # ==================== 模型初始化 ====================
    # 根据配置选择不同规模的UNesT模型
    if cfig['model_type'] == 'base':
        from networks.unest import UNesT
    elif cfig['model_type'] == 'small':
        from networks.unest_small_patch_4 import UNesT
    elif cfig['model_type'] == 'large':
        from networks.unest_large_patch_4 import UNesT
    model = UNesT(in_channels=1,
                out_channels=cfig['num_classes'],
                patch_size=cfig['patch_size'],
                depths=cfig['depth'],
                num_heads=cfig['num_heads'],
                embed_dim=cfig['embed_dims']
            ).to(device)
    
    # 加载预训练权重（如果指定）
    if cfig['use_pretrained']:
        ckpt = torch.load(cfig['use_pretrained'], map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print('Use pretrained weights from: {}'.format(cfig['use_pretrained']))
    model.to(device)

    # ==================== 日志记录器 ====================
    logdir = cfig['logdir']
    writer = SummaryWriter(logdir=logdir)

    # ==================== 优化器初始化 ====================
    if cfig['opt'] == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr=cfig['lr'],weight_decay= cfig['decay'])

    elif cfig['opt'] == "adamw":
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=cfig['lr'], weight_decay=cfig['decay'])

    elif cfig['opt'] == "sgd":
        optimizer = torch.optim.SGD(params = model.parameters(), lr=cfig['lr'], momentum=cfig['momentum'], weight_decay=cfig['decay'])

    # ==================== 学习率调度器 ====================
    if cfig['lrdecay']:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfig['warmup_steps'], t_total=cfig['num_steps'])

    # ==================== 损失函数初始化 ====================
    if cfig['loss_type'] == 'dice_ce':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False, smooth_nr=0, smooth_dr=1e-6)
    elif cfig['loss_type'] == 'dice':
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    elif cfig['loss_type'] == 'ce':
        loss_function = nn.CrossEntropyLoss()
    elif cfig['loss_type'] == 'wce':
        # 加权交叉熵损失，对指定类别增加权重
        weight = np.ones(133).tolist()
        for w in cfig['weight_classes']:
            weight[w] = 10.0

        class_weights = torch.FloatTensor(weight).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif cfig['loss_type'] == 'dice_wce':
        # Dice + 加权交叉熵组合损失
        weight = np.ones(133).tolist()
        for w in cfig['weight_classes']:
            weight[w] = 10.0
        class_weights = torch.FloatTensor(weight).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False, smooth_nr=0, smooth_dr=1e-6, ce_weight=class_weights)
    
    # ==================== 数据加载 ====================
    train_loader, test_loader, val_shape_dict = get_loader(cfig)
    global_step = 0
    dice_val_best = 0.0

    # ==================== 训练循环 ====================
    while global_step < cfig['num_steps']:
        global_step, dice_val_best = train(global_step,train_loader,dice_val_best, val_shape_dict)
    
    # ==================== 保存最终模型 ====================
    checkpoint = {'global_step': global_step,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    save_ckp(checkpoint, logdir+'/model_final_epoch.pt')

# ==================== 日志输出重定向 ====================
class Logger:
    """
    同时输出到控制台和文件的日志记录器
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ==================== 程序入口 ====================
if __name__ == '__main__':
    # ==================== 限制线程数（解决多进程死锁问题）====================
    import os
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
    # 加载YAML配置文件
    yaml_file = 'wholebrainSeg/yaml/unest_large.yaml'
    with open(yaml_file, 'r') as f:
        cfig = yaml.safe_load(f)
    
    # 设置训练设备（优先使用CUDA）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建日志目录并重定向输出
    import sys
    import datetime
    logdir = cfig['logdir']
    os.makedirs(logdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logdir, f'training_{timestamp}.log')
    logger = Logger(log_file)
    sys.stdout = logger
    print(f'日志文件: {log_file}')
    print(f'训练开始时间: {timestamp}')
    print('=' * 60)
    
    try:
        # 启动训练
        main(cfig, device)
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()
        print(f'训练完成，日志已保存至: {log_file}')

# UNesT 全脑分割模型：从训练到应用的完整流程

---

## 一、模型架构概述

**UNesT** 是一种基于分层 Transformer 的医学图像分割网络，采用 **Encoder-Decoder** 结构，核心是 **NestTransformer3D** 编码器，输出 **133 个类别**（132个脑区 + 1个背景）。

**模型规模配置**：

| 模型 | depths | embed_dims | num_heads | 参数量 |
|------|--------|------------|-----------|--------|
| small | [2,2,8] | [96,192,384] | [3,6,12] | 较小 |
| **base** | [2,2,8] | [128,256,512] | [4,8,16] | 中等 |
| large | [2,2,20] | [192,384,768] | [6,12,24] | 较大 |

---

## 二、训练流程

### 1. 数据准备

**目录结构**：
```
train/
    ├── images/
    ├── labels/
validation/
    ├── images/
    ├── labels/
```

**生成数据列表JSON**：使用 `utils/create_json.py` 创建训练/验证数据列表。

### 2. 配置文件设置 (`yaml/unest_base.yaml`)

```yaml
{
  'logdir': '',           # 实验工作目录
  'data_dir': '',         # 数据路径
  'jsondir': '',          # JSON文件路径
  'use_pretrained': '',   # 预训练模型路径
  'fold': 0,              # 交叉验证折数
  'num_classes': 133,     # 分割类别数
  'model_type': 'base',   # 模型类型
  'patch_size': 4,        # Patch大小
  'depth': [2, 2, 8],     # Transformer深度
  'embed_dims': [128, 256, 512],
  'num_heads': [4, 8, 16],
  'num_steps': 50000,     # 训练步数
  'lr': 0.00001,          # 学习率
  'batch_size': 1,
  'loss_type': 'dice',    # 损失函数
  'roi_x/y/z': 96,        # 输入ROI尺寸
  'eval_num': 400,        # 验证间隔步数
}
```

### 3. 数据增强与加载 (`utils/data_utils.py`)

**训练时增强**：
- `RandSpatialCropd`: 随机裁剪到 96×96×96
- `RandFlipd`: 翻转增强
- `RandRotated`: 旋转增强
- `NormalizeIntensityd`: 强度归一化
- `RandScaleIntensityd/ShiftIntensityd`: 强度扰动

### 4. 训练执行 (`main.py`)

**关键组件**：
- **优化器**：AdamW（默认）
- **学习率调度**：WarmupCosineSchedule
- **损失函数**：DiceLoss / DiceCELoss
- **验证策略**：Sliding Window Inference（overlap=0.2）

**训练命令**：
```bash
# 修改 main.py 中的 yaml_file 路径
python main.py
```

**训练流程**：
1. 加载 YAML 配置 → 初始化模型
2. 加载训练/验证数据（SmartCacheDataset）
3. 循环训练 50000 步，每 400 步验证一次
4. 保存最佳模型到 `logdir/model.pt`
5. 训练结束保存最终模型 `model_final_epoch.pt`

---

## 三、推理流程

### 1. 单折推理 (`inference.py`)

```bash
python inference.py \
    --imagesTs_path test_images_path \
    --saved_model_path path2saved_model \
    --base_dir output_path \
    --fold 0 \
    --overlap 0.7 \
    --device 0
```

**推理流程**：
1. 加载 NIfTI 格式测试图像
2. 强度归一化预处理
3. **Sliding Window Inference** (96×96×96, overlap=0.7)
4. Softmax 输出概率图
5. 保存概率图为 `.npy` 格式

**输出目录结构**：
```
pred_0.7/
    ├── fold0/
    │   ├── case1.npy
    │   ├── case2.npy
    ├── fold1/
    ...
```

### 2. 多折集成 (`ensemble.py`)

```bash
python ensemble.py \
    --prob_dir ./pred_0.7 \
    --img_path test_images_path \
    --out_path ./results
```

**集成流程**：
1. 加载所有折的概率图
2. **平均集成**：`mean_prob = sum(prob_folds) / n_folds`
3. `argmax` 获取最终分割标签
4. 恢复原始图像空间，保存为 NIfTI 格式

---

## 四、TICV/PFV 版本（颅内体积/垂体窝体积估计）

### 训练

使用 `main_ticv.py` 和 `yaml/unest_ticv.yaml`。

**模型变体**：`UNesT_ticv` 额外输出 TICV 和 PFV 分割头。

### 推理

```bash
python inference_ticv.py \
    --imagesTs_path test_images_path \
    --saved_model_path path2saved_model \
    --fold 0 \
    --overlap 0.7 \
    --device 0 \
    --results_folder_brain output_path4wholebrain \
    --results_folder_ticv output_path4ticv \
    --results_folder_pfv output_path4pfv
```

---

## 五、完整工作流图

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据准备                                 │
│  NIfTI图像 + 标签 → create_json.py → JSON数据列表                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         模型训练                                 │
│  YAML配置 → main.py → UNesT模型 → 训练50000步 → model.pt        │
│  (数据增强 + DiceLoss + AdamW + WarmupCosineSchedule)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         模型推理                                 │
│  测试图像 → inference.py (5折) → 概率图.npy                      │
│  (Sliding Window + Softmax)                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         结果集成                                 │
│  概率图 → ensemble.py → 平均集成 → argmax → 最终分割.nii.gz      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、Singularity 容器部署

```bash
singularity run -e --contain \
    --home /path/to/inputs/directory/ \
    -B /path/to/inputs/directory/:/INPUTS \
    -B /path/to/working/directory/:/WORKING_DIR \
    -B /path/to/output/directory/:/OUTPUTS \
    -B /tmp:/tmp \
    --nv \
    /path/to/wholebrain.sif \
    [--ticv --w_skull --overlap 0.5 --device 1]
```

**参数说明**：
- `--nv`：启用 GPU
- `--w_skull`：处理非去颅骨数据
- `--ticv`：启用 TICV/PFV 估计
- `--overlap`：滑动窗口重叠率（默认0.7）

---

## 七、关键文件说明

| 文件 | 功能 |
|------|------|
| `main.py` | 训练入口脚本 |
| `inference.py` | 单折推理脚本 |
| `ensemble.py` | 多折结果集成 |
| `main_ticv.py` | TICV版本训练 |
| `inference_ticv.py` | TICV版本推理 |
| `networks/unest.py` | 模型定义 |
| `utils/data_utils.py` | 数据加载与增强 |
| `yaml/*.yaml` | 模型配置文件 |

---

## 八、引用

```bibtex
@article{yu2023unest,
  title={UNesT: local spatial representation learning with hierarchical transformer for efficient medical segmentation},
  author={Yu, Xin and Yang, Qi and Zhou, Yinchi and Cai, Leon Y and Gao, Riqiang and Lee, Ho Hin and Li, Thomas and Bao, Shunxing and Xu, Zhoubing and Lasko, Thomas A and others},
  journal={Medical Image Analysis},
  pages={102939},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{10.1117/12.3009084,
  author = {Xin Yu and Yucheng Tang and Qi Yang and Ho Hin Lee and Shunxing Bao and Yuankai Huo and Bennett A. Landman},
  title = {{Enhancing hierarchical transformers for whole brain segmentation with intracranial measurements integration}},
  volume = {12930},
  booktitle = {Medical Imaging 2024: Clinical and Biomedical Imaging},
  year = {2024},
  doi = {10.1117/12.3009084}
}
```

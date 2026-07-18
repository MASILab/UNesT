# 全脑分割数据集训练指南

## 目录结构

### 原始数据结构
```
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
├── 76384925062202/                    # 单个case
│   ├── dcm/                           # DICOM原始影像
│   │   ├── 1_xxx.dcm
│   │   ├── 2_xxx.dcm
│   │   └── ...
│   └── c_results/                     # 标注结果
│       ├── cleanup_labelmap96_src.nii.gz  # 分割标签（96类）
│       ├── brain_preproc_img.nii.gz       # 预处理后的影像
│       ├── cropped_img.nii.gz             # 裁剪后的影像
│       └── ICV_mask.nii.gz                # 颅内体积掩码
├── UIH164/                            # 另一个case目录
└── ...
```

### 处理后数据结构
```
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/
├── train/
│   ├── images/
│   │   ├── 76384925062202.nii.gz
│   │   └── ...
│   └── labels/
│       ├── 76384925062202_seg.nii.gz
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── json/
    └── fold0.json                     # 数据列表
```

---

## 快速开始

### 一键运行（推荐）
```bash
cd /home/tenoke4090/B_WorkPath/mrqs/UNesT/wholebrainSeg
bash run_wholebrain_training.sh
```

### 分步执行

#### 步骤1: 数据预处理
```bash
python utils/preprocess_wholebrain_dataset.py \
    --source_dir /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset \
    --output_dir /home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset \
    --target_spacing 1.0 1.0 1.0 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

**参数说明：**
- `--source_dir`: 原始数据根目录
- `--output_dir`: 处理后数据输出目录
- `--target_spacing`: 目标体素间距(mm)，默认1mm各向同性
- `--train_ratio`: 训练集比例
- `--val_ratio`: 验证集比例
- `--test_ratio`: 测试集比例

#### 步骤2: 生成JSON数据列表
```bash
python utils/create_json.py
```

#### 步骤3: 模型训练
```bash
# 方法1: 修改main.py中的yaml路径后运行
python main.py

# 方法2: 指定GPU
CUDA_VISIBLE_DEVICES=0 python main.py
```

#### 步骤4: 模型推理
```bash
python inference.py \
    --imagesTs_path /path/to/test/images \
    --saved_model_path /path/to/model.pt \
    --base_dir ./predictions \
    --fold 0 \
    --overlap 0.7 \
    --device 0
```

#### 步骤5: 结果集成（多折训练）
```bash
python ensemble.py \
    --prob_dir ./predictions/pred_0.7 \
    --img_path /path/to/test/images \
    --out_path ./final_predictions
```

---

## 配置文件说明

### YAML配置 (`yaml/unest_wholebrain_custom.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `logdir` | `./logs/...` | 日志和模型保存目录 |
| `data_dir` | `./data` | 数据根目录 |
| `jsondir` | `./json` | JSON列表目录 |
| `num_classes` | `133` | 分割类别数 |
| `model_type` | `base` | 模型规模 |
| `num_steps` | `50000` | 训练步数 |
| `lr` | `0.00001` | 学习率 |
| `batch_size` | `1` | 批次大小 |
| `roi_x/y/z` | `96` | 训练ROI大小 |
| `loss_type` | `dice_ce` | 损失函数类型 |

---

## 依赖安装

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install monai
pip install nibabel pydicom SimpleITK
pip install scipy numpy tensorboardX tqdm pyyaml
```

或使用项目的requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 常见问题

### 1. 标签类别数不匹配
如果你的数据集标签是96类而不是133类，需要修改：
- `yaml`文件中的`num_classes`
- 预处理脚本中的标签映射

### 2. GPU显存不足
- 减小`roi_x/y/z`（如96→64）
- 减小`batch_size`
- 使用`amp=True`开启混合精度训练

### 3. DICOM转换失败
确保安装了SimpleITK：
```bash
pip install SimpleITK
```

### 4. 数据增强导致训练不稳定
减小增强概率：
```yaml
'aug_type': {
    'flip': 0.0,
    'rotate': 0.0,
    'scale_intensity': 0.0,
    'shif_intensity': 0.0
}
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `utils/preprocess_wholebrain_dataset.py` | 数据预处理脚本 |
| `utils/create_json.py` | JSON列表生成脚本 |
| `yaml/unest_wholebrain_custom.yaml` | 训练配置文件 |
| `run_wholebrain_training.sh` | 一键运行脚本 |
| `main.py` | 训练主脚本 |
| `inference.py` | 推理脚本 |
| `ensemble.py` | 结果集成脚本 |

---

## 训练输出

```
logs/wholebrain_custom/
├── model.pt              # 最佳模型
├── model_final_epoch.pt  # 最终模型
└── events.out.tfevents.* # TensorBoard日志
```

**可视化训练过程：**
```bash
tensorboard --logdir logs/wholebrain_custom
```

---

## 联系方式

如有问题，请参考：
- 项目README: `README.md`
- 预训练指南: `pretrainning.md`
- MONAI文档: https://docs.monai.io/

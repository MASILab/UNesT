用户想查询MRI影像伪影检测相关的训练参数。这是一个比较专业的医学图像处理问题，涉及到MRI伪影检测的训练设置。

让我先在当前项目中搜索是否有相关的伪影检测代码或配置，然后可能需要查询知识库获取更多相关信息。

首先搜索项目中是否有伪影(artifact)相关的代码。
让我先搜索项目中是否有伪影检测相关的代码和配置，然后查询相关知识库。
知识库中没有直接关于MRI影像伪影检测的训练参数信息。让我尝试使用网络搜索来获取相关信息。不过在这个项目中，我应该基于我对MRI伪影检测的专业知识来回答用户的问题。

MRI伪影检测是一个医学图像处理领域的问题，涉及到使用深度学习模型来识别和分类MRI图像中的各种伪影。让我整理一下这方面的训练参数信息。
当前项目和知识库中没有直接的MRI伪影检测相关代码。让我基于专业知识为您整理MRI影像伪影检测的典型训练参数：

## MRI影像伪影检测 - 训练参数参考

### 一、常见伪影类型

| 伪影类型 | 英文名称 | 成因 |
|---------|---------|------|
| 运动伪影 | Motion Artifacts | 患者/生理运动 |
| 吉布斯伪影 | Gibbs Ringing | 截断傅里叶变换 |
| 混叠伪影 | Aliasing/Wrap-around | 采样不足 |
| 化学位移伪影 | Chemical Shift | 脂肪/水共振频率差异 |
| 磁敏感伪影 | Susceptibility | 磁场不均匀 |
| 层间干扰 | Cross-talk | 层间距不足 |
| 卷褶伪影 | Fold-over | FOV过小 |

---

### 二、典型训练参数配置

```python
# MRI伪影检测典型配置
config = {
    # ========== 数据参数 ==========
    'input_channels': 1,           # MRI通常为灰度图
    'input_size': (256, 256),      # 或 (128, 128, 128) for 3D
    'num_classes': 7,              # 伪影类别数（含正常）
    'batch_size': 16,              # 2D: 16-32; 3D: 2-4
    
    # ========== 数据增强 ==========
    'augmentation': {
        'rotation': 15,            # 旋转角度范围
        'flip': True,              # 水平/垂直翻转
        'scale': (0.9, 1.1),       # 缩放范围
        'noise_std': 0.05,         # 高斯噪声标准差
        'bias_field': True,        # 模拟偏置场伪影
        'intensity_shift': 0.1,    # 强度偏移
    },
    
    # ========== 模型参数 ==========
    'backbone': 'ResNet50',        # 或 EfficientNet, DenseNet
    'pretrained': True,            # ImageNet预训练
    'dropout': 0.5,
    
    # ========== 训练参数 ==========
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,         # 或 3e-4
    'weight_decay': 1e-4,
    'scheduler': 'CosineAnnealingLR',
    'warmup_epochs': 5,
    'num_epochs': 100,
    
    # ========== 损失函数 ==========
    'loss': {
        'type': 'FocalLoss',       # 处理类别不平衡
        'gamma': 2.0,
        'alpha': None,             # 或设置类别权重
    },
    # 或使用多任务损失：
    # 'loss': 'CrossEntropy + DiceLoss',
    
    # ========== 正则化 ==========
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,            # Mixup增强
    
    # ========== 验证策略 ==========
    'val_split': 0.2,
    'early_stopping_patience': 15,
    'eval_metrics': ['Accuracy', 'F1', 'AUC'],
}
```

---

### 三、针对UNesT项目的适配建议

基于您当前UNesT项目的架构，可以修改为伪影检测任务：

```yaml
# yaml/artifact_detection.yaml
{
  'task': 'classification',       # 改为分类任务
  'num_classes': 7,               # 伪影类别数
  'model_type': 'base',
  'input_mode': '2D',             # 或 '3D'
  
  # 训练参数
  'num_steps': 30000,
  'lr': 0.0001,
  'batch_size': 16,
  'loss_type': 'focal',           # 或 'ce' + label_smoothing
  
  # 数据增强
  'aug_intensity': True,
  'aug_motion_blur': True,        # 模拟运动伪影
  
  # 数据路径
  'data_dir': '/path/to/mri_artifacts',
  'jsondir': '/path/to/splits',
}
```

---

### 四、数据准备建议

```
dataset/
├── train/
│   ├── normal/           # 正常MRI
│   ├── motion/           # 运动伪影
│   ├── gibbs/            # 吉布斯伪影
│   ├── aliasing/         # 混叠伪影
│   ├── chemical_shift/   # 化学位移
│   ├── susceptibility/   # 磁敏感
│   └── other/            # 其他伪影
├── val/
│   └── ... (同上)
└── test/
    └── ... (同上)
```

---

### 五、常用评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| **准确率** | 整体正确率 | \(\frac{TP+TN}{TP+TN+FP+FN}\) |
| **F1-Score** | 精确率与召回率调和平均 | \(\frac{2 \cdot P \cdot R}{P+R}\) |
| **AUC-ROC** | 分类器性能 | ROC曲线下面积 |
| **混淆矩阵** | 各类别分类详情 | - |

---

如果您需要我基于UNesT项目创建一个伪影检测的训练脚本或配置文件，请切换到 **Craft模式** 告诉我具体需求。
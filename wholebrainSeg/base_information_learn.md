用户想了解 "(5-fold ensembled)" 的含义。这是机器学习中常用的交叉验证和模型集成技术。让我简要解释这个概念。
## 5-Fold Ensembled（五折集成）解释

### 含义

**5-fold ensembled** 指的是使用 **5折交叉验证** 训练5个模型，然后将5个模型的预测结果进行**集成**。

---

### 具体流程

```
原始训练数据集
        ↓
┌───────────────────────────────────────────────┐
│              划分为5份 (Fold 0-4)              │
├───────────────────────────────────────────────┤
│  Fold 0: [验证] [训练] [训练] [训练] [训练]    │
│  Fold 1: [训练] [验证] [训练] [训练] [训练]    │
│  Fold 2: [训练] [训练] [验证] [训练] [训练]    │
│  Fold 3: [训练] [训练] [训练] [验证] [训练]    │
│  Fold 4: [训练] [训练] [训练] [训练] [验证]    │
└───────────────────────────────────────────────┘
        ↓
训练5个独立的模型 → 保存5个权重文件 (model_fold0.pt ~ model_fold4.pt)
        ↓
推理时：5个模型分别预测 → 概率平均 → 最终结果
```

---

### 为什么这样做？

| 优势 | 说明 |
|------|------|
| **数据充分利用** | 每个样本都作为验证集被评估过 |
| **减少过拟合** | 5个模型从不同数据划分学习，泛化能力更强 |
| **预测更稳定** | 集成多个模型减少单一模型的偏差 |
| **性能提升** | 平均集成通常比单模型效果更好 |

---

### 本项目中的实现

**训练阶段**：
```bash
# 训练5个独立的模型
python main.py --fold 0  # 使用 fold0.json
python main.py --fold 1  # 使用 fold1.json
python main.py --fold 2  # 使用 fold2.json
python main.py --fold 3  # 使用 fold3.json
python main.py --fold 4  # 使用 fold4.json
```

**推理阶段**：
```bash
# 每个模型分别推理，保存概率图
python inference.py --fold 0 --saved_model_path model_fold0.pt
python inference.py --fold 1 --saved_model_path model_fold1.pt
...
```

**集成阶段** (`ensemble.py`)：
```python
# 对5个模型的概率图取平均
infer_outputs = (prob_fold0 + prob_fold1 + ... + prob_fold4) / 5
# argmax 得到最终分割
labels = np.argmax(infer_outputs, axis=1)
```

---

### 简单类比

类似于 **"多位专家会诊"**：
- 5位医生（模型）各自独立诊断
- 综合所有人的意见（概率平均）
- 得出更可靠的最终诊断结果
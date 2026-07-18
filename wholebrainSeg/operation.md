/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset册目录存放原始影像和标注结果数据集
如76384925062202为一个标注案例，其数据结构如下：
原始影像数据：
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/76384925062202/dcm

数据标注结果：
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/76384925062202/c_results
如何对这些数据进行预处理，使得图像可以用于训练，将训练代码写出来


用户想要我为 inference_yc.py 文件添加详细的中文注释。这个文件看起来是一个更完整的推理脚本，包含了 skull stripping（颅骨剥离）、强度归一化、后处理等步骤。让我为它添加详细的中文注释。
我已经为 inference_yc.py 添加了详细的中文注释。这个脚本比之前的 inference.py 更完整，包含了：
1. 颅骨剥离
2. FCM强度归一化
3. 完整的预处理pipeline
4. 后处理（最大连通域提取）

让我给用户一个总结。
已为 `inference_yc.py` 添加详细中文注释。该脚本相比 `inference.py` 更完整，包含**端到端推理流程**：

---

## inference_yc.py vs inference.py 对比

| 功能 | inference.py | inference_yc.py |
|------|-------------|-----------------|
| 图像加载 | ✅ | ✅ |
| 颅骨剥离 | ❌ | ✅ ROBEX |
| 强度归一化 | ✅ 简单Z-score | ✅ FCM白质归一化 |
| 重采样 | ❌ | ✅ 1mm各向同性 |
| 方向校正 | ❌ | ✅ RAS统一方向 |
| 前景裁剪 | ❌ | ✅ |
| 滑动窗口推理 | ✅ | ✅ |
| 后处理 | ❌ | ✅ 最大连通域提取 |

---

## 推理流程图

```
输入图像
    │
    ▼
┌─────────────────┐
│ 保存原始副本     │ ──► {原文件名}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 颅骨剥离(ROBEX)  │ ──► strip_{原文件名}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 重采样(1mm³)     │
│ 方向校正(RAS)    │
│ 前景裁剪         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FCM强度归一化    │ ──► norm_{原文件名}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 滑动窗口推理     │
│ Softmax + Argmax│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 最大连通域提取   │ ──► label_{原文件名}
└────────┬────────┘
         │
         ▼
    最终分割结果
```

---

## 使用方法

```bash
python inference_yc.py \
    --data_dir /path/to/test/images \
    --model_path /path/to/model \
    --results_dir /path/to/results \
    --overlap 0.2 \
    --device 0
```

**依赖安装：**
```bash
pip install pyrobex
pip install intensity-normalization
pip install antspyx
```
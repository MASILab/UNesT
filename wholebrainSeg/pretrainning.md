用户想了解如何执行MONAI Bundle的Whole Brain Segmentation，包括数据下载和预处理。让我先查看用户提到的相关文件，然后搜索MONAI Bundle的相关信息。

我应该：
1. 查看 call_Run_Deep_brain.sh 脚本
2. 查看 MONAI Bundle 相关的文档和代码
3. 查询知识库获取更多信息
让我查看相关文件并获取MONAI Bundle的使用指南。
知识库没有返回关于MONAI Bundle的具体内容。让我从GitHub链接获取MONAI Bundle的文档，并查看项目中的相关文件来提供完整的使用指南。
让我获取MONAI Bundle的详细文档：
GitHub页面没有返回具体的MONAI Bundle文档内容，让我尝试获取README文件或docs文件夹的内容。
Let me try a different URL structure for MONAI model zoo
Let me try the models directory listing to find the correct path
网页抓取没有返回实际内容。让我基于项目现有的README和我对这个领域的了解，为用户提供MONAI Bundle的完整使用指南。
根据项目README和MONAI Bundle的标准使用方式，为您提供完整的使用指南：

## MONAI Bundle Whole Brain Segmentation 使用指南

---

### 一、环境准备

```bash
# 安装MONAI
pip install monai

# 安装MONAI Bundle支持
pip install monai[all]

# 或使用项目requirements
cd /home/tenoke4090/B_WorkPath/mrqs/UNesT
pip install -r requirements.txt
```

---

### 二、数据下载

#### 1. 预训练权重下载
```bash
# Whole Brain Segmentation权重
wget https://vanderbilt.box.com/shared/static/lemw2wfyxtbno7vy1eemth4x6qreeie5 -O wholebrainseg_weight.pth

# 带TICV/PFV的权重
wget https://vanderbilt.box.com/shared/static/ghu19kx8372vxo7nos9dyco7d9qtbt5q -O wholebrainseg_ticv_weight.pth
```

#### 2. 训练数据集
常用的全脑分割公开数据集：
| 数据集 | 说明 | 下载链接 |
|--------|------|----------|
| **OASIS** | 开放获取神经影像 | https://www.oasis-brains.org/ |
| **ADNI** | 阿尔茨海默病神经影像 | http://adni.loni.usc.edu/ |
| **IXI** | 信息提取 | https://brain-development.org/ixi-dataset/ |
| **HCP** | 人类连接组计划 | https://www.humanconnectome.org/ |

---

### 三、数据预处理

#### 1. 数据组织结构
```
data/
├── train/
│   ├── images/          # 原始MRI图像
│   └── labels/          # 分割标签
├── validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### 2. 创建数据列表JSON
使用项目提供的 `create_json.py`：

```bash
cd /home/tenoke4090/B_WorkPath/mrqs/UNesT/wholebrainSeg
python utils/create_json.py \
    --data_dir /path/to/data \
    --output_dir ./json_files
```

#### 3. 预处理步骤（NIfTI格式）
```python
# 典型预处理流程
import nibabel as nib
import numpy as np
from monai.transforms import *

# 定义预处理transform
preprocess = Compose([
    LoadImage(image_only=True),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    Orientation(axs="RAS"),
    ScaleIntensityRange(a_min=-100, a_max=100, b_min=0.0, b_max=1.0, clip=True),
    CropForeground(),
    ToTensor(),
])
```

---

### 四、训练

#### 1. 配置YAML文件
编辑 `yaml/unest_base.yaml`：
```yaml
{
  'logdir': './logs/experiment1',
  'data_dir': '/path/to/data',
  'jsondir': './json_files',
  'use_pretrained': '',  # 或指定预训练权重路径
  'fold': 0,
  'num_classes': 133,
  'model_type': 'base',
  'num_steps': 50000,
  'lr': 0.00001,
  'batch_size': 1,
  'roi_x': 96,
  'roi_y': 96,
  'roi_z': 96,
}
```

#### 2. 启动训练
```bash
# 基础训练
python main.py

# 带TICV/PFV估计的训练
python main_ticv.py
```

---

### 五、推理

#### 1. 单折推理
```bash
python inference.py \
    --imagesTs_path /path/to/test/images \
    --saved_model_path ./weights/model.pth \
    --base_dir ./outputs \
    --fold 0 \
    --overlap 0.7 \
    --device 0
```

#### 2. 多折集成
```bash
# 首先运行5折推理 (fold 0-4)
for i in {0..4}; do
    python inference.py --fold $i --overlap 0.7
done

# 集成预测结果
python ensemble.py \
    --prob_dir ./outputs \
    --img_path /path/to/test/images \
    --out_path ./final_predictions
```

#### 3. TICV/PFV推理
```bash
python inference_ticv.py \
    --imagesTs_path /path/to/test/images \
    --saved_model_path ./weights/ticv_model.pth \
    --fold 0 \
    --overlap 0.7 \
    --device 0 \
    --results_folder_brain ./brain_seg \
    --results_folder_ticv ./ticv_output \
    --results_folder_pfv ./pfv_output
```

---

### 六、MNI空间转换回原始空间

#### 1. 依赖安装
```bash
# ANTs
sudo apt-get install ants

# NiftyReg
git clone https://github.com/KCL-BMEIS/niftyreg.git
cd niftyreg && mkdir build && cd build
cmake .. && make && sudo make install
```

#### 2. 执行转换
```bash
cd /home/tenoke4090/B_WorkPath/mrqs/UNesT/wholebrainSeg/mni2orig

# 编辑 call_Run_Deep_brain.sh 配置路径
# 然后执行
bash call_Run_Deep_brain.sh
```

---

### 七、MONAI Bundle API使用

```python
from monai.bundle import download, load

# 下载Bundle
download(name="wholeBrainSeg_Large_UNEST_segmentation", bundle_dir="./models")

# 加载模型
model = load(
    name="wholeBrainSeg_Large_UNEST_segmentation",
    bundle_dir="./models"
)

# 推理
from monai.transforms import LoadImage, ToTensor, NormalizeIntensity
from monai.inferers import sliding_window_inference

# 加载图像
image = LoadImage(image_only=True)("patient.nii.gz")
image = NormalizeIntensity()(image)
image = ToTensor()(image)

# 执行推理
with torch.no_grad():
    pred = sliding_window_inference(
        image, 
        roi_size=(96, 96, 96), 
        sw_batch_size=1, 
        predictor=model,
        overlap=0.7
    )
```

---

### 八、Singularity容器使用

```bash
# 下载预构建容器
wget https://vanderbilt.box.com/s/glnq14bvbmx8a9lpklzoe7olgdcwkqp9 -O wholebrain.sif

# 运行容器
singularity run -e --contain \
    --home /path/to/inputs/directory/ \
    -B /path/to/inputs/directory/:/INPUTS \
    -B /path/to/working/directory/:/WORKING_DIR \
    -B /path/to/output/directory/:/OUTPUTS \
    -B /tmp:/tmp \
    --nv \
    wholebrain.sif \
    [--ticv --w_skull --overlap 0.5 --device 1]
```

**参数说明：**
- `--nv`: 启用GPU支持
- `--w_skull`: 输入为非去颅骨数据
- `--ticv`: 启用TICV/PFV估计
- `--overlap`: 滑动窗口重叠率（默认0.7）
- `--device`: GPU设备ID
#!/bin/bash
# ============================================================
# UNesT 全脑分割训练一键运行脚本
# ============================================================
# 
# 使用方法:
#   bash run_wholebrain_training.sh              # 标准模式 (匹配 inference.py)
#   bash run_wholebrain_training.sh --yc         # YC模式 (匹配 inference_yc.py)
#   bash run_wholebrain_training.sh --yc --skip_skull_strip  # YC模式但跳过颅骨剥离
#
# 两种模式区别:
#   标准模式: Z-score归一化，推理时使用 inference.py
#   YC模式:   颅骨剥离 + FCM归一化，推理时使用 inference_yc.py
#
# 作者: AI Assistant
# ============================================================

set -e  # 遇到错误即退出

# ==================== 解析参数 ====================
USE_YC_MODE=false
SKIP_SKULL_STRIP=false

for arg in "$@"; do
    case $arg in
        --yc)
            USE_YC_MODE=true
            shift
            ;;
        --skip_skull_strip)
            SKIP_SKULL_STRIP=true
            shift
            ;;
    esac
done

# ==================== 配置区域 ====================
# 数据路径
SOURCE_DATA_DIR="/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset"
OUTPUT_DATA_DIR="/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset"

# 训练配置
YAML_CONFIG="yaml/unest_wholebrain_custom.yaml"
GPU_ID=0
FOLD=0

# ==================== 显示模式信息 ====================
echo "=========================================="
if [ "$USE_YC_MODE" = true ]; then
    echo "UNesT 全脑分割训练 - YC模式"
    echo "匹配推理脚本: inference_yc.py"
    echo "预处理包含: 颅骨剥离 + FCM归一化"
    if [ "$SKIP_SKULL_STRIP" = true ]; then
        echo "注意: 已跳过颅骨剥离步骤"
    fi
else
    echo "UNesT 全脑分割训练 - 标准模式"
    echo "匹配推理脚本: inference.py"
    echo "预处理包含: Z-score归一化"
fi
echo "=========================================="
echo ""

# ==================== 环境检查 ====================
# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# 检查必要的Python包
echo "检查Python依赖..."
python -c "import torch; import monai; import nibabel; print('基础包OK')" || {
    echo "错误: 缺少基础Python包"
    echo "请运行: pip install torch monai nibabel"
    exit 1
}

# YC模式额外检查
if [ "$USE_YC_MODE" = true ] && [ "$SKIP_SKULL_STRIP" = false ]; then
    python -c "import pyrobex; print('pyrobex OK')" 2>/dev/null || {
        echo "错误: 缺少pyrobex包（颅骨剥离需要）"
        echo "安装: pip install pyrobex"
        echo "或使用 --skip_skull_strip 跳过颅骨剥离"
        exit 1
    }
fi

if [ "$USE_YC_MODE" = true ]; then
    python -c "from intensity_normalization.normalize.fcm import FCMNormalize; print('intensity-normalization OK')" 2>/dev/null || {
        echo "错误: 缺少intensity-normalization包"
        echo "安装: pip install intensity-normalization"
        exit 1
    }
fi

echo "依赖检查通过!"
echo ""

# ==================== 步骤1: 数据预处理 ====================
echo "=========================================="
echo "步骤1: 数据预处理"
echo "=========================================="

if [ ! -d "${OUTPUT_DATA_DIR}/train/images" ] || [ -z "$(ls -A ${OUTPUT_DATA_DIR}/train/images 2>/dev/null)" ]; then
    echo "开始数据预处理..."
    
    if [ "$USE_YC_MODE" = true ]; then
        # YC模式预处理
        PREPROCESS_CMD="python utils/preprocess_for_inference_yc.py --source_dir ${SOURCE_DATA_DIR} --output_dir ${OUTPUT_DATA_DIR}"
        if [ "$SKIP_SKULL_STRIP" = true ]; then
            PREPROCESS_CMD="${PREPROCESS_CMD} --skip_skull_strip"
        fi
        eval ${PREPROCESS_CMD}
    else
        # 标准模式预处理
        python utils/preprocess_wholebrain_dataset.py \
            --source_dir ${SOURCE_DATA_DIR} \
            --output_dir ${OUTPUT_DATA_DIR} \
            --target_spacing 1.0 1.0 1.0 \
            --train_ratio 0.7 \
            --val_ratio 0.15 \
            --test_ratio 0.15 \
            --random_seed 42
    fi
else
    echo "数据已预处理，跳过此步骤"
    echo "如需重新预处理，请先删除 ${OUTPUT_DATA_DIR}/train 目录"
fi

# 检查JSON文件
if [ ! -f "${OUTPUT_DATA_DIR}/json/fold0.json" ]; then
    echo "错误: 未生成JSON数据列表"
    exit 1
fi

# 显示数据统计
echo ""
echo "数据集统计:"
TRAIN_COUNT=$(ls -1 "${OUTPUT_DATA_DIR}/train/images/" 2>/dev/null | wc -l)
VAL_COUNT=$(ls -1 "${OUTPUT_DATA_DIR}/val/images/" 2>/dev/null | wc -l)
TEST_COUNT=$(ls -1 "${OUTPUT_DATA_DIR}/test/images/" 2>/dev/null | wc -l)
echo "  训练集: ${TRAIN_COUNT} cases"
echo "  验证集: ${VAL_COUNT} cases"
echo "  测试集: ${TEST_COUNT} cases"
echo ""

# ==================== 步骤2: 模型训练 ====================
echo "=========================================="
echo "步骤2: 模型训练"
echo "=========================================="

if [ -f "main.py" ]; then
    # 备份并更新配置文件
    if [ ! -f "yaml/unest_base.yaml.bak" ]; then
        cp yaml/unest_base.yaml yaml/unest_base.yaml.bak
    fi
    cp ${YAML_CONFIG} yaml/unest_base.yaml
    
    echo "开始训练..."
    echo "配置文件: ${YAML_CONFIG}"
    echo "GPU: ${GPU_ID}"
    echo "按Ctrl+C可中断训练"
    echo ""
    
    # 启动训练
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py
    
    echo ""
    echo "训练完成!"
else
    echo "错误: 未找到main.py"
    exit 1
fi

# ==================== 步骤3: 模型推理 ====================
echo ""
echo "=========================================="
echo "步骤3: 模型推理"
echo "=========================================="

TEST_IMAGES="${OUTPUT_DATA_DIR}/test/images"
MODEL_WEIGHT="${OUTPUT_DATA_DIR}/logs/wholebrain_custom/model.pt"
OUTPUT_DIR="${OUTPUT_DATA_DIR}/predictions"

# 查找最新的模型文件
if [ ! -f "${MODEL_WEIGHT}" ]; then
    # 尝试查找其他可能的模型位置
    MODEL_WEIGHT=$(find ./logs -name "*.pt" -o -name "*.pth" 2>/dev/null | head -1)
fi

if [ -d "${TEST_IMAGES}" ] && [ -n "${MODEL_WEIGHT}" ]; then
    echo "开始推理..."
    echo "测试图像: ${TEST_IMAGES}"
    echo "模型权重: ${MODEL_WEIGHT}"
    echo "输出目录: ${OUTPUT_DIR}"
    echo ""
    
    if [ "$USE_YC_MODE" = true ]; then
        # 使用 inference_yc.py 推理
        python inference_yc.py \
            --data_dir ${TEST_IMAGES} \
            --model_path $(dirname ${MODEL_WEIGHT}) \
            --results_dir ${OUTPUT_DIR} \
            --overlap 0.7 \
            --device ${GPU_ID}
    else
        # 使用 inference.py 推理
        python inference.py \
            --imagesTs_path ${TEST_IMAGES} \
            --saved_model_path $(dirname ${MODEL_WEIGHT}) \
            --base_dir ${OUTPUT_DIR} \
            --fold ${FOLD} \
            --overlap 0.7 \
            --device ${GPU_ID}
    fi
    
    echo ""
    echo "推理完成!"
else
    echo "跳过推理: 测试数据或模型权重不存在"
fi

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "全部流程完成!"
echo "=========================================="
echo ""
echo "输出目录:"
echo "  - 训练日志: ${OUTPUT_DATA_DIR}/logs/"
echo "  - 预测结果: ${OUTPUT_DIR}/"
echo ""
echo "推理模式: $([ "$USE_YC_MODE" = true ] && echo "inference_yc.py (颅骨剥离+FCM归一化)" || echo "inference.py (Z-score归一化)")"
echo ""

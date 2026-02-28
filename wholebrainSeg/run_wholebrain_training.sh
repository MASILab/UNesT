#!/bin/bash

# ============================================================
# UNesT 全脑分割训练一键运行脚本
# ============================================================
# 
# 使用方法:
#   bash run_wholebrain_training.sh
#
# 步骤:
#   1. 数据预处理（DICOM转NIfTI，生成JSON列表）
#   2. 模型训练
#   3. 模型推理
#   4. 结果集成
#
# 作者: AI Assistant
# ============================================================

set -e  # 遇到错误即退出

# ==================== 配置区域 ====================
# 数据路径
SOURCE_DATA_DIR="/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset"
OUTPUT_DATA_DIR="/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset"

# 训练配置
YAML_CONFIG="yaml/unest_wholebrain_custom.yaml"
GPU_ID=0
FOLD=0

# 预训练权重（可选）
PRETRAINED_WEIGHT=""  # 留空表示从头训练

# ==================== 环境检查 ====================
echo "=========================================="
echo "UNesT 全脑分割训练流程"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# ==================== 步骤1: 数据预处理 ====================
echo ""
echo "=========================================="
echo "步骤1: 数据预处理"
echo "=========================================="

if [ ! -d "${OUTPUT_DATA_DIR}/train/images" ] || [ -z "$(ls -A ${OUTPUT_DATA_DIR}/train/images 2>/dev/null)" ]; then
    echo "开始数据预处理..."
    python utils/preprocess_wholebrain_dataset.py \
        --source_dir ${SOURCE_DATA_DIR} \
        --output_dir ${OUTPUT_DATA_DIR} \
        --target_spacing 1.0 1.0 1.0 \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --random_seed 42
else
    echo "数据已预处理，跳过此步骤"
fi

# 检查JSON文件
if [ ! -f "${OUTPUT_DATA_DIR}/json/fold0.json" ]; then
    echo "生成JSON数据列表..."
    python utils/create_json.py
fi

# ==================== 步骤2: 模型训练 ====================
echo ""
echo "=========================================="
echo "步骤2: 模型训练"
echo "=========================================="

# 修改main.py中的yaml路径
if [ -f "main.py" ]; then
    # 创建临时配置文件链接
    if [ ! -f "yaml/unest_base.yaml.bak" ]; then
        cp yaml/unest_base.yaml yaml/unest_base.yaml.bak
    fi
    cp ${YAML_CONFIG} yaml/unest_base.yaml
    
    echo "开始训练..."
    echo "配置文件: ${YAML_CONFIG}"
    echo "GPU: ${GPU_ID}"
    
    # 启动训练
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py
    
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

if [ -d "${TEST_IMAGES}" ] && [ -f "${MODEL_WEIGHT}" ]; then
    echo "开始推理..."
    
    python inference.py \
        --imagesTs_path ${TEST_IMAGES} \
        --saved_model_path ${MODEL_WEIGHT} \
        --base_dir ${OUTPUT_DIR} \
        --fold ${FOLD} \
        --overlap 0.7 \
        --device ${GPU_ID}
    
    echo "推理完成!"
else
    echo "跳过推理: 测试数据或模型权重不存在"
fi

# ==================== 步骤4: 结果集成（多折训练时使用） ====================
echo ""
echo "=========================================="
echo "步骤4: 结果集成（可选）"
echo "=========================================="

if [ -d "${OUTPUT_DIR}/pred_0.7" ]; then
    echo "集成预测结果..."
    
    python ensemble.py \
        --prob_dir ${OUTPUT_DIR}/pred_0.7 \
        --img_path ${TEST_IMAGES} \
        --out_path ${OUTPUT_DIR}/final_predictions
    
    echo "集成完成!"
else
    echo "跳过集成: 无多折预测结果"
fi

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "全部流程完成!"
echo "=========================================="
echo ""
echo "输出目录:"
echo "  - 模型权重: ${OUTPUT_DATA_DIR}/logs/wholebrain_custom/"
echo "  - 预测结果: ${OUTPUT_DIR}/"
echo ""

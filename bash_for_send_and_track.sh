#!/bin/bash

# SGLang Router 测试脚本
# 用途：发送请求并追踪路由分配情况

# =============================================================================
# 配置参数 - 根据需要修改这些参数
# =============================================================================

# 基础配置
NUM_REQUESTS=200               # 发送的请求数量
REQUEST_RATE=50.0              # 请求速率（req/s），inf表示最大速率
DATASET="random"               # 数据集类型: random, sharegpt, custom

# Random数据集参数
INPUT_LEN=512                  # 平均输入长度（tokens）
OUTPUT_LEN=128                 # 平均输出长度（tokens）
RANGE_RATIO=0.5                # 长度变化范围（±50%）

# 路由器配置
ROUTER_URL="http://localhost:60009"    # 路由器地址

# 输出配置
# OUTPUT_FILE="router_test_$(date +%Y%m%d_%H%M%S).csv"  # 自动生成时间戳
OUTPUT_FILE=""                 # 留空则自动生成带时间戳的文件名

# 自定义数据集路径（仅当DATASET="custom"时使用）
DATASET_PATH=""

# =============================================================================
# 预设测试场景 - 取消注释以使用
# =============================================================================

# # 场景1: 快速测试（少量请求，快速验证）
# NUM_REQUESTS=20
# REQUEST_RATE=5.0
# DATASET="random"
# INPUT_LEN=256
# OUTPUT_LEN=64

# # 场景2: 性能测试（中等规模）
# NUM_REQUESTS=500
# REQUEST_RATE=20.0
# DATASET="random"
# INPUT_LEN=512
# OUTPUT_LEN=128

# # 场景3: 压力测试（大量请求，高速率）
# NUM_REQUESTS=1000
# REQUEST_RATE=50.0
# DATASET="random"
# INPUT_LEN=1024
# OUTPUT_LEN=256

# # 场景4: 极限测试（最大速率）
# NUM_REQUESTS=200
# REQUEST_RATE="inf"
# DATASET="random"
# INPUT_LEN=512
# OUTPUT_LEN=128

# # 场景5: 长文本测试
# NUM_REQUESTS=50
# REQUEST_RATE=5.0
# DATASET="random"
# INPUT_LEN=2048
# OUTPUT_LEN=512
# RANGE_RATIO=0.3

# # 场景6: ShareGPT真实数据测试
# NUM_REQUESTS=100
# REQUEST_RATE=10.0
# DATASET="sharegpt"

# =============================================================================
# 执行脚本 - 通常不需要修改以下内容
# =============================================================================

echo "=========================================="
echo "SGLang Router 测试配置"
echo "=========================================="
echo "请求数量: $NUM_REQUESTS"
echo "请求速率: $REQUEST_RATE req/s"
echo "数据集: $DATASET"
if [ "$DATASET" = "random" ]; then
    echo "输入长度: $INPUT_LEN ±$(echo "scale=0; $INPUT_LEN * $RANGE_RATIO / 1" | bc)%"
    echo "输出长度: $OUTPUT_LEN ±$(echo "scale=0; $OUTPUT_LEN * $RANGE_RATIO / 1" | bc)%"
fi
echo "路由器地址: $ROUTER_URL"
echo "=========================================="
echo ""

# 构建命令
CMD="python /nas/ganluo/sglang/send_request_and_track.py"
CMD="$CMD --num-requests $NUM_REQUESTS"
CMD="$CMD --request-rate $REQUEST_RATE"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --router-url $ROUTER_URL"

# 添加random数据集参数
if [ "$DATASET" = "random" ]; then
    CMD="$CMD --input-len $INPUT_LEN"
    CMD="$CMD --output-len $OUTPUT_LEN"
    CMD="$CMD --range-ratio $RANGE_RATIO"
fi

# 添加custom数据集路径
if [ "$DATASET" = "custom" ] && [ ! -z "$DATASET_PATH" ]; then
    CMD="$CMD --dataset-path $DATASET_PATH"
fi

# 添加输出文件
if [ ! -z "$OUTPUT_FILE" ]; then
    CMD="$CMD --output $OUTPUT_FILE"
fi

# 显示完整命令
echo "执行命令:"
echo "$CMD"
echo ""

# 计算预计完成时间
if [ "$REQUEST_RATE" != "inf" ]; then
    ESTIMATED_TIME=$(echo "scale=1; $NUM_REQUESTS / $REQUEST_RATE" | bc)
    echo "预计测试时间: ${ESTIMATED_TIME}秒（不含处理时间）"
    echo ""
fi

# 执行
eval $CMD

# 显示结果文件位置
echo ""
echo "=========================================="
echo "测试完成！"
if [ -z "$OUTPUT_FILE" ]; then
    echo "结果已保存到: router_test_*.csv"
else
    echo "结果已保存到: $OUTPUT_FILE"
fi
echo "=========================================="
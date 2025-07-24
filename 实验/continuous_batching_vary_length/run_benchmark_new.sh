#!/bin/bash

# 服务器启动命令（保持不变）
# python3 -m sglang.launch_server ...

# --- 基准测试配置 ---
REQUEST_RATES=(50 30 10 15 20 25 35 40 45 55 60 65 70 75 80 85 90 95 100 5)
MEAN_PROMPT_LEN=256
MEAN_OUTPUT_LEN=64
STD_DEV_RATIO=1  # 使用标准差比例代替方差，更直观
NUM_PROMPTS=2000
MODEL_TOKENIZER_PATH="/data/pretrained_models/Llama-2-7b-hf"

# --- 文件输出配置 ---
RESULTS_DIR="/home/lg/sglang/实验/continuous_batching_vary_length/results_std_1_new--max-concurrency50"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_REPORT_FILE="${RESULTS_DIR}/benchmark_report_${TIMESTAMP}.jsonl"
DATASET_FILE="${RESULTS_DIR}/benchmark_dataset_${TIMESTAMP}.jsonl" # 数据集文件
TMP_JSON_PREFIX="${RESULTS_DIR}/tmp_run_results"

mkdir -p "$RESULTS_DIR"

# --- 脚本开始 ---
echo "开始基准测试..."
echo "本次将测试QPS: ${REQUEST_RATES[@]}"
echo "输入长度均值: $MEAN_PROMPT_LEN, 输出长度均值: $MEAN_OUTPUT_LEN"
echo "将进行5次重复测试"
echo "----------------------------------------------------"

# --- 步骤1: 在所有测试开始前，生成包含可变长度的统一数据集 ---
echo "正在生成测试数据集..."
python3 /home/lg/sglang/实验/continuous_batching_vary_length/generate_variable_length_dataset.py \
    --num-prompts "$NUM_PROMPTS" \
    --mean-input-len "$MEAN_PROMPT_LEN" \
    --mean-output-len "$MEAN_OUTPUT_LEN" \
    --std-dev-ratio "$STD_DEV_RATIO" \
    --output-file "$DATASET_FILE"
echo "数据集生成完毕: $DATASET_FILE"
echo "----------------------------------------------------"


# --- 步骤2: 循环测试 ---
for ITERATION in {1..5}; do
    echo ""
    echo "开始第 $ITERATION 轮测试..."
    
    for RATE in "${REQUEST_RATES[@]}"; do
        CURRENT_TMP_JSON="${TMP_JSON_PREFIX}_${ITERATION}_${RATE}.json"
        
        echo ""
        echo ">>> 正在测试 Request Rate (QPS): $RATE ..."

        # --- 调用 bench_serving，使用数据集文件 ---
        python3 -m sglang.bench_serving_new \
            --backend sglang \
            --host "0.0.0.0" \
            --port 33711 \
            --tokenizer "$MODEL_TOKENIZER_PATH" \
            --dataset-path "$DATASET_FILE" \
            --request-rate "$RATE" \
            --num-prompts "$NUM_PROMPTS" \
            --output-file "$CURRENT_TMP_JSON"\
            --max-concurrency 50

        if [ -s "$CURRENT_TMP_JSON" ]; then
            echo "测试完成，正在将结果追加到 $MASTER_REPORT_FILE"
            cat "$CURRENT_TMP_JSON" >> "$MASTER_REPORT_FILE"
            echo "" >> "$MASTER_REPORT_FILE"
        else
            echo "警告: 第 $ITERATION 轮 QPS $RATE 未能生成结果文件，跳过处理。"
        fi
        
        echo "----------------------------------------------------"
        sleep 5
    done
    
    echo "第 $ITERATION 轮测试完成"
    echo "============================================================"
done

# --- 清理临时文件和数据集文件 ---
echo "所有测试完成。正在清理临时文件..."
rm -f ${TMP_JSON_PREFIX}_*.json
# rm -f "$DATASET_FILE" # 如果想保留数据集用于复现，可以注释掉这行

echo "基准测试全部完成！"
echo "最终报告保存在: $MASTER_REPORT_FILE"
#!/bin/bash

# 记得在另一个terminal首先启动服务器:
# python3 -m sglang.launch_server \
#     --model-path "/data/pretrained_models/Llama-2-7b-hf" \
#     --host "0.0.0.0" \
#     --port 32209 \
#     --max-total-tokens 81920 \
#     --log-level error

# # --- 切换到正确的工作目录 ---
# cd /home/lg/sglang || {
#     echo "错误：无法切换到sglang目录"
#     exit 1
# }

# --- 基准测试配置 ---
REQUEST_RATES=(10 15 20 25 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
PROMPT_LEN=256
OUTPUT_LEN=64
NUM_PROMPTS=2000
MODEL_TOKENIZER_PATH="/data/pretrained_models/Llama-2-7b-hf"

# --- 文件输出配置 ---
RESULTS_DIR="/home/lg/sglang/实验/continuous_batching_fix_length/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_REPORT_FILE="${RESULTS_DIR}/benchmark_report_${TIMESTAMP}.jsonl"
TMP_JSON_PREFIX="${RESULTS_DIR}/tmp_run_results"

# 创建结果目录（如果不存在）
mkdir -p "$RESULTS_DIR"

# --- 脚本开始 ---
echo "开始基准测试..."
echo "当前工作目录: $(pwd)"
echo "本次将测试QPS: ${REQUEST_RATES[@]}"
echo "将进行5次重复测试"
echo "结果将保存到 $MASTER_REPORT_FILE"
echo "----------------------------------------------------"

# --- 外层循环：重复5次 ---
for ITERATION in {1..5}; do
    echo ""
    echo "开始第 $ITERATION 轮测试..."
    
    # --- 内层循环：测试不同QPS ---
    for RATE in "${REQUEST_RATES[@]}"; do
        CURRENT_TMP_JSON="${TMP_JSON_PREFIX}_${ITERATION}_${RATE}.json"
        
        echo ""
        echo ">>> 正在测试 Request Rate (QPS): $RATE ..."

        python3 -m sglang.bench_serving \
            --backend sglang \
            --host "0.0.0.0" \
            --port 33200 \
            --dataset-name random-ids \
            --tokenizer "$MODEL_TOKENIZER_PATH" \
            --random-input-len "$PROMPT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --request-rate "$RATE" \
            --num-prompts "$NUM_PROMPTS" \
            --output-file "$CURRENT_TMP_JSON"

        if [ -s "$CURRENT_TMP_JSON" ]; then
            echo "测试完成，正在将结果追加到 $MASTER_REPORT_FILE"
            cat "$CURRENT_TMP_JSON" >> "$MASTER_REPORT_FILE"
            echo "" >> "$MASTER_REPORT_FILE"  # 添加换行符分隔不同测试结果
        else
            echo "警告: 第 $ITERATION 轮 QPS $RATE 未能生成结果文件，跳过处理。"
        fi
        
        echo "----------------------------------------------------"
        sleep 5  # 在测试之间短暂暂停，让系统冷却
    done
    
    echo "第 $ITERATION 轮测试完成"
    echo "============================================================"
done

# --- 清理临时文件 ---
echo "所有测试完成。正在清理临时文件..."
rm -f ${TMP_JSON_PREFIX}_*.json

echo "基准测试全部完成！"
echo "最终报告保存在: $MASTER_REPORT_FILE"
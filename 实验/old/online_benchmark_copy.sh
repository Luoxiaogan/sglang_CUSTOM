#!/bin/bash

# 记得在另一个terminal首先启动服务器:
# python3 -m sglang.launch_server \
#     --model-path "/home/lg/arc/model" \
#     --host "0.0.0.0" \
#     --port 30000 \
#     --max-total-tokens 81920 \
#     --log-level error


# --- 基准测试配置 ---
REQUEST_RATES=(200 205 210 215 220 225 230 235 240 245 250 255 260 265 270 275 280 285 290 295 300)

PROMPT_LEN=256
OUTPUT_LEN=64
NUM_PROMPTS=2000
MODEL_TOKENIZER_PATH="/home/lg/arc/model"

# --- 文件输出配置 ---
MASTER_REPORT_FILE="master_benchmark_report.jsonl"
TMP_JSON_PREFIX="tmp_run_results"

# --- 脚本开始 ---
echo "开始增量基准测试 (最终简化版)..."
echo "本次将测试QPS: ${REQUEST_RATES[@]}"
echo "结果将追加到 $MASTER_REPORT_FILE"
echo "----------------------------------------------------"

# --- 循环执行基准测试 ---
for RATE in "${REQUEST_RATES[@]}"; do
  CURRENT_TMP_JSON="${TMP_JSON_PREFIX}_${RATE}.json"
  
  echo ""
  echo ">>> 正在测试 Request Rate (QPS): $RATE ..."

  python3 -m sglang.bench_serving \
    --backend sglang \
    --host "0.0.0.0" \
    --port 30000 \
    --dataset-name random \
    --tokenizer "$MODEL_TOKENIZER_PATH" \
    --random-input-len "$PROMPT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --request-rate "$RATE" \
    --num-prompts "$NUM_PROMPTS" \
    --output-file "$CURRENT_TMP_JSON"

  if [ -s "$CURRENT_TMP_JSON" ]; then
    echo "测试完成，正在将结果追加到 $MASTER_REPORT_FILE"
    
    # --- 核心修改：直接复制临时文件的内容，并添加一个换行符 ---
    cat "$CURRENT_TMP_JSON" >> "$MASTER_REPORT_FILE"
    echo "" >> "$MASTER_REPORT_FILE"

  else
    echo "警告: QPS $RATE 未能生成结果文件，跳过处理。"
  fi
  
  echo "----------------------------------------------------"
  sleep 5
done

# --- 清理本次运行产生的临时文件 ---
echo "所有QPS测试完成。正在清理临时文件..."
# rm -f ${TMP_JSON_PREFIX}_*.json

echo "所有操作完成！最终报告位于: $MASTER_REPORT_FILE"
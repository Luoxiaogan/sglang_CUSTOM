#!/bin/bash

# 记得在另一个terminal首先启动服务器:
# python3 -m sglang.launch_server \
# --model-path "/data/pretrained_models/Llama-2-7b-hf" \
# --host "0.0.0.0" \
# --port 32209 \
# --max-total-tokens 40960 \
# --base-gpu-id 1 \
# --log-level error > sglang_server_new.log 2>&1 &

# --- 切换到正确的工作目录 ---
cd /home/lg/sglang || {
    echo "错误：无法切换到sglang目录"
    exit 1
}

# --- 检查服务器是否运行 ---
echo "正在检查服务器状态..."
if ! curl -s "http://0.0.0.0:32209/health" > /dev/null; then
    echo "错误：无法连接到服务器（端口32209）。请确保服务器已启动。"
    exit 1
fi
echo "服务器状态检查通过"

# --- 基准测试配置 ---
REQUEST_RATES=(50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)
NUM_REPEATS=5 # <-- 新增: 设置每个QPS的重复实验次数

PROMPT_LEN=256
OUTPUT_LEN=64
NUM_PROMPTS=2000
MODEL_TOKENIZER_PATH="/data/pretrained_models/Llama-2-7b-hf"

# --- 文件输出配置 ---
RESULTS_DIR="/home/lg/sglang/实验/old/results"
mkdir -p "$RESULTS_DIR"  # 创建结果目录（如果不存在）

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_REPORT_FILE="${RESULTS_DIR}/benchmark_report_${TIMESTAMP}.jsonl"
TMP_JSON_PREFIX="${RESULTS_DIR}/tmp_run_results"

# --- 脚本开始 ---
echo "开始增量基准测试 (每个QPS重复${NUM_REPEATS}次)..."
echo "当前工作目录: $(pwd)"
echo "Python版本: $(python3 --version)"
echo "本次将测试QPS: ${REQUEST_RATES[@]}"
echo "结果将保存到: $MASTER_REPORT_FILE"
echo "----------------------------------------------------"

# --- 循环执行基准测试 ---
for RATE in "${REQUEST_RATES[@]}"; do
  echo ""
  echo ">>> 开始测试 Request Rate (QPS): $RATE (共${NUM_REPEATS}轮)"
  echo "===================================================="

  # --- 新增: 内层循环，用于重复实验 ---
  for i in $(seq 1 $NUM_REPEATS); do
    echo ""
    echo "--> 正在进行第 $i / $NUM_REPEATS 轮测试 (QPS: $RATE)..."

    # --- 修改: 为临时文件添加运行编号，防止被覆盖 ---
    CURRENT_TMP_JSON="${TMP_JSON_PREFIX}_${RATE}_run_${i}.json"

    echo "执行命令: python3 -m sglang.bench_serving ..."
    python3 -m sglang.bench_serving \
      --backend sglang \
      --host "0.0.0.0" \
      --port 32209 \
      --dataset-name random-ids \
      --tokenizer "$MODEL_TOKENIZER_PATH" \
      --random-input-len "$PROMPT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --request-rate "$RATE" \
      --num-prompts "$NUM_PROMPTS" \
      --output-file "$CURRENT_TMP_JSON" \
      --disable-ignore-eos \
      --seed "$i" \
      --disable-tqdm

    # 检查命令执行状态
    if [ $? -ne 0 ]; then
      echo "错误：bench_serving命令执行失败 (QPS: ${RATE}, Run: ${i})"
      continue
    fi

    # 检查并追加结果
    if [ -s "$CURRENT_TMP_JSON" ]; then
      echo "测试完成 (QPS: ${RATE}, Run: ${i})，正在将结果追加到 $MASTER_REPORT_FILE"
      cat "$CURRENT_TMP_JSON" >> "$MASTER_REPORT_FILE"
      echo "" >> "$MASTER_REPORT_FILE"
    else
      echo "警告: QPS $RATE (Run: $i) 未能生成结果文件，跳过处理。"
    fi
    sleep 2 # 每轮测试后短暂休眠
  done

  echo "----------------------------------------------------"
  sleep 5 # 完成一个QPS的所有轮次后，休眠更长时间
done

# --- 清理临时文件 ---
echo "所有测试完成。正在清理临时文件..."
rm -f ${TMP_JSON_PREFIX}_*.json

echo "所有操作完成！"
echo "最终报告保存在: $MASTER_REPORT_FILE"
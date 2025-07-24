#!/bin/bash

# 记得在另一个terminal首先启动服务器:
# python3 -m sglang.launch_server \
# --model-path "/home/lg/arc/model" \
# --host "0.0.0.0" \
# --port 30000 \
# --max-total-tokens 81920 \
# --log-level error

# nohup python3 -m sglang.launch_server \
# --model-path "/home/lg/arc/model" \
# --host "0.0.0.0" \
# --port 31212 \
# --max-total-tokens 40960 \
# --base-gpu-id 1 \
# --log-level error > sglang_server.log 2>&1 &


# --- 在这里配置你的测试参数 ---

# 固定的测试参数
BATCH_SIZE=10
NUM_BATCHES=20
INPUT_LEN=20
OUTPUT_LEN=10
TOKENIZER="/home/lg/arc/model" # 或者使用你本地模型的路径
URL="http://localhost:31212/generate"

# 需要循环测试的请求速率 (用空格隔开)
RATES=(30 5 10 15 20 25 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)

# 输出结果文件名
OUTPUT_FILE="static_batch_results.jsonl"

# --- 脚本主逻辑 ---

# 开始测试前，清空旧的结果文件
echo "Removing old results file: $OUTPUT_FILE"
rm -f $OUTPUT_FILE

echo "Starting benchmark run..."
echo "========================================"
echo "Batch Size: $BATCH_SIZE"
echo "Num Batches: $NUM_BATCHES"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Testing Rates: ${RATES[*]}"
echo "========================================"

# 循环遍历每个rate值
for r in "${RATES[@]}"; do
    echo ""
    echo "----- Running test for rate: $r req/s -----"
    
    # 执行Python脚本，并传入所有参数
    python /home/lg/sglang/static_batch_tester.py \
        --rate "$r" \
        --batch-size "$BATCH_SIZE" \
        --num-batches "$NUM_BATCHES" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --tokenizer "$TOKENIZER" \
        --url "$URL" \
        --output-file "$OUTPUT_FILE"
        
    echo "----- Test for rate: $r finished -----"
done

echo ""
echo "========================================"
echo "All benchmark runs completed."
echo "Results have been saved to $OUTPUT_FILE"
echo "========================================"
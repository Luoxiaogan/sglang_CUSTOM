#!/bin/bash

# 需要测试的batch sizes
BATCH_SIZES=(5 6 8 9 10)

# 基础配置
NUM_BATCHES=20
MEAN_INPUT_LEN=20
MEAN_OUTPUT_LEN=10
VARIANCE=1.0
TOKENIZER="/data/pretrained_models/Llama-2-7b-hf"
URL="http://localhost:31209/generate"
RATES=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

# 遍历每个batch size
for bs in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "============================================"
    echo "Starting tests for batch size: $bs"
    echo "============================================"
    
    # 设置对应的输出文件
    OUTPUT_FILE="/home/lg/sglang/实验/static_length_with_distribution/results_bs=${bs}.jsonl"
    
    echo "Starting benchmark run..."
    echo "========================================"
    echo "Batch Size: $bs"
    echo "Num Batches: $NUM_BATCHES"
    echo "Mean Input Length: $MEAN_INPUT_LEN"
    echo "Mean Output Length: $MEAN_OUTPUT_LEN"
    echo "Variance: $VARIANCE"
    echo "Testing Rates: ${RATES[*]}"
    echo "Output File: $OUTPUT_FILE"
    echo "========================================"

    for i in {1..5}; do
        echo ""
        echo "----- Starting iteration $i -----"
        for r in "${RATES[@]}"; do
            echo ""
            echo "----- Running test for rate: $r req/s -----"
            
            # 生成本次测试的输入输出长度
            read INPUT_LEN OUTPUT_LEN <<< $(python /home/lg/sglang/实验/static_length_with_distribution/length_generator.py \
                --mean-input-len "$MEAN_INPUT_LEN" \
                --mean-output-len "$MEAN_OUTPUT_LEN" \
                --variance "$VARIANCE")
                
            echo "Generated lengths - Input: $INPUT_LEN, Output: $OUTPUT_LEN"
            
            # 执行Python脚本，并传入所有参数
            python /home/lg/sglang/实验/smaller_bs/static_batch_tester.py \
                --rate "$r" \
                --batch-size "$bs" \
                --num-batches "$NUM_BATCHES" \
                --input-len "$INPUT_LEN" \
                --output-len "$OUTPUT_LEN" \
                --tokenizer "$TOKENIZER" \
                --url "$URL" \
                --output-file "$OUTPUT_FILE"
                
            echo "----- Test for rate: $r finished -----"
        done
    done

    echo ""
    echo "========================================"
    echo "Tests completed for batch size: $bs"
    echo "Results have been saved to $OUTPUT_FILE"
    echo "========================================"
done

echo ""
echo "============================================"
echo "All batch size tests completed!"
echo "============================================" 
#!/bin/bash

# =============================================================================
# SGLang Router Gradient Optimization Runner
# =============================================================================
# This script runs the gradient optimizer using configuration from gradient_config.sh
# =============================================================================

# Source configuration file
CONFIG_FILE="/nas/ganluo/sglang/gradient_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

echo "=========================================="
echo "SGLang Router Gradient Optimization"
echo "=========================================="
echo ""

# Process initial probabilities
if [ "$INITIAL_PROBS" = "uniform" ]; then
    # Calculate uniform distribution
    INITIAL_PROB=$(echo "scale=4; 1.0 / $NUM_WORKERS" | bc)
    
    # Build initial probabilities string
    PROB_STRING=""
    for ((i=0; i<$NUM_WORKERS-1; i++)); do
        PROB_STRING="$PROB_STRING $INITIAL_PROB"
    done
    # Last probability to ensure sum = 1.0
    LAST_PROB=$(echo "scale=4; 1.0 - $INITIAL_PROB * ($NUM_WORKERS - 1)" | bc)
    PROB_STRING="$PROB_STRING $LAST_PROB"
else
    # Use specified probabilities
    PROB_STRING="$INITIAL_PROBS"
fi

echo "Configuration Summary:"
echo "----------------------"
echo "Workers: ${WORKERS[@]}"
echo "Initial probabilities:$PROB_STRING"
echo ""
echo "Optimization Settings:"
echo "  Objective: $OBJECTIVE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Perturbation: $PERTURBATION"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Tolerance: $TOLERANCE"
echo ""
echo "Benchmark Settings:"
echo "  Requests per benchmark: $NUM_REQUESTS"
echo "  Request rate: $REQUEST_RATE req/s"
echo "  Dataset: $DATASET"
if [ "$DATASET" = "random" ]; then
    echo "  Input length: $INPUT_LEN ±$(echo "scale=0; $INPUT_LEN * $RANGE_RATIO / 1" | bc | sed 's/^$/0/')%"
    echo "  Output length: $OUTPUT_LEN ±$(echo "scale=0; $OUTPUT_LEN * $RANGE_RATIO / 1" | bc | sed 's/^$/0/')%"
fi
echo ""
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "=========================================="
echo ""

# Build command
CMD="python $OPTIMIZER_SCRIPT"
CMD="$CMD --workers ${WORKERS[@]}"
CMD="$CMD --initial-probs$PROB_STRING"
CMD="$CMD --router-port $ROUTER_PORT"
CMD="$CMD --router-host $ROUTER_HOST"
CMD="$CMD --objective $OBJECTIVE"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --perturbation $PERTURBATION"
CMD="$CMD --max-iterations $MAX_ITERATIONS"
CMD="$CMD --tolerance $TOLERANCE"
CMD="$CMD --num-requests $NUM_REQUESTS"
CMD="$CMD --request-rate $REQUEST_RATE"
CMD="$CMD --input-len $INPUT_LEN"
CMD="$CMD --output-len $OUTPUT_LEN"

echo "Command:"
echo "$CMD"
echo ""

# Confirmation prompt
read -p "Start optimization? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Optimization cancelled."
    exit 0
fi

echo ""
echo "Starting optimization..."
echo "=========================================="

# Execute
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Optimization completed successfully!"
    echo "=========================================="
    echo "Results saved in: $OUTPUT_BASE_DIR/gradient_optimization_*"
    echo ""
    echo "To view results:"
    echo "  ls -la $OUTPUT_BASE_DIR/gradient_optimization_*/"
    echo "  cat $OUTPUT_BASE_DIR/gradient_optimization_*/optimization_results.csv"
else
    echo ""
    echo "=========================================="
    echo "❌ Optimization failed!"
    echo "=========================================="
    echo "Check the error messages above for details."
    exit 1
fi
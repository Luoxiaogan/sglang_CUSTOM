#!/bin/bash

# =============================================================================
# SGLang Router Gradient Optimization Configuration
# =============================================================================
# This file contains all configuration parameters for gradient optimization.
# Modify these values to customize your optimization run.
# =============================================================================

# -------------------------------------------------------------------------
# WORKER CONFIGURATION
# -------------------------------------------------------------------------
# List of worker URLs - modify to match your setup
WORKERS=(
    "http://localhost:30001"    # Worker 1 (e.g., GPU 0)
    "http://localhost:30002"    # Worker 2 (e.g., GPU 1)
    # "http://localhost:30003"    # Worker 3 (e.g., GPU 2)
)

# Router configuration
ROUTER_PORT=29001              # Port for the router service
ROUTER_HOST="0.0.0.0"         # Host for the router service

# -------------------------------------------------------------------------
# INITIAL PROBABILITY DISTRIBUTION
# -------------------------------------------------------------------------
# Initial probabilities for each worker (must sum to 1.0)
# Options:
#   1. "uniform" - Automatically calculate uniform distribution
#   2. Specific values - e.g., "0.5 0.3 0.2"
INITIAL_PROBS="uniform"        # Use "uniform" or specify like "0.5 0.3 0.2"

# Example configurations:
# INITIAL_PROBS="uniform"             # Equal distribution
# INITIAL_PROBS="0.5 0.3 0.2"        # Custom: 50%, 30%, 20%
# INITIAL_PROBS="0.7 0.2 0.1"        # Heavy bias to first worker
# INITIAL_PROBS="0.33 0.33 0.34"     # Nearly uniform

# -------------------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# -------------------------------------------------------------------------
# Optimization objective
# Options:
#   - "maximize_throughput" : Maximize total token throughput (prefill + decode)
#   - "minimize_latency"    : Minimize average request latency
#   - "balanced"           : Weighted combination of throughput and latency
OBJECTIVE="maximize_throughput"

# Learning rate for gradient ascent/descent
# Higher values = faster convergence but less stable
# Lower values = slower convergence but more stable
LEARNING_RATE=0.01             # Typical range: 0.001 to 0.1

# Perturbation size for gradient calculation
# Smaller values = more accurate gradients but more sensitive to noise
# Larger values = less accurate but more robust
PERTURBATION=0.01              # Typical range: 0.001 to 0.05

# Maximum number of optimization iterations
MAX_ITERATIONS=20              # Typical range: 10 to 100

# Convergence tolerance
# Stop when probability changes are smaller than this
TOLERANCE=0.0001               # Typical range: 1e-5 to 1e-3

# Minimum improvement required to continue
# Stop if objective doesn't improve by at least this much
MIN_IMPROVEMENT=0.001          # Typical range: 0.0001 to 0.01

# -------------------------------------------------------------------------
# BENCHMARK CONFIGURATION
# -------------------------------------------------------------------------
# Number of requests per benchmark
# More requests = more stable metrics but longer runtime
NUM_REQUESTS=100               # Typical range: 50 to 500

# Request rate (requests per second)
# Higher rates stress the system more
REQUEST_RATE=20.0              # Typical range: 10 to 100
# REQUEST_RATE="inf"           # Maximum rate (uncomment for stress testing)

# Dataset configuration
DATASET="random"               # Options: "random", "sharegpt", "custom"

# Random dataset parameters
INPUT_LEN=512                  # Average input length in tokens
OUTPUT_LEN=50                  # Average output length in tokens
RANGE_RATIO=0.25              # Length variation (±25%)

# ShareGPT dataset path (if using sharegpt)
# SHAREGPT_PATH="/path/to/sharegpt.json"

# Custom dataset path (if using custom)
# CUSTOM_DATASET_PATH="/path/to/custom_dataset.json"

# -------------------------------------------------------------------------
# BALANCED OBJECTIVE WEIGHTS (only used when OBJECTIVE="balanced")
# -------------------------------------------------------------------------
THROUGHPUT_WEIGHT=0.5          # Weight for throughput in balanced objective
LATENCY_WEIGHT=0.5            # Weight for latency in balanced objective
# Note: These don't need to sum to 1.0

# -------------------------------------------------------------------------
# OUTPUT CONFIGURATION
# -------------------------------------------------------------------------
# Base directory for optimization results
# Each run creates a timestamped subdirectory
OUTPUT_BASE_DIR="/nas/ganluo/sglang"

# Log level for debugging
LOG_LEVEL="INFO"               # Options: "DEBUG", "INFO", "WARN", "ERROR"

# -------------------------------------------------------------------------
# ADVANCED CONFIGURATION
# -------------------------------------------------------------------------
# Timeout for each benchmark run (seconds)
BENCHMARK_TIMEOUT=300          # 5 minutes default

# Wait time for router startup (seconds)
ROUTER_STARTUP_WAIT=5          # Time to wait after starting router

# Number of iterations without improvement before stopping
NO_IMPROVEMENT_LIMIT=3         # Stop after this many iterations without improvement

# -------------------------------------------------------------------------
# EXPERIMENTAL FEATURES
# -------------------------------------------------------------------------
# Enable adaptive learning rate (reduces learning rate over time)
ADAPTIVE_LEARNING=false        # true or false

# Learning rate decay factor (if adaptive learning is enabled)
LEARNING_DECAY=0.95           # Multiply learning rate by this each iteration

# Enable momentum (use gradient history for smoother updates)
USE_MOMENTUM=false            # true or false

# Momentum factor (if momentum is enabled)
MOMENTUM_FACTOR=0.9           # Weight for previous gradient

# -------------------------------------------------------------------------
# SYSTEM PATHS - Usually don't need to change these
# -------------------------------------------------------------------------
# Path to the main start_router.py script
ROUTER_SCRIPT="/nas/ganluo/sglang/start_router.py"

# Path to the send_req.py benchmark script
BENCHMARK_SCRIPT="/nas/ganluo/sglang/send_req.py"

# Path to the gradient optimizer script
OPTIMIZER_SCRIPT="/nas/ganluo/sglang/gradient_optimizer.py"


# -------------------------------------------------------------------------
# VALIDATION - Don't modify below this line
# -------------------------------------------------------------------------
# Calculate number of workers
NUM_WORKERS=${#WORKERS[@]}

# Validate configuration
if [ "$NUM_WORKERS" -eq 0 ]; then
    echo "Error: No workers configured"
    exit 1
fi

echo "Configuration loaded:"
echo "  Workers: $NUM_WORKERS"
echo "  Objective: $OBJECTIVE"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Requests per benchmark: $NUM_REQUESTS"
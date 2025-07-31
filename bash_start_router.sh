#!/bin/bash

# SGLang Router 启动脚本
# 用途：启动不同配置的路由器进行测试

# =============================================================================
# 配置参数 - 根据需要修改这些参数
# =============================================================================

# 路由策略选择
# 可选值: cache_aware, round_robin, random, power_of_two, marginal_utility, marginal_utility_recorder
POLICY="round_robin"

# Worker节点配置
# 修改为你的实际worker地址和端口
WORKERS=(
    "http://localhost:30001"    # GPU 2
    "http://localhost:30002"    # GPU 3
    "http://localhost:30003"    # GPU 3
)

# 路由器配置
ROUTER_PORT=29001              # 路由器监听端口
ROUTER_HOST="0.0.0.0"          # 路由器监听地址

# 请求追踪配置
ENABLE_TRACKING=true           # 是否启用请求追踪
MAX_TRACE_ENTRIES=100000       # 最大追踪条目数
TRACE_TTL=3600                 # 追踪记录保留时间（秒）

# 日志级别: DEBUG, INFO, WARN, ERROR
LOG_LEVEL="INFO"

# Marginal Utility Recorder 输出目录配置（仅在使用 marginal_utility_recorder 策略时生效）
# 设置为空字符串将使用默认路径 /tmp/marginal_utility_metrics
# 
# ⚠️ 注意：这是路由器内部决策记录，与 send_req.py 生成的请求记录不同
# - 本文件记录：路由器的决策过程（梯度、评分、选择原因等）
# - send_req.py 记录：客户端视角的请求执行情况（延迟、吞吐量等）
# 
# CSV 文件内容说明：
# - 路由决策记录：记录每次选择 worker 时的梯度、评分等信息（selection_reason="gradient_based"）
# - 请求完成记录：记录请求完成后的实际性能指标（selection_reason="completion_record"）
# 两种记录都保存在同一个 CSV 文件中，通过 selection_reason 字段区分
MARGINAL_UTILITY_OUTPUT_DIR="/nas/ganluo/sglang/test_results_hetero"

# GPU映射配置（可选）
# 格式: '{"端口号": "GPU设备"}'
# PORT_GPU_MAPPING='{"40005": "cuda:2", "40006": "cuda:3"}'

# =============================================================================
# 预设配置 - 取消注释以使用预设配置
# =============================================================================

# # 配置1: 高性能随机路由
# POLICY="random"
# WORKERS=("http://localhost:40005" "http://localhost:40006")
# ROUTER_PORT=40009

# # 配置2: 缓存感知路由（适合重复请求多的场景）
# POLICY="cache_aware"
# WORKERS=("http://localhost:40005" "http://localhost:40006")
# ROUTER_PORT=40009

# # 配置3: 轮询路由（负载最均衡）
# POLICY="round_robin"
# WORKERS=("http://localhost:40005" "http://localhost:40006")
# ROUTER_PORT=40009

# # 配置4: 四GPU大规模测试
# POLICY="random"
# WORKERS=("http://localhost:40005" "http://localhost:40006" "http://localhost:40007" "http://localhost:40008")
# ROUTER_PORT=40009

# # 配置5: Marginal Utility 策略（高性能，无记录）
# POLICY="marginal_utility"
# WORKERS=("http://localhost:40005" "http://localhost:40006")
# ROUTER_PORT=40009

# # 配置6: Marginal Utility Recorder 策略（带性能指标记录）
# POLICY="marginal_utility_recorder"
# WORKERS=("http://localhost:40005" "http://localhost:40006")
# ROUTER_PORT=40009
# # 记录文件将保存在 /tmp/marginal_utility_metrics/ 目录下

# =============================================================================
# 执行脚本 - 通常不需要修改以下内容
# =============================================================================

echo "=========================================="
echo "SGLang Router 启动配置"
echo "=========================================="
echo "路由策略: $POLICY"
echo "Worker节点: ${WORKERS[@]}"
echo "路由器地址: $ROUTER_HOST:$ROUTER_PORT"
echo "请求追踪: $ENABLE_TRACKING"
echo "日志级别: $LOG_LEVEL"
echo "=========================================="
echo ""

# 构建命令
CMD="python /nas/ganluo/sglang/start_router.py"
CMD="$CMD --policy $POLICY"
CMD="$CMD --host $ROUTER_HOST"
CMD="$CMD --port $ROUTER_PORT"
CMD="$CMD --log-level $LOG_LEVEL"

# 添加所有worker
# for worker in "${WORKERS[@]}"; do
#     CMD="$CMD --workers $worker"
# done
# 需要一次性添加所有worker
CMD="$CMD --workers ${WORKERS[@]}"

# 添加追踪配置
if [ "$ENABLE_TRACKING" = true ]; then
    CMD="$CMD --enable-tracking"
    CMD="$CMD --max-trace-entries $MAX_TRACE_ENTRIES"
    CMD="$CMD --trace-ttl $TRACE_TTL"
fi

# 添加GPU映射（如果设置了）
if [ ! -z "$PORT_GPU_MAPPING" ]; then
    CMD="$CMD --port-gpu-mapping '$PORT_GPU_MAPPING'"
fi

# 添加 Marginal Utility 输出目录（如果是相应策略且设置了目录）
if [ "$POLICY" = "marginal_utility_recorder" ] && [ ! -z "$MARGINAL_UTILITY_OUTPUT_DIR" ]; then
    CMD="$CMD --marginal-utility-output-dir $MARGINAL_UTILITY_OUTPUT_DIR"
fi

# 显示完整命令
echo "执行命令:"
echo "$CMD"
echo ""

# 执行
eval $CMD
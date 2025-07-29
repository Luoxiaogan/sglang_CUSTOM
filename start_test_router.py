from sglang_router_rs import Router, PolicyType
import time
import json

# ====== 端口到GPU映射配置 ======
# 请根据您的实际部署情况修改此映射
PORT_GPU_MAPPING = {
    60005: "cuda:2",  # 端口 30005 对应 GPU 2
    60006: "cuda:3",  # 端口 30006 对应 GPU 3
    # 添加更多映射...
}

# 保存映射到文件，供测试脚本使用
with open("/tmp/sglang_port_gpu_mapping.json", "w") as f:
    json.dump(PORT_GPU_MAPPING, f)
print("端口-GPU映射已保存到: /tmp/sglang_port_gpu_mapping.json")
print("映射关系:")
for port, gpu in PORT_GPU_MAPPING.items():
    print(f"  端口 {port} -> {gpu}")
print()

# 创建启用请求追踪的路由器
router = Router(
    worker_urls=[
        "http://localhost:60005",
        "http://localhost:60006"
    ],
    policy=PolicyType.CacheAware,
    port=60009,
    enable_request_tracking=True,  # 启用请求追踪
    max_trace_entries=100000,       # 最多保存10万条记录
    trace_ttl_seconds=3600,         # 记录保存1小时
    log_level="INFO"
)

print("启动路由器（启用请求追踪）...")
try:
    router.start()
except KeyboardInterrupt:
    print("\n路由器已停止")
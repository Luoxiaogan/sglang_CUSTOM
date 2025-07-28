from sglang_router_rs import Router, PolicyType
import time

# 创建启用请求追踪的路由器
router = Router(
    worker_urls=[
        "http://localhost:30005",
        "http://localhost:30006"
    ],
    policy=PolicyType.CacheAware,
    port=30009,
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
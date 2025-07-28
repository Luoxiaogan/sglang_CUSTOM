# SGLang Router 请求追踪功能修复指南

## 修复概述

请求追踪功能已经修复。问题在于代码尝试使用类型转换（downcast）来访问 Router 的 request_tracker，但由于类型系统的限制，这种方法失败了。

修复方案是在 `RouterTrait` 接口中添加了 `request_tracker()` 方法，让所有路由器实现都可以暴露其请求追踪器。

## 已修改的文件

1. `/src/routers/mod.rs` - 在 RouterTrait 中添加了 request_tracker() 方法
2. `/src/routers/router.rs` - 在 Router 中实现了 request_tracker() 方法
3. `/src/server.rs` - 修改了所有请求追踪端点，使用新的 trait 方法

## 编译步骤

```bash
# 1. 进入 sgl-router 目录
cd /nas/ganluo/sglang_CUSTOM/sgl-router

# 2. 重新编译并安装（使用 verbose 模式查看进度）
pip install -e . -v

# 如果编译失败，可以先清理缓存
cargo clean
pip cache purge
pip install -e . -v
```

## 测试步骤

### 1. 启动 SGLang 服务器

```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30005 \
--base-gpu-id 2

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30006 \
--base-gpu-id 3
```

### 2. 启动路由器（启用请求追踪）

```bash
# 终端 3
python /nas/ganluo/sglang_CUSTOM/start_test_router.py
```

### 3. 测试请求追踪功能

```bash
# 终端 4
# 运行测试脚本
python /nas/ganluo/sglang_CUSTOM/test_request_tracking.py
```

### 4. 手动测试 API

```bash
# 检查追踪统计
curl http://localhost:30009/v1/traces/stats

# 获取最近的请求
curl http://localhost:30009/v1/traces?limit=10

# 发送测试请求
curl -X POST http://localhost:30009/generate \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: test-manual-123" \
  -d '{
    "text": "Hello world",
    "sampling_params": {
      "max_new_tokens": 10,
      "temperature": 0.7
    }
  }'

# 查询特定请求
curl http://localhost:30009/v1/traces/test-manual-123
```

## 预期结果

1. **追踪统计** (`/v1/traces/stats`) 应返回：
   ```json
   {
     "total_requests": 5,
     "active_traces": 5,
     "requests_per_node": {
       "worker-0": 3,
       "worker-1": 2
     },
     "requests_per_status": {
       "completed": 5
     }
   }
   ```

2. **单个请求追踪** (`/v1/traces/{request_id}`) 应返回：
   ```json
   {
     "request_id": "test-req-123",
     "node_id": "worker-0",
     "worker_url": "http://localhost:30005",
     "routing_policy": "cache_aware",
     "status": "completed",
     "timestamp": "2025-07-28T10:26:31.123Z",
     "cache_hit_rate": 0.8,
     "queue_size": 0
   }
   ```

## 故障排查

1. **如果编译失败**
   - 确保已安装所有依赖：`conda install -c conda-forge rust pkg-config openssl`
   - 检查 Rust 版本：`rustc --version`（需要 1.70.0 或更高）

2. **如果 API 仍返回 404**
   - 确认路由器启动时显示了 "Initializing request tracker"
   - 检查 `start_test_router.py` 中的 `enable_request_tracking=True`

3. **如果返回 "Request tracking is not enabled"**
   - 确认使用了最新编译的版本
   - 重启所有服务

## 注意事项

- 请求追踪会占用内存，默认保存最多 100,000 条记录
- 记录默认保存 1 小时（3600 秒）
- 可以在启动路由器时调整这些参数
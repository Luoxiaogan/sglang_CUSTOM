# SGLang Router 请求追踪功能 - 部署与测试指南

## 概述

我们已经在 sgl-router 中实现了原生的请求追踪功能，可以记录每个请求被路由到哪个具体的 node_id。这个功能包括：

1. **请求追踪模块** (`request_tracker.rs`)：
   - 使用 HashMap 存储请求追踪信息
   - 支持 TTL 过期和 LRU 清理
   - 线程安全的读写锁实现

2. **配置支持**：
   - 在 RouterConfig 中添加了 `request_tracking` 配置
   - 支持启用/禁用、最大条目数、TTL 等设置

3. **路由集成**：
   - 在路由决策时记录请求信息
   - 自动将 worker_url 映射到 node_id

4. **API 端点**：
   - GET `/v1/traces/{request_id}` - 查询单个请求
   - POST `/v1/traces/batch` - 批量查询
   - GET `/v1/traces` - 列出最近的请求
   - GET `/v1/traces/stats` - 获取统计信息

## 服务器部署步骤

### 1. 上传代码到服务器
```bash
# 在本地打包代码
cd /Users/luogan/Code/sglang
tar -czf sgl-router-tracking.tar.gz sgl-router/

# 上传到服务器
scp sgl-router-tracking.tar.gz user@server:~/

# 在服务器上解压
ssh user@server
tar -xzf sgl-router-tracking.tar.gz
```

### 2. 编译 sgl-router
```bash
cd sgl-router

# 安装 Rust（如果没有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 编译项目
cargo build --release

# 编译 Python 绑定（如果需要）
maturin build --release
```

### 3. 安装到 Python 环境
```bash
# 找到编译好的 wheel 文件
ls target/wheels/

# 安装
pip install target/wheels/sglang_router_rs-*.whl --force-reinstall
```

## 测试步骤

### 1. 创建测试脚本

创建 `test_request_tracking.py`：

```python
import asyncio
import aiohttp
import json
from datetime import datetime
import time

async def test_request_tracking():
    """测试请求追踪功能"""
    
    # 配置
    ROUTER_URL = "http://localhost:30001"
    MODEL_PATH = "meta-llama/Llama-2-7b-hf"
    
    print("=== SGLang Router 请求追踪测试 ===\n")
    
    # 1. 检查路由器健康状态
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ROUTER_URL}/health") as resp:
            if resp.status != 200:
                print("❌ 路由器不健康")
                return
            print("✅ 路由器健康检查通过")
    
    # 2. 发送一些测试请求
    print("\n发送测试请求...")
    request_ids = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            request_id = f"test-req-{int(time.time() * 1000)}-{i}"
            
            # 发送生成请求
            data = {
                "text": f"Test prompt {i}: Hello, world!",
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.7
                },
                "stream": False
            }
            
            headers = {
                "X-Request-Id": request_id  # 自定义请求ID
            }
            
            try:
                async with session.post(
                    f"{ROUTER_URL}/generate",
                    json=data,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        print(f"✅ 请求 {i+1} 发送成功: {request_id}")
                        request_ids.append(request_id)
                    else:
                        print(f"❌ 请求 {i+1} 失败: {resp.status}")
            except Exception as e:
                print(f"❌ 请求 {i+1} 出错: {e}")
            
            await asyncio.sleep(0.1)  # 小延迟
    
    # 3. 等待请求处理
    print("\n等待请求处理...")
    await asyncio.sleep(2)
    
    # 4. 查询请求追踪信息
    print("\n查询请求追踪信息...")
    
    async with aiohttp.ClientSession() as session:
        # 查询单个请求
        if request_ids:
            request_id = request_ids[0]
            async with session.get(f"{ROUTER_URL}/v1/traces/{request_id}") as resp:
                if resp.status == 200:
                    trace = await resp.json()
                    print(f"\n单个请求追踪信息:")
                    print(f"  请求ID: {trace['request_id']}")
                    print(f"  节点ID: {trace['node_id']}")
                    print(f"  Worker URL: {trace['worker_url']}")
                    print(f"  路由策略: {trace['routing_policy']}")
                    print(f"  状态: {trace['status']}")
                    print(f"  时间戳: {trace['timestamp']}")
                else:
                    print(f"❌ 无法获取请求追踪: {resp.status}")
        
        # 批量查询
        if len(request_ids) > 1:
            batch_data = {"request_ids": request_ids[:3]}
            async with session.post(
                f"{ROUTER_URL}/v1/traces/batch",
                json=batch_data
            ) as resp:
                if resp.status == 200:
                    traces = await resp.json()
                    print(f"\n批量查询结果: 找到 {len(traces)} 个请求")
                    for trace in traces:
                        print(f"  - {trace['request_id']} -> {trace['node_id']}")
        
        # 获取最近的请求
        async with session.get(
            f"{ROUTER_URL}/v1/traces?limit=10"
        ) as resp:
            if resp.status == 200:
                recent_traces = await resp.json()
                print(f"\n最近的请求: {len(recent_traces)} 个")
                
        # 获取统计信息
        async with session.get(f"{ROUTER_URL}/v1/traces/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"\n追踪统计信息:")
                print(f"  总请求数: {stats['total_requests']}")
                print(f"  活跃请求数: {stats['active_traces']}")
                print(f"  节点分布:")
                for node_id, count in stats['requests_per_node'].items():
                    print(f"    - {node_id}: {count} 个请求")
                print(f"  状态分布:")
                for status, count in stats['requests_per_status'].items():
                    print(f"    - {status}: {count} 个请求")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_request_tracking())
```

### 2. 启动测试环境

创建 `start_test_router.py`：

```python
from sglang_router_rs import Router, PolicyType
import time

# 创建启用请求追踪的路由器
router = Router(
    worker_urls=[
        "http://localhost:30002",
        "http://localhost:30003"
    ],
    policy=PolicyType.CacheAware,
    port=30001,
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
```

### 3. 启动 SGLang 服务器

在两个不同的终端中启动 SGLang 服务器：

```bash
# GPU 0 (终端 1)
python -m sglang.launch_server \
    --model-path /nas/models/Meta-Llama-3-8B-Instruct \
    --port 30005 \
    --gpu-id 2 \
    --model-alias gpu_0

# GPU 1 (终端 2)  
python -m sglang.launch_server \
    --model-path /nas/models/Meta-Llama-3-8B-Instruct \
    --port 30006 \
    --gpu-id 3 \
    --model-alias gpu_1
```

**这里的启动方法有问题, 你对于sglang_api的理解出错了, 正确的方法是:**
````bash
# GPU 2 (终端 1)
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30005 \
--base-gpu-id 2

# GPU 3 (终端 2)
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30006 \
--base-gpu-id 3
```


### 4. 运行测试

```bash
# 启动路由器
python start_test_router.py

# 在另一个终端运行测试
python test_request_tracking.py
```

## 集成到 sglang_test_framework

更新 `routing_test.py` 以使用新的追踪功能：

```python
# 在 RoutingTest 类中添加方法
async def fetch_request_traces(self, request_ids: List[str]) -> Dict[str, Any]:
    """获取请求追踪信息"""
    async with aiohttp.ClientSession() as session:
        # 批量查询
        data = {"request_ids": request_ids}
        async with session.post(
            f"http://localhost:{self.config.base_port}/v1/traces/batch",
            json=data
        ) as resp:
            if resp.status == 200:
                traces = await resp.json()
                return {trace['request_id']: trace for trace in traces}
    return {}

# 在 analyze_results 中使用
traces = await self.fetch_request_traces(request_ids)
for req in results:
    if req['req_id'] in traces:
        trace = traces[req['req_id']]
        req['gpu_id'] = trace['node_id']
        req['worker_url'] = trace['worker_url']
```

## 验证功能

1. **检查 API 响应**：
   - 访问 `http://localhost:30001/v1/traces/stats` 查看统计
   - 使用 curl 测试各个端点

2. **查看 CSV 输出**：
   - 确认 gpu_id 列已正确填充
   - 验证每个请求都有对应的节点分配

3. **性能影响**：
   - 比较启用/禁用追踪的性能差异
   - 监控内存使用情况

## 故障排查

1. **编译错误**：
   - 确保 Rust 版本 >= 1.70
   - 检查所有依赖是否正确安装

2. **追踪数据为空**：
   - 确认 `enable_request_tracking=True`
   - 检查路由器日志中的追踪相关消息

3. **API 返回 501**：
   - 确认使用的是更新后的路由器
   - 检查路由器类型是否正确

## 代码变更总结

### 新增文件
- `src/request_tracker.rs` - 请求追踪核心模块

### 修改文件
- `src/lib.rs` - 添加请求追踪模块和配置
- `src/config/types.rs` - 添加 RequestTrackingConfig
- `src/routers/router.rs` - 集成请求追踪到路由逻辑
- `src/routers/factory.rs` - 支持创建带追踪的路由器
- `src/server.rs` - 添加 4 个新的 API 端点
- `Cargo.toml` - 添加 chrono 依赖（如需要）

这个实现提供了完整的请求级追踪功能，可以准确记录每个请求被路由到哪个节点，并通过 REST API 提供查询接口。
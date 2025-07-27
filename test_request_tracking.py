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
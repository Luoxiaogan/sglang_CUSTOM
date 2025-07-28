import asyncio
import aiohttp
import json
from datetime import datetime
import time

async def test_request_tracking():
    """测试请求追踪功能（修复版）"""
    
    # 配置
    ROUTER_URL = "http://localhost:40009"
    MODEL_PATH = "/nas/models/Meta-Llama-3-8B-Instruct"
    
    print("=== SGLang Router 请求追踪测试 ===\n")
    
    # 1. 检查路由器健康状态
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ROUTER_URL}/health") as resp:
            if resp.status != 200:
                print("❌ 路由器不健康")
                return
            print("✅ 路由器健康检查通过")
    
    # 2. 获取当前的请求数量（用于后续对比）
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ROUTER_URL}/v1/traces?limit=1") as resp:
            if resp.status == 200:
                initial_traces = await resp.json()
                initial_count = len(initial_traces)
            else:
                initial_count = 0
    
    # 3. 发送测试请求
    print("\n发送测试请求...")
    sent_requests = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            # 发送生成请求（不使用自定义ID，让系统自动生成）
            data = {
                "text": f"Test prompt {i}: Hello, world!",
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.7
                },
                "stream": False
            }
            
            try:
                start_time = time.time()
                async with session.post(
                    f"{ROUTER_URL}/generate",
                    json=data
                ) as resp:
                    if resp.status == 200:
                        result = json.loads(await resp.text())
                        print(f"✅ 请求 {i+1} 发送成功")
                        sent_requests.append({
                            "index": i,
                            "time": start_time,
                            "result": result
                        })
                    else:
                        print(f"❌ 请求 {i+1} 失败: {resp.status}")
            except Exception as e:
                print(f"❌ 请求 {i+1} 出错: {e}")
            
            await asyncio.sleep(0.1)  # 小延迟
    
    # 4. 等待请求处理
    print("\n等待请求处理...")
    await asyncio.sleep(2)
    
    # 5. 获取最新的请求追踪信息
    print("\n查询请求追踪信息...")
    
    async with aiohttp.ClientSession() as session:
        # 获取最近的请求
        async with session.get(
            f"{ROUTER_URL}/v1/traces?limit=10"
        ) as resp:
            if resp.status == 200:
                recent_traces = await resp.json()
                print(f"\n找到 {len(recent_traces)} 个最近的请求")
                
                # 显示最新的几个请求
                for i, trace in enumerate(recent_traces[:5]):
                    print(f"\n请求 {i+1}:")
                    print(f"  请求ID: {trace['request_id']}")
                    print(f"  节点: {trace['node_id']} ({trace['worker_url']})")
                    print(f"  路由策略: {trace['routing_policy']}")
                    print(f"  状态: {trace['status']}")
                    print(f"  输入tokens: {trace.get('input_tokens', 'N/A')}")
                    print(f"  完成时间: {trace.get('completion_time', 'N/A')}")
            else:
                print(f"❌ 无法获取请求列表: {resp.status}")
        
        # 获取统计信息
        async with session.get(f"{ROUTER_URL}/v1/traces/stats") as resp:
            if resp.status == 200:
                try:
                    stats = json.loads(await resp.text())
                    print(f"\n追踪统计信息:")
                    
                    # 安全地访问字段，使用 get() 方法
                    if 'total_requests' in stats:
                        print(f"  总请求数: {stats['total_requests']}")
                    elif 'total_traces' in stats:
                        print(f"  总请求数: {stats['total_traces']}")
                    
                    if 'active_traces' in stats:
                        print(f"  活跃请求数: {stats['active_traces']}")
                    elif 'active_requests' in stats:
                        print(f"  活跃请求数: {stats['active_requests']}")
                    
                    if 'requests_per_node' in stats:
                        print(f"  节点分布:")
                        for node_id, count in stats['requests_per_node'].items():
                            print(f"    - {node_id}: {count} 个请求")
                    elif 'traces_per_node' in stats:
                        print(f"  节点分布:")
                        for node_id, count in stats['traces_per_node'].items():
                            print(f"    - {node_id}: {count} 个请求")
                    
                    if 'requests_per_status' in stats:
                        print(f"  状态分布:")
                        for status, count in stats['requests_per_status'].items():
                            print(f"    - {status}: {count} 个请求")
                    elif 'traces_per_status' in stats:
                        print(f"  状态分布:")
                        for status, count in stats['traces_per_status'].items():
                            print(f"    - {status}: {count} 个请求")
                    
                    # 如果字段名都不匹配，打印实际的响应结构
                    available_keys = list(stats.keys())
                    if not any(key in stats for key in ['total_requests', 'total_traces', 'active_traces', 
                                                         'active_requests', 'requests_per_node', 'traces_per_node',
                                                         'requests_per_status', 'traces_per_status']):
                        print(f"\n实际的统计数据结构: {available_keys}")
                        print(f"原始数据: {json.dumps(stats, indent=2)}")
                        
                except Exception as e:
                    print(f"\n解析统计信息时出错: {e}")
            else:
                # 注意：当前实现中 /v1/traces/stats 实际上被当作 /v1/traces/{request_id}
                # 所以会返回 404
                print(f"\n注意：统计端点可能未正确实现（返回 {resp.status}）")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_request_tracking())
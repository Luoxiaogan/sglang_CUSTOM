import asyncio
import aiohttp
import json
from datetime import datetime
import time
import os

async def test_request_tracking():
    """测试请求追踪功能（V2版本）"""
    
    # 配置
    ROUTER_URL = "http://localhost:30009"
    MODEL_PATH = "/nas/models/Meta-Llama-3-8B-Instruct"
    
    # 尝试加载端口-GPU映射
    port_gpu_mapping = {}
    mapping_file = "/tmp/sglang_port_gpu_mapping.json"
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r") as f:
                # 转换键为整数
                raw_mapping = json.load(f)
                port_gpu_mapping = {int(k): v for k, v in raw_mapping.items()}
            print(f"已加载端口-GPU映射: {port_gpu_mapping}")
        except Exception as e:
            print(f"警告：无法加载映射文件 {mapping_file}: {e}")
    else:
        print(f"提示：未找到映射文件 {mapping_file}，将只显示端口号")
    
    print("\n=== SGLang Router 请求追踪测试 V2 ===\n")
    
    # 1. 检查路由器健康状态
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ROUTER_URL}/health") as resp:
            if resp.status != 200:
                print("❌ 路由器不健康")
                return
            print("✅ 路由器健康检查通过")
    
    # 2. 发送测试请求
    print("\n发送测试请求...")
    sent_requests = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            # 发送生成请求
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
                        # 不强制要求JSON content-type，手动解析
                        text = await resp.text()
                        try:
                            result = json.loads(text)
                            print(f"✅ 请求 {i+1} 发送成功")
                            sent_requests.append({
                                "index": i,
                                "time": start_time,
                                "result": result
                            })
                        except json.JSONDecodeError:
                            print(f"⚠️  请求 {i+1} 返回非JSON响应: {text[:100]}...")
                    else:
                        print(f"❌ 请求 {i+1} 失败: {resp.status}")
            except Exception as e:
                print(f"❌ 请求 {i+1} 出错: {e}")
            
            await asyncio.sleep(0.1)  # 小延迟
    
    # 3. 等待请求处理
    print("\n等待请求处理...")
    await asyncio.sleep(2)
    
    # 4. 获取最新的请求追踪信息
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
                    # 从 worker_url 提取端口号
                    port = int(trace['worker_url'].split(':')[-1].strip('/'))
                    
                    # 获取GPU映射（如果有）
                    gpu_info = port_gpu_mapping.get(port, f"端口{port}")
                    
                    print(f"\n请求 {i+1}:")
                    print(f"  请求ID: {trace['request_id']}")
                    print(f"  节点: {trace['node_id']} -> {gpu_info}")
                    print(f"  Worker URL: {trace['worker_url']}")
                    print(f"  路由策略: {trace['routing_policy']}")
                    print(f"  状态: {trace['status']}")
                    print(f"  输入tokens: {trace.get('input_tokens', 'N/A')}")
                    print(f"  完成时间: {trace.get('completion_time', 'N/A')}")
            else:
                print(f"❌ 无法获取请求列表: {resp.status}")
        
        # 获取统计信息
        async with session.get(f"{ROUTER_URL}/v1/traces/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"\n追踪统计信息:")
                # 使用实际返回的字段名
                print(f"  总追踪数: {stats.get('total_traces', 'N/A')}")
                
                # 配置信息
                if 'config' in stats:
                    print(f"  配置信息:")
                    print(f"    - 已启用: {stats['config'].get('enabled', 'N/A')}")
                    print(f"    - 最大条目: {stats['config'].get('max_entries', 'N/A')}")
                    print(f"    - TTL: {stats['config'].get('ttl_seconds', 'N/A')} 秒")
                
                # 节点分布（显示端口和GPU信息）
                if 'node_distribution' in stats:
                    print(f"  节点分布:")
                    # 从node_id推算端口号
                    for node_id, count in stats['node_distribution'].items():
                        # 假设格式是 gpu_N，提取N并计算端口
                        if node_id.startswith("gpu_"):
                            gpu_index = int(node_id.split("_")[1])
                            port = 30001 + gpu_index
                            gpu_info = port_gpu_mapping.get(port, f"端口{port}")
                            print(f"    - {node_id} -> {gpu_info}: {count} 个请求")
                        else:
                            print(f"    - {node_id}: {count} 个请求")
                
                # 状态分布
                if 'status_distribution' in stats:
                    print(f"  状态分布:")
                    for status, count in stats['status_distribution'].items():
                        print(f"    - {status}: {count} 个请求")
                
                # 路由分布
                if 'route_distribution' in stats:
                    print(f"  路由分布:")
                    for route, count in stats['route_distribution'].items():
                        print(f"    - {route}: {count} 个请求")
                
                # 时间范围
                if 'oldest_trace' in stats:
                    print(f"  最早请求: {stats['oldest_trace']}")
                if 'newest_trace' in stats:
                    print(f"  最新请求: {stats['newest_trace']}")
                    
            else:
                print(f"\n❌ 无法获取统计信息: {resp.status}")

    print("\n" + "="*50)
    print("测试完成！")
    print("\n提示：节点ID是基于端口号生成的，例如：")
    print("- gpu_4 = 端口 30005 (30005 - 30001 = 4)")
    print("- gpu_5 = 端口 30006 (30006 - 30001 = 5)")
    print("这是设计特性，便于在多节点环境中进行稳定的标识和路由。")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_request_tracking())
#!/usr/bin/env python3
"""
专门测试路由器(60009)的queue时间戳功能
确保使用正确的API格式并提供详细诊断
"""

import asyncio
import json
import time
import aiohttp
import argparse
from datetime import datetime


async def test_api_format(session, url):
    """测试API格式，确定使用哪种格式"""
    print("\n🔍 检测API格式...")
    
    # 测试新格式 (text + sampling_params)
    payload_new = {
        "text": "Hello",
        "sampling_params": {
            "max_new_tokens": 5,
            "temperature": 0.1
        },
        "stream": False
    }
    
    # 测试旧格式 (prompt)
    payload_old = {
        "prompt": "Hello", 
        "max_tokens": 5,
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        # 尝试新格式
        async with session.post(url + "/generate", json=payload_new) as response:
            result = await response.json()
            if "error" not in result or "text" not in str(result.get("error", {})):
                print("✅ 使用新API格式 (text + sampling_params)")
                return "new"
    except:
        pass
    
    try:
        # 尝试旧格式
        async with session.post(url + "/generate", json=payload_old) as response:
            result = await response.json()
            if "error" not in result:
                print("✅ 使用旧API格式 (prompt)")
                return "old"
    except:
        pass
    
    print("❌ 无法确定API格式")
    return None


async def send_request_with_format(session, url, prompt, api_format):
    """根据API格式发送请求"""
    if api_format == "new":
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 50,
                "temperature": 0.1
            },
            "stream": False
        }
    else:  # old format
        payload = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.1,
            "stream": False
        }
    
    start_time = time.time()
    async with session.post(url + "/generate", json=payload) as response:
        result = await response.json()
        end_time = time.time()
        result['_request_time'] = end_time - start_time
        return result


async def analyze_timestamps(result, request_id=None):
    """分析响应中的时间戳"""
    prefix = f"请求 {request_id}: " if request_id is not None else ""
    
    if "error" in result:
        print(f"{prefix}❌ 错误: {result['error']}")
        return False
    
    if "meta_info" not in result:
        print(f"{prefix}❌ 响应中没有meta_info")
        return False
    
    meta = result["meta_info"]
    timestamps = {
        "server_created_time": meta.get("server_created_time"),
        "queue_time_start": meta.get("queue_time_start"),
        "queue_time_end": meta.get("queue_time_end"),
        "server_first_token_time": meta.get("server_first_token_time")
    }
    
    # 检查时间戳
    all_present = True
    for name, value in timestamps.items():
        if value is None:
            print(f"{prefix}❌ {name}: None/Missing")
            all_present = False
        else:
            print(f"{prefix}✅ {name}: {value:.6f}")
    
    # 如果所有时间戳都存在，计算时间间隔
    if timestamps["queue_time_start"] and timestamps["queue_time_end"]:
        queue_duration = timestamps["queue_time_end"] - timestamps["queue_time_start"]
        print(f"{prefix}⏱️  纯排队时间: {queue_duration*1000:.2f}ms")
    
    if timestamps["server_created_time"] and timestamps["queue_time_start"]:
        tokenize_time = timestamps["queue_time_start"] - timestamps["server_created_time"]
        print(f"{prefix}⏱️  Tokenize时间: {tokenize_time*1000:.2f}ms")
    
    if timestamps["server_created_time"] and timestamps["server_first_token_time"]:
        total_server_time = timestamps["server_first_token_time"] - timestamps["server_created_time"]
        print(f"{prefix}⏱️  总服务器时间: {total_server_time*1000:.2f}ms")
    
    # 检查时间戳的合理性
    if all(v is not None for v in timestamps.values()):
        # 检查时间戳顺序
        if not (timestamps["server_created_time"] <= timestamps["queue_time_start"] <= 
                timestamps["queue_time_end"] <= timestamps["server_first_token_time"]):
            print(f"{prefix}⚠️  警告: 时间戳顺序异常")
    
    return all_present


async def test_single_request(base_url):
    """测试单个请求"""
    print("\n" + "="*60)
    print("测试单个请求的Queue时间戳")
    print("="*60)
    
    async with aiohttp.ClientSession() as session:
        # 先检测API格式
        api_format = await test_api_format(session, base_url)
        if not api_format:
            return False
        
        # 发送测试请求
        print("\n📤 发送测试请求...")
        result = await send_request_with_format(
            session, base_url, 
            "Write a haiku about programming", 
            api_format
        )
        
        print(f"⏱️  请求耗时: {result.get('_request_time', 0)*1000:.2f}ms")
        print("\n📊 时间戳分析:")
        print("-"*40)
        
        return await analyze_timestamps(result)


async def test_concurrent_requests(base_url, num_requests=10):
    """测试并发请求"""
    print("\n" + "="*60)
    print(f"测试{num_requests}个并发请求的Queue时间戳")
    print("="*60)
    
    async with aiohttp.ClientSession() as session:
        # 先检测API格式
        api_format = await test_api_format(session, base_url)
        if not api_format:
            return False
        
        # 创建并发请求
        print(f"\n📤 发送{num_requests}个并发请求...")
        tasks = []
        for i in range(num_requests):
            prompt = f"Tell me fact number {i+1} about space exploration"
            tasks.append(send_request_with_format(session, base_url, prompt, api_format))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"✅ 所有请求在 {total_time:.2f}秒 内完成")
        print(f"📊 平均每请求: {total_time/num_requests*1000:.2f}ms\n")
        
        # 分析每个请求
        all_success = True
        queue_times = []
        
        for i, result in enumerate(results):
            print(f"\n--- 请求 {i} ---")
            success = await analyze_timestamps(result, i)
            if not success:
                all_success = False
            
            # 收集queue时间
            if "meta_info" in result:
                meta = result["meta_info"]
                if meta.get("queue_time_start") and meta.get("queue_time_end"):
                    queue_time = meta["queue_time_end"] - meta["queue_time_start"]
                    queue_times.append(queue_time * 1000)  # 转为毫秒
        
        # 统计queue时间
        if queue_times:
            print(f"\n📊 排队时间统计 (ms):")
            print(f"  最小: {min(queue_times):.2f}")
            print(f"  最大: {max(queue_times):.2f}")
            print(f"  平均: {sum(queue_times)/len(queue_times):.2f}")
        
        return all_success


async def main(args):
    """主测试函数"""
    print(f"\n🚀 路由器Queue时间戳测试")
    print(f"时间: {datetime.now()}")
    print(f"目标: {args.base_url}")
    
    # 先测试连通性
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(args.base_url + "/health") as response:
                if response.status == 200:
                    print("✅ 路由器健康检查通过")
                else:
                    print(f"❌ 路由器健康检查失败: {response.status}")
                    return
    except Exception as e:
        print(f"❌ 无法连接到路由器: {e}")
        return
    
    # 运行测试
    single_success = await test_single_request(args.base_url)
    concurrent_success = await test_concurrent_requests(args.base_url, args.num_requests)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if single_success and concurrent_success:
        print("\n🎉 所有测试通过！Queue时间戳功能正常！")
        print("\n后续步骤:")
        print("1. 使用send_req.py进行完整的路由测试")
        print("2. 检查服务器日志确认时间戳记录")
        print("3. 进行更大规模的负载测试")
    else:
        print("\n❌ 测试失败！Queue时间戳存在问题")
        print("\n排查建议:")
        print("1. 确认scheduler.py的修改已部署")
        print("2. 重启所有服务器和路由器")
        print("3. 检查服务器日志中的[Queue]调试信息")
        print("4. 确认BatchTokenIDOut正确传递queue时间戳")
        
        # 额外的调试建议
        print("\n🔍 深入排查:")
        print("- 检查scheduler_output_processor_mixin.py中queue_time_start/end的收集")
        print("- 确认tokenizer_manager.py正确解析这些时间戳")
        print("- 验证时间基准统一（time.time() vs time.perf_counter()）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试路由器queue时间戳功能")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:60009",
        help="路由器URL (默认: http://localhost:60009)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="并发请求数量 (默认: 10)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
#!/usr/bin/env python3
"""
验证queue时间戳修复的测试脚本
- 测试queue_time_start和queue_time_end是否正确记录
- 验证修复后的功能是否正常工作
"""

import asyncio
import json
import time
import aiohttp
import argparse
from datetime import datetime


async def send_test_request(session, url, prompt="Write a short story about a robot"):
    """发送单个测试请求"""
    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False
    }
    
    async with session.post(url + "/generate", json=payload) as response:
        result = await response.json()
        return result


async def test_single_request(base_url):
    """测试单个请求的时间戳"""
    print("\n" + "="*60)
    print("测试单个请求的Queue时间戳")
    print("="*60 + "\n")
    
    async with aiohttp.ClientSession() as session:
        result = await send_test_request(session, base_url)
        
        print("📋 响应内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if "meta_info" in result:
            meta = result["meta_info"]
            print("\n📊 时间戳分析:")
            print("-"*40)
            
            timestamps = {
                "server_created_time": meta.get("server_created_time"),
                "queue_time_start": meta.get("queue_time_start"),
                "queue_time_end": meta.get("queue_time_end"),
                "server_first_token_time": meta.get("server_first_token_time")
            }
            
            for name, value in timestamps.items():
                status = "✅" if value is not None else "❌"
                print(f"  {status} {name}: {value}")
            
            # 计算时间间隔
            if timestamps["queue_time_start"] and timestamps["queue_time_end"]:
                queue_duration = timestamps["queue_time_end"] - timestamps["queue_time_start"]
                print(f"\n⏱️  纯排队时间: {queue_duration:.3f}秒")
            
            if timestamps["server_created_time"] and timestamps["server_first_token_time"]:
                total_server_time = timestamps["server_first_token_time"] - timestamps["server_created_time"]
                print(f"⏱️  总服务器时间: {total_server_time:.3f}秒")
                
            return all(v is not None for v in timestamps.values())
        else:
            print("❌ 响应中没有meta_info")
            return False


async def test_concurrent_requests(base_url, num_requests=10):
    """测试并发请求的时间戳"""
    print("\n" + "="*60)
    print(f"测试{num_requests}个并发请求的Queue时间戳")
    print("="*60 + "\n")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            prompt = f"Tell me fact number {i+1} about space"
            tasks.append(send_test_request(session, base_url, prompt))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        print(f"✅ 所有请求在 {duration:.2f}秒 内完成\n")
        
        # 分析结果
        all_have_timestamps = True
        queue_durations = []
        
        for i, result in enumerate(results):
            if "meta_info" in result:
                meta = result["meta_info"]
                queue_start = meta.get("queue_time_start")
                queue_end = meta.get("queue_time_end")
                
                if queue_start and queue_end:
                    queue_duration = queue_end - queue_start
                    queue_durations.append(queue_duration)
                    print(f"  请求 {i}: ✅ 排队时间 = {queue_duration:.3f}秒")
                else:
                    print(f"  请求 {i}: ❌ 缺少queue时间戳")
                    all_have_timestamps = False
            else:
                print(f"  请求 {i}: ❌ 没有meta_info")
                all_have_timestamps = False
        
        if queue_durations:
            avg_queue = sum(queue_durations) / len(queue_durations)
            max_queue = max(queue_durations)
            min_queue = min(queue_durations)
            print(f"\n📊 排队时间统计:")
            print(f"  平均: {avg_queue:.3f}秒")
            print(f"  最小: {min_queue:.3f}秒")
            print(f"  最大: {max_queue:.3f}秒")
        
        return all_have_timestamps


async def main(args):
    """主测试函数"""
    print(f"\n🚀 开始测试Queue时间戳修复")
    print(f"时间: {datetime.now()}")
    print(f"服务器: {args.base_url}")
    
    # 测试单个请求
    single_success = await test_single_request(args.base_url)
    
    # 测试并发请求
    concurrent_success = await test_concurrent_requests(args.base_url, args.num_requests)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if single_success and concurrent_success:
        print("\n🎉 所有测试通过！Queue时间戳修复成功！")
        print("\n建议后续步骤:")
        print("1. 运行更大规模的负载测试")
        print("2. 检查服务器日志中的调试信息")
        print("3. 使用send_req.py进行完整的路由测试")
    else:
        print("\n❌ 测试失败！Queue时间戳仍然有问题")
        print("\n排查建议:")
        print("1. 确认代码已正确部署到服务器")
        print("2. 检查服务器启动参数是否包含--enable-metrics")
        print("3. 查看服务器日志中的enable_metrics状态")
        print("4. 使用--log-level debug查看详细的时间戳设置日志")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证queue时间戳修复")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="服务器基础URL"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="并发请求数量"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
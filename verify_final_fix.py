#!/usr/bin/env python3
"""
最终验证脚本：测试queue时间戳是否正确返回
"""

import requests
import json
import time
from datetime import datetime
import statistics

def test_queue_time_fix():
    """测试修复后的queue时间戳"""
    
    # 测试配置
    server_url = "http://localhost:60005/generate"
    router_url = "http://localhost:60009/generate"
    
    test_cases = [
        {"text": "Hello world", "max_tokens": 5},
        {"text": "What is machine learning?", "max_tokens": 20},
        {"text": "Explain quantum computing in simple terms", "max_tokens": 50},
    ]
    
    print("Queue时间戳修复验证测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试直连服务器
    print(f"\n1. 测试直连服务器: {server_url}")
    print("-" * 60)
    
    direct_results = []
    for i, case in enumerate(test_cases):
        data = {
            "text": case["text"],
            "sampling_params": {
                "max_new_tokens": case["max_tokens"],
                "temperature": 0.1
            }
        }
        
        try:
            response = requests.post(server_url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                meta = result.get("meta_info", {})
                
                queue_start = meta.get("queue_time_start")
                queue_end = meta.get("queue_time_end")
                
                if queue_start is not None and queue_end is not None:
                    queue_time = (queue_end - queue_start) * 1000  # ms
                    direct_results.append(queue_time)
                    print(f"  测试 {i+1}: ✅ 成功")
                    print(f"    - queue_time_start: {queue_start}")
                    print(f"    - queue_time_end: {queue_end}")
                    print(f"    - 纯队列时间: {queue_time:.2f} ms")
                else:
                    print(f"  测试 {i+1}: ❌ 失败 - queue时间戳为null")
                    print(f"    - queue_time_start: {queue_start}")
                    print(f"    - queue_time_end: {queue_end}")
            else:
                print(f"  测试 {i+1}: ❌ HTTP错误 {response.status_code}")
        except Exception as e:
            print(f"  测试 {i+1}: ❌ 异常: {e}")
        
        time.sleep(0.5)
    
    # 测试通过路由器
    print(f"\n2. 测试通过路由器: {router_url}")
    print("-" * 60)
    
    router_results = []
    for i, case in enumerate(test_cases):
        data = {
            "text": case["text"],
            "sampling_params": {
                "max_new_tokens": case["max_tokens"],
                "temperature": 0.1
            }
        }
        
        try:
            response = requests.post(router_url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                meta = result.get("meta_info", {})
                
                queue_start = meta.get("queue_time_start")
                queue_end = meta.get("queue_time_end")
                
                if queue_start is not None and queue_end is not None:
                    queue_time = (queue_end - queue_start) * 1000  # ms
                    router_results.append(queue_time)
                    print(f"  测试 {i+1}: ✅ 成功")
                    print(f"    - queue_time_start: {queue_start}")
                    print(f"    - queue_time_end: {queue_end}")
                    print(f"    - 纯队列时间: {queue_time:.2f} ms")
                else:
                    print(f"  测试 {i+1}: ❌ 失败 - queue时间戳为null")
                    print(f"    - queue_time_start: {queue_start}")
                    print(f"    - queue_time_end: {queue_end}")
            else:
                print(f"  测试 {i+1}: ❌ HTTP错误 {response.status_code}")
        except Exception as e:
            print(f"  测试 {i+1}: ❌ 异常: {e}")
        
        time.sleep(0.5)
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if direct_results:
        print(f"\n直连服务器结果:")
        print(f"  - 成功率: {len(direct_results)}/{len(test_cases)} ({len(direct_results)/len(test_cases)*100:.0f}%)")
        print(f"  - 平均队列时间: {statistics.mean(direct_results):.2f} ms")
        print(f"  - 最小/最大: {min(direct_results):.2f} / {max(direct_results):.2f} ms")
    else:
        print(f"\n直连服务器: ❌ 所有测试失败")
    
    if router_results:
        print(f"\n路由器结果:")
        print(f"  - 成功率: {len(router_results)}/{len(test_cases)} ({len(router_results)/len(test_cases)*100:.0f}%)")
        print(f"  - 平均队列时间: {statistics.mean(router_results):.2f} ms")
        print(f"  - 最小/最大: {min(router_results):.2f} / {max(router_results):.2f} ms")
    else:
        print(f"\n路由器: ❌ 所有测试失败")
    
    # 最终判定
    print("\n" + "=" * 80)
    if direct_results and router_results:
        print("✅ 修复成功！queue时间戳正确返回")
    else:
        print("❌ 修复失败！请检查服务器日志")
        print("\n可能的原因:")
        print("1. 服务器需要重启以加载新代码")
        print("2. 需要使用 --enable-metrics 标志")
        print("3. 代码同步问题")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_queue_time_fix()
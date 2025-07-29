#!/usr/bin/env python3
"""
完整修复验证测试脚本
测试queue时间戳是否能正确传递到客户端
"""

import requests
import json
import time
from datetime import datetime
import statistics

def test_queue_timestamps():
    """测试queue时间戳传递"""
    
    # 测试配置
    server_url = "http://localhost:60005/generate"
    router_url = "http://localhost:60009/generate"
    
    print("Queue时间戳完整修复验证")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"直连服务器: {server_url}")
    print(f"路由器地址: {router_url}")
    
    # 测试请求
    test_requests = [
        {
            "text": "Hello, this is a test",
            "sampling_params": {"max_new_tokens": 10, "temperature": 0.1}
        },
        {
            "text": "What is the meaning of life?",
            "sampling_params": {"max_new_tokens": 20, "temperature": 0.1}
        },
        {
            "text": "Explain how neural networks work",
            "sampling_params": {"max_new_tokens": 30, "temperature": 0.1}
        }
    ]
    
    def test_endpoint(url, name):
        """测试单个端点"""
        print(f"\n\n测试 {name}: {url}")
        print("-" * 60)
        
        success_count = 0
        queue_times = []
        
        for i, data in enumerate(test_requests, 1):
            try:
                # 发送请求
                start_time = time.time()
                response = requests.post(url, json=data, timeout=30)
                latency = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    result = response.json()
                    meta = result.get("meta_info", {})
                    
                    # 获取时间戳
                    queue_start = meta.get("queue_time_start")
                    queue_end = meta.get("queue_time_end")
                    server_created = meta.get("server_created_time")
                    server_first_token = meta.get("server_first_token_time")
                    
                    print(f"\n请求 {i}: {data['text'][:30]}...")
                    print(f"  响应延迟: {latency:.1f} ms")
                    print(f"  server_created_time: {server_created}")
                    print(f"  server_first_token_time: {server_first_token}")
                    print(f"  queue_time_start: {queue_start}")
                    print(f"  queue_time_end: {queue_end}")
                    
                    # 检查是否包含hidden_states字段（验证之前的修复）
                    if "hidden_states" in meta:
                        print(f"  ✅ hidden_states字段存在（长度: {len(meta['hidden_states'])}）")
                    
                    # 计算纯队列时间
                    if queue_start is not None and queue_end is not None:
                        queue_time = (queue_end - queue_start) * 1000  # ms
                        queue_times.append(queue_time)
                        success_count += 1
                        print(f"  ✅ 纯队列时间: {queue_time:.3f} ms")
                    else:
                        print(f"  ❌ 无法计算纯队列时间（queue时间戳为null）")
                    
                    # 输出生成的文本前50个字符
                    generated_text = result.get("text", "")[:50]
                    print(f"  生成文本: {generated_text}...")
                    
                else:
                    print(f"\n请求 {i}: ❌ HTTP错误 {response.status_code}")
                    print(f"  错误信息: {response.text}")
                    
            except Exception as e:
                print(f"\n请求 {i}: ❌ 异常 - {type(e).__name__}: {str(e)}")
            
            # 避免请求过快
            time.sleep(0.5)
        
        # 统计结果
        print(f"\n{name} 统计:")
        print(f"  成功率: {success_count}/{len(test_requests)} ({success_count/len(test_requests)*100:.0f}%)")
        if queue_times:
            print(f"  平均队列时间: {statistics.mean(queue_times):.3f} ms")
            print(f"  最小/最大队列时间: {min(queue_times):.3f} / {max(queue_times):.3f} ms")
            return True
        else:
            print(f"  ❌ 所有请求的queue时间戳都为null")
            return False
    
    # 测试直连服务器
    direct_success = test_endpoint(server_url, "直连服务器")
    
    # 测试路由器
    router_success = test_endpoint(router_url, "路由器")
    
    # 最终结果
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if direct_success and router_success:
        print("✅ 修复成功！Queue时间戳正确传递到客户端")
        print("\n关键修复点:")
        print("1. BatchTokenIDOut 包含 queue_time_start/end 字段")
        print("2. BatchStrOut 添加了相同的字段")
        print("3. detokenizer_manager 正确传递这些字段")
        print("4. output_hidden_states 初始化问题已修复")
    else:
        print("❌ 修复未完全成功")
        print("\n可能的原因:")
        print("1. 服务器需要重启以加载新代码")
        print("2. 代码同步问题")
        print("3. 确保使用 --enable-metrics 标志启动服务器")
        
        print("\n建议:")
        print("1. 重启所有服务器进程")
        print("2. 确认代码修改已保存")
        print("3. 检查服务器日志中的 [QueueTime] 调试信息")

if __name__ == "__main__":
    test_queue_timestamps()
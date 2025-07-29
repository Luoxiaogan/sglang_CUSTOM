#!/usr/bin/env python3
"""
测试脚本：调试queue时间戳传输问题
需要服务器以debug模式运行
"""

import requests
import json
import time
from datetime import datetime

def test_with_debug():
    """测试单个请求并查看调试日志"""
    server_url = "http://localhost:60005/generate"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送测试请求到服务器...")
    print(f"服务器地址: {server_url}")
    print("=" * 80)
    
    # 简单的测试请求
    data = {
        "text": "Hello, this is a debug test",
        "sampling_params": {
            "max_new_tokens": 5,
            "temperature": 0.1
        }
    }
    
    try:
        # 发送请求
        start_time = time.time()
        response = requests.post(server_url, json=data, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ 请求成功 (耗时 {end_time - start_time:.3f}s)")
            print("\n完整响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查meta_info字段
            if "meta_info" in result:
                meta_info = result["meta_info"]
                print(f"\n📊 Meta Info 分析:")
                
                # 检查所有时间相关字段
                time_fields = [
                    "server_created_time",
                    "server_first_token_time", 
                    "queue_time_start",
                    "queue_time_end",
                    "e2e_latency"
                ]
                
                for field in time_fields:
                    if field in meta_info:
                        value = meta_info[field]
                        if value is not None:
                            print(f"  ✅ {field}: {value}")
                        else:
                            print(f"  ❌ {field}: null")
                    else:
                        print(f"  ❌ {field}: 字段不存在")
                
                # 尝试计算纯队列时间
                if (meta_info.get("queue_time_start") is not None and 
                    meta_info.get("queue_time_end") is not None):
                    pure_queue_time = meta_info["queue_time_end"] - meta_info["queue_time_start"]
                    print(f"\n  📊 纯队列时间: {pure_queue_time*1000:.1f} ms")
                else:
                    print(f"\n  ❌ 无法计算纯队列时间（缺少必要字段）")
            else:
                print(f"\n❌ 响应中没有 meta_info 字段")
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"\n❌ 发生错误: {type(e).__name__}: {str(e)}")

    print("\n" + "=" * 80)
    print("请检查服务器日志中的 [QueueTime] 调试信息")
    print("如果服务器不是以 --log-level debug 运行，请重启服务器并添加该参数")

if __name__ == "__main__":
    print("Queue时间戳调试测试")
    print("=" * 80)
    test_with_debug()
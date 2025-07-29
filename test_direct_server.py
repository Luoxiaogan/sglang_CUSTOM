#!/usr/bin/env python3
"""
直接连接SGLang服务器测试脚本
用于验证服务器是否真的返回queue_time_start/end字段
"""

import requests
import json
import time
from datetime import datetime

def test_direct_server():
    """直接测试服务器响应"""
    # 直接访问服务器，绕过路由器
    server_url = "http://localhost:60005/generate"
    
    # 测试数据
    test_prompts = [
        "Hello world",
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms"
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始直连服务器测试")
    print(f"服务器地址: {server_url}")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n测试 {i+1}: {prompt[:50]}...")
        
        # 构造请求
        data = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 10,
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
                
                # 美化打印完整响应
                print(f"\n完整响应 (耗时 {end_time - start_time:.3f}s):")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                # 检查meta_info字段
                if "meta_info" in result:
                    meta_info = result["meta_info"]
                    print(f"\n✅ meta_info 字段存在")
                    
                    # 检查队列时间字段
                    queue_fields = ["queue_time_start", "queue_time_end", 
                                  "server_created_time", "server_first_token_time"]
                    
                    print("\n队列时间相关字段:")
                    for field in queue_fields:
                        if field in meta_info:
                            value = meta_info[field]
                            if value is not None:
                                print(f"  ✅ {field}: {value}")
                            else:
                                print(f"  ❌ {field}: null")
                        else:
                            print(f"  ❌ {field}: 不存在")
                    
                    # 计算纯队列时间
                    if (meta_info.get("queue_time_start") is not None and 
                        meta_info.get("queue_time_end") is not None):
                        pure_queue_time = meta_info["queue_time_end"] - meta_info["queue_time_start"]
                        print(f"\n📊 纯队列时间: {pure_queue_time*1000:.1f} ms")
                else:
                    print(f"\n❌ 响应中没有 meta_info 字段")
                
                # 检查其他关键字段
                print(f"\n其他字段:")
                print(f"  text: {result.get('text', 'N/A')[:50]}...")
                print(f"  usage: {result.get('usage', 'N/A')}")
                
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时")
        except Exception as e:
            print(f"❌ 发生错误: {type(e).__name__}: {str(e)}")
        
        print("-" * 80)
        time.sleep(1)  # 避免请求过快

def test_via_router():
    """通过路由器测试（对比用）"""
    router_url = "http://localhost:60009/generate"
    
    print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] 开始路由器测试（对比）")
    print(f"路由器地址: {router_url}")
    print("=" * 80)
    
    data = {
        "text": "Quick test through router",
        "sampling_params": {
            "max_new_tokens": 10,
            "temperature": 0.1
        }
    }
    
    try:
        response = requests.post(router_url, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("\n完整响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查meta_info
            if "meta_info" in result:
                meta_info = result["meta_info"]
                print(f"\n✅ meta_info 存在，包含字段:")
                for key, value in meta_info.items():
                    print(f"  - {key}: {value}")
            else:
                print(f"\n❌ 响应中没有 meta_info 字段")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 发生错误: {type(e).__name__}: {str(e)}")

def test_server_info():
    """测试服务器信息接口"""
    info_url = "http://localhost:60005/get_server_info"
    
    print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] 获取服务器信息")
    print("=" * 80)
    
    try:
        response = requests.get(info_url, timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("服务器信息:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 获取服务器信息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 发生错误: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    print("SGLang 直连服务器测试")
    print("=" * 80)
    
    # 1. 测试服务器信息
    test_server_info()
    
    # 2. 直连服务器测试
    test_direct_server()
    
    # 3. 通过路由器测试（对比）
    test_via_router()
    
    print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] 测试完成")
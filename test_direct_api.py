#!/usr/bin/env python3
"""
直接测试不同的 API 端点，看看响应格式的差异
"""

import requests
import json
import time

def test_endpoint(url, endpoint, data):
    """测试特定端点"""
    print(f"\n测试: {url}{endpoint}")
    print("-" * 60)
    
    try:
        start = time.time()
        response = requests.post(f"{url}{endpoint}", json=data, timeout=30)
        elapsed = time.time() - start
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {elapsed:.3f}s")
        print(f"响应头: {dict(response.headers)}")
        
        # 尝试解析 JSON
        try:
            response_json = json.loads(response.text)
            print(f"响应类型: JSON")
            print(f"顶层键: {list(response_json.keys())}")
            
            # 递归查找所有包含时间相关字段的对象
            def find_time_fields(obj, path=""):
                if isinstance(obj, dict):
                    # 检查是否有时间相关字段
                    time_fields = [k for k in obj.keys() if 'time' in k.lower() or 'latency' in k.lower()]
                    if time_fields:
                        print(f"\n在 {path} 找到时间字段: {time_fields}")
                        for field in time_fields:
                            print(f"  {field}: {obj[field]}")
                    
                    # 递归检查
                    for k, v in obj.items():
                        find_time_fields(v, f"{path}.{k}" if path else k)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        find_time_fields(item, f"{path}[{i}]")
            
            find_time_fields(response_json)
            
            # 打印完整响应（限制长度）
            print(f"\n完整响应（前800字符）:")
            print(json.dumps(response_json, indent=2)[:800])
            
        except json.JSONDecodeError:
            print(f"响应类型: 非JSON")
            print(f"响应内容（前200字符）: {response.text[:200]}")
            
    except Exception as e:
        print(f"错误: {e}")


def main():
    # 测试服务器
    servers = [
        "http://localhost:40005",
        "http://localhost:40006",
    ]
    
    # 测试不同的端点和请求格式
    test_cases = [
        {
            "endpoint": "/generate",
            "data": {
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
                "stream": False
            }
        },
        {
            "endpoint": "/generate", 
            "data": {
                "text": "Hello",
                "max_new_tokens": 5,  # 简化格式
                "temperature": 0,
                "stream": False
            }
        },
        {
            "endpoint": "/v1/completions",
            "data": {
                "model": "default",
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0,
                "stream": False
            }
        },
        {
            "endpoint": "/generate",
            "data": {
                "text": "Hello",
                "sampling_params": {
                    "max_new_tokens": 5,
                    "temperature": 0,
                    # 添加更多参数，可能触发不同的代码路径
                    "top_p": 1.0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
                "stream": False,
                "return_logprob": False,  # 明确指定
            }
        }
    ]
    
    print("=" * 60)
    print("测试不同的 API 端点和请求格式")
    print("=" * 60)
    
    for server in servers:
        for i, test_case in enumerate(test_cases):
            print(f"\n\n### 测试用例 {i+1} - 服务器 {server} ###")
            test_endpoint(server, test_case["endpoint"], test_case["data"])
            time.sleep(0.5)  # 避免请求过快
    
    # 测试路由器
    print(f"\n\n### 测试路由器 ###")
    test_endpoint("http://localhost:40009", "/generate", test_cases[0]["data"])


if __name__ == "__main__":
    main()
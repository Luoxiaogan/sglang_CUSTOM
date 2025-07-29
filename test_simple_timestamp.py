#!/usr/bin/env python3
"""
简单测试 - 专注于非流式响应的时间戳
"""

import requests
import json
import sys


def test_server_directly(server_url):
    """直接测试服务器"""
    print(f"\n测试服务器: {server_url}")
    print("-" * 50)
    
    # 测试数据
    data = {
        "text": "Hello world",
        "sampling_params": {
            "max_new_tokens": 10,
            "temperature": 0.0
        },
        "stream": False  # 明确指定非流式
    }
    
    try:
        # 发送请求
        response = requests.post(f"{server_url}/generate", json=data, timeout=30)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        # 获取响应文本
        response_text = response.text
        print(f"响应长度: {len(response_text)} 字符")
        
        # 尝试解析 JSON
        try:
            response_json = json.loads(response_text)
            
            # 打印响应的顶层键
            print(f"\n响应的顶层键: {list(response_json.keys())}")
            
            # 检查是否有 text 字段
            if "text" in response_json:
                print(f"text 字段内容: {response_json['text'][:100]}...")
            
            # 检查是否有 meta_info
            if "meta_info" in response_json:
                print(f"\n✅ 找到 meta_info!")
                meta_info = response_json["meta_info"]
                print(f"meta_info 的键: {list(meta_info.keys())}")
                
                # 打印完整的 meta_info
                print("\n完整的 meta_info 内容:")
                print(json.dumps(meta_info, indent=2))
                
                # 特别检查时间戳
                timestamp_fields = [
                    "server_created_time", 
                    "server_first_token_time",
                    "created_time",
                    "first_token_time",
                    "time_cost"
                ]
                
                print("\n时间戳字段检查:")
                for field in timestamp_fields:
                    if field in meta_info:
                        print(f"  ✅ {field}: {meta_info[field]}")
                    else:
                        print(f"  ❌ {field}: 不存在")
                        
            else:
                print("\n❌ 响应中没有 meta_info 字段")
                
            # 打印完整响应（限制长度）
            print("\n完整响应内容（前1000字符）:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False)[:1000])
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 解析失败: {e}")
            print(f"原始响应（前500字符）: {response_text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
        

def test_router(router_url):
    """测试路由器"""
    print(f"\n测试路由器: {router_url}")
    print("-" * 50)
    
    # 测试数据
    data = {
        "text": "Hello from router",
        "sampling_params": {
            "max_new_tokens": 10,
            "temperature": 0.0
        },
        "stream": False
    }
    
    try:
        # 发送请求
        response = requests.post(f"{router_url}/generate", json=data, timeout=30)
        
        print(f"状态码: {response.status_code}")
        
        # 获取响应文本
        response_text = response.text
        
        # 尝试解析 JSON
        try:
            response_json = json.loads(response_text)
            
            # 打印响应的顶层键
            print(f"\n响应的顶层键: {list(response_json.keys())}")
            
            # 检查是否有 meta_info
            if "meta_info" in response_json:
                print(f"\n✅ 路由器响应包含 meta_info")
                meta_info = response_json["meta_info"]
                print(f"meta_info 内容:")
                print(json.dumps(meta_info, indent=2))
            else:
                print("\n❌ 路由器响应中没有 meta_info")
                
            # 打印部分响应
            print("\n响应内容（前500字符）:")
            print(json.dumps(response_json, indent=2)[:500])
            
        except json.JSONDecodeError:
            print(f"\n❌ 路由器响应不是 JSON")
            print(f"原始响应: {response_text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")


def main():
    # 默认地址
    router_url = "http://localhost:40009"
    server_urls = ["http://localhost:40005", "http://localhost:40006"]
    
    # 从命令行参数获取
    if len(sys.argv) > 1:
        router_url = sys.argv[1]
    if len(sys.argv) > 2:
        server_urls = sys.argv[2:]
    
    print("=" * 60)
    print("SGLang 时间戳测试工具（简化版）")
    print("=" * 60)
    
    # 测试每个服务器
    for server_url in server_urls:
        test_server_directly(server_url)
    
    # 测试路由器
    test_router(router_url)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    

if __name__ == "__main__":
    main()
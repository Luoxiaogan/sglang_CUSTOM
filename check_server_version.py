#!/usr/bin/env python3
"""
检查服务器是否使用了修改后的 tokenizer_manager.py
"""

import requests
import json
import sys

def check_server_version(server_url):
    """发送请求并检查响应中的版本标记"""
    print(f"\n检查服务器: {server_url}")
    print("-" * 50)
    
    # 测试数据
    data = {
        "text": "Version check",
        "sampling_params": {
            "max_new_tokens": 5,
            "temperature": 0.0
        },
        "stream": False
    }
    
    try:
        response = requests.post(f"{server_url}/generate", json=data, timeout=30)
        
        if response.status_code == 200:
            response_json = json.loads(response.text)
            
            if "meta_info" in response_json:
                meta_info = response_json["meta_info"]
                
                # 检查调试版本标记
                if "_debug_version" in meta_info:
                    print(f"✅ 服务器使用调试版本: {meta_info['_debug_version']}")
                else:
                    print("❌ 服务器没有使用调试版本（没有 _debug_version 字段）")
                
                # 检查时间戳字段
                timestamp_fields = ["server_created_time", "server_first_token_time"]
                for field in timestamp_fields:
                    if field in meta_info:
                        print(f"✅ 找到 {field}: {meta_info[field]}")
                    else:
                        print(f"❌ 未找到 {field}")
                
                print(f"\nmeta_info 完整内容:")
                print(json.dumps(meta_info, indent=2))
            else:
                print("❌ 响应中没有 meta_info")
                
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    servers = [
        "http://localhost:40005",
        "http://localhost:40006",
        "http://localhost:40009"  # 路由器
    ]
    
    if len(sys.argv) > 1:
        servers = sys.argv[1:]
    
    print("=" * 60)
    print("检查 SGLang 服务器版本")
    print("=" * 60)
    
    for server in servers:
        check_server_version(server)
    
    print("\n" + "=" * 60)
    print("如果看到调试版本标记，说明代码修改已生效")
    print("如果没有看到，需要：")
    print("1. 确保服务器端的 tokenizer_manager.py 已更新")
    print("2. 删除所有 .pyc 文件")
    print("3. 完全重启服务器")
    print("=" * 60)


if __name__ == "__main__":
    main()
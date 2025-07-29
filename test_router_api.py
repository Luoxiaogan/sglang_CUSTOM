#!/usr/bin/env python3
"""
简单测试路由器API格式
"""

import asyncio
import aiohttp
import json


async def test_api_formats():
    """测试不同的API格式"""
    router_url = "http://localhost:60009"
    
    # 测试格式1：新格式 (text + sampling_params)
    formats = [
        {
            "name": "Format 1: text + sampling_params",
            "payload": {
                "text": "Hello, how are you?",
                "sampling_params": {
                    "max_new_tokens": 10,
                    "temperature": 0.1
                }
            }
        },
        {
            "name": "Format 2: prompt + max_tokens",
            "payload": {
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
                "temperature": 0.1
            }
        },
        {
            "name": "Format 3: text only",
            "payload": {
                "text": "Hello, how are you?",
                "max_new_tokens": 10,
                "temperature": 0.1
            }
        },
        {
            "name": "Format 4: messages (chat format)",
            "payload": {
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 10,
                "temperature": 0.1
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        # 先测试健康检查
        try:
            async with session.get(router_url + "/health") as response:
                if response.status == 200:
                    print("✅ Router health check passed\n")
                else:
                    print(f"❌ Router health check failed: {response.status}\n")
        except Exception as e:
            print(f"❌ Cannot connect to router: {e}\n")
            return
        
        # 测试不同格式
        for fmt in formats:
            print(f"Testing {fmt['name']}...")
            print(f"Payload: {json.dumps(fmt['payload'], indent=2)}")
            
            try:
                async with session.post(
                    router_url + "/generate", 
                    json=fmt['payload'],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    
                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"✅ Success!")
                        print(f"Response keys: {list(result.keys())}")
                        if "meta_info" in result:
                            meta_keys = list(result["meta_info"].keys())
                            print(f"Meta info keys: {meta_keys}")
                            # 检查queue时间戳
                            if "queue_time_start" in result["meta_info"]:
                                print(f"  queue_time_start: {result['meta_info']['queue_time_start']}")
                            if "queue_time_end" in result["meta_info"]:
                                print(f"  queue_time_end: {result['meta_info']['queue_time_end']}")
                            
            except asyncio.TimeoutError:
                print("❌ Timeout")
            except Exception as e:
                print(f"❌ Exception: {type(e).__name__}: {e}")
            
            print("-" * 50 + "\n")


async def test_endpoints():
    """测试不同的端点"""
    router_url = "http://localhost:60009"
    
    endpoints = [
        "/generate",
        "/v1/completions",
        "/v1/chat/completions",
        "/completions",
        "/chat/completions"
    ]
    
    payload = {
        "text": "Hello",
        "sampling_params": {"max_new_tokens": 5}
    }
    
    print("Testing different endpoints...\n")
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"Testing {endpoint}...")
            try:
                async with session.post(
                    router_url + endpoint, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Status 200, response type: {type(result)}")
                        if isinstance(result, dict) and "error" not in result:
                            print(f"   Keys: {list(result.keys())[:5]}...")
                    else:
                        print(f"❌ Status {response.status}")
            except Exception as e:
                print(f"❌ {type(e).__name__}")
            print()


async def main():
    print("🚀 Testing Router API Formats\n")
    await test_api_formats()
    print("\n" + "="*60 + "\n")
    await test_endpoints()


if __name__ == "__main__":
    asyncio.run(main())
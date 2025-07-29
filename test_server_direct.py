#!/usr/bin/env python3
"""
直接测试服务器（绕过路由器）以验证queue时间戳功能
"""

import asyncio
import aiohttp
import json
import sys


async def test_server_directly(server_url: str):
    """直接测试服务器的queue时间戳"""
    
    data = {
        "text": "Hello, how are you?",
        "sampling_params": {
            "max_new_tokens": 10,
            "temperature": 0.1
        },
        "stream": False
    }
    
    print(f"\n{'='*60}")
    print(f"Testing server directly: {server_url}")
    print(f"{'='*60}\n")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 健康检查
            async with session.get(f"{server_url}/health") as resp:
                if resp.status == 200:
                    print("✅ Server health check passed")
                else:
                    print(f"❌ Server health check failed: {resp.status}")
                    return
                    
            # 发送测试请求
            print(f"\n📤 Sending request...")
            async with session.post(
                f"{server_url}/generate",
                json=data
            ) as resp:
                print(f"Status: {resp.status}")
                
                response_text = await resp.text()
                try:
                    response_json = json.loads(response_text)
                    print("\n📋 Response:")
                    print(json.dumps(response_json, indent=2))
                    
                    if "meta_info" in response_json:
                        meta = response_json["meta_info"]
                        print("\n📊 Queue Timestamps Analysis:")
                        print("-" * 40)
                        
                        timestamps = {
                            "server_created_time": meta.get("server_created_time"),
                            "queue_time_start": meta.get("queue_time_start"),
                            "queue_time_end": meta.get("queue_time_end"),
                            "server_first_token_time": meta.get("server_first_token_time")
                        }
                        
                        all_present = True
                        for name, value in timestamps.items():
                            if value is None:
                                print(f"❌ {name}: None/Missing")
                                all_present = False
                            else:
                                print(f"✅ {name}: {value}")
                        
                        if timestamps["queue_time_start"] and timestamps["queue_time_end"]:
                            queue_duration = (timestamps["queue_time_end"] - timestamps["queue_time_start"]) * 1000
                            print(f"\n⏱️  Pure queue time: {queue_duration:.2f}ms")
                        
                        if all_present:
                            print("\n🎉 All queue timestamps are present!")
                        else:
                            print("\n⚠️  Some timestamps are missing")
                            
                except Exception as e:
                    print(f"❌ Failed to parse response: {e}")
                    print(f"Response text: {response_text[:500]}")
                    
        except Exception as e:
            print(f"❌ Request failed: {type(e).__name__}: {e}")


async def main():
    servers = ["http://localhost:60005", "http://localhost:60006"]
    
    print("🚀 Testing servers directly (bypassing router)")
    
    for server in servers:
        await test_server_directly(server)
        print()


if __name__ == "__main__":
    asyncio.run(main())
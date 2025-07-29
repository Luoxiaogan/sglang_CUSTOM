#!/usr/bin/env python3
"""
简单测试路由器响应格式
"""

import asyncio
import aiohttp


async def test_router_response():
    """测试路由器的原始响应"""
    router_url = "http://localhost:60009"
    
    # 使用send_req.py相同的格式
    data = {
        "text": "Hello, how are you?",
        "sampling_params": {
            "max_new_tokens": 10,
            "temperature": 0.1
        },
        "stream": False
    }
    
    print(f"Testing router at {router_url}/generate")
    print(f"Request data: {data}")
    print("-" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{router_url}/generate",
                json=data
            ) as resp:
                print(f"Status: {resp.status}")
                print(f"Content-Type: {resp.headers.get('Content-Type', 'Not specified')}")
                print(f"Headers: {dict(resp.headers)}")
                print("-" * 60)
                
                # 获取原始文本响应
                response_text = await resp.text()
                print(f"Response text (first 500 chars):")
                print(response_text[:500])
                print("-" * 60)
                
                # 尝试解析为JSON
                try:
                    import json
                    response_json = json.loads(response_text)
                    print("Successfully parsed as JSON!")
                    print(f"JSON keys: {list(response_json.keys())}")
                    if "meta_info" in response_json:
                        print(f"meta_info keys: {list(response_json['meta_info'].keys())}")
                        # 检查queue时间戳
                        for key in ["queue_time_start", "queue_time_end", "server_created_time", "server_first_token_time"]:
                            if key in response_json["meta_info"]:
                                print(f"  {key}: {response_json['meta_info'][key]}")
                except Exception as e:
                    print(f"Failed to parse as JSON: {e}")
                    
        except Exception as e:
            print(f"Request failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_router_response())
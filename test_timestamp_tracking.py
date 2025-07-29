#!/usr/bin/env python3
"""
全面测试 SGLang 时间戳追踪功能
用于诊断为什么 server_created_time 和 server_first_token_time 没有出现在响应中
"""

import asyncio
import aiohttp
import json
import time
import argparse
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime


class TimestampDiagnostics:
    def __init__(self, router_url: str, server_urls: List[str]):
        self.router_url = router_url
        self.server_urls = server_urls
        self.results = []
        
    async def test_direct_server_generate(self, server_url: str) -> Dict[str, Any]:
        """直接测试服务器的 /generate 端点"""
        print(f"\n{'='*60}")
        print(f"测试直接服务器: {server_url}/generate")
        print(f"{'='*60}")
        
        test_data = {
            "text": "Hello, this is a test",
            "sampling_params": {
                "max_new_tokens": 10,
                "temperature": 0.0
            },
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    f"{server_url}/generate",
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    end_time = time.time()
                    
                    status = resp.status
                    headers = dict(resp.headers)
                    content_type = headers.get('Content-Type', '')
                    
                    response_text = await resp.text()
                    
                    print(f"状态码: {status}")
                    print(f"响应时间: {end_time - start_time:.3f}s")
                    print(f"Content-Type: {content_type}")
                    print(f"响应长度: {len(response_text)} 字符")
                    
                    # 尝试解析JSON
                    try:
                        response_json = json.loads(response_text)
                        print("\n✅ 响应是有效的 JSON")
                        print(f"响应结构: {list(response_json.keys())}")
                        
                        # 检查 meta_info
                        if "meta_info" in response_json:
                            print("\n✅ 找到 meta_info")
                            meta_info = response_json["meta_info"]
                            print(f"meta_info 字段: {list(meta_info.keys())}")
                            
                            # 检查时间戳
                            timestamps = ["server_created_time", "server_first_token_time", 
                                        "created_time", "first_token_time", "time_cost"]
                            
                            print("\n时间戳检查:")
                            for ts in timestamps:
                                if ts in meta_info:
                                    print(f"  ✅ {ts}: {meta_info[ts]}")
                                else:
                                    print(f"  ❌ {ts}: 未找到")
                            
                            # 打印完整的 meta_info
                            print("\n完整的 meta_info:")
                            print(json.dumps(meta_info, indent=2))
                        else:
                            print("\n❌ 响应中没有 meta_info")
                            
                        # 打印完整响应（截断）
                        print("\n完整响应（前500字符）:")
                        print(json.dumps(response_json, indent=2)[:500])
                        
                    except json.JSONDecodeError:
                        print("\n❌ 响应不是 JSON 格式")
                        print(f"响应内容（前200字符）: {response_text[:200]}")
                        
                    return {
                        "server": server_url,
                        "endpoint": "/generate",
                        "status": status,
                        "has_json": isinstance(response_text, str) and response_text.startswith('{'),
                        "has_meta_info": "meta_info" in response_text,
                        "response_sample": response_text[:200]
                    }
                    
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
                return {
                    "server": server_url,
                    "endpoint": "/generate",
                    "error": str(e)
                }
    
    async def test_router_generate(self) -> Dict[str, Any]:
        """测试路由器的 /generate 端点"""
        print(f"\n{'='*60}")
        print(f"测试路由器: {self.router_url}/generate")
        print(f"{'='*60}")
        
        test_data = {
            "text": "Hello, this is a router test",
            "sampling_params": {
                "max_new_tokens": 10,
                "temperature": 0.0
            },
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    f"{self.router_url}/generate",
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    end_time = time.time()
                    
                    status = resp.status
                    headers = dict(resp.headers)
                    content_type = headers.get('Content-Type', '')
                    
                    response_text = await resp.text()
                    
                    print(f"状态码: {status}")
                    print(f"响应时间: {end_time - start_time:.3f}s")
                    print(f"Content-Type: {content_type}")
                    print(f"响应长度: {len(response_text)} 字符")
                    
                    # 尝试解析JSON
                    try:
                        response_json = json.loads(response_text)
                        print("\n✅ 响应是有效的 JSON")
                        print(f"响应结构: {list(response_json.keys())}")
                        
                        # 检查 meta_info
                        if "meta_info" in response_json:
                            print("\n✅ 找到 meta_info")
                            meta_info = response_json["meta_info"]
                            print(f"meta_info 字段: {list(meta_info.keys())}")
                            
                            # 打印完整的 meta_info
                            print("\n完整的 meta_info:")
                            print(json.dumps(meta_info, indent=2))
                        else:
                            print("\n❌ 响应中没有 meta_info")
                            
                        # 打印完整响应（截断）
                        print("\n完整响应（前500字符）:")
                        print(json.dumps(response_json, indent=2)[:500])
                        
                    except json.JSONDecodeError:
                        print("\n❌ 响应不是 JSON 格式")
                        print(f"响应内容（前200字符）: {response_text[:200]}")
                        
                    return {
                        "server": "router",
                        "endpoint": "/generate",
                        "status": status,
                        "has_json": isinstance(response_text, str) and response_text.startswith('{'),
                        "has_meta_info": "meta_info" in response_text,
                        "response_sample": response_text[:200]
                    }
                    
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
                return {
                    "server": "router",
                    "endpoint": "/generate",
                    "error": str(e)
                }
    
    async def test_openai_compatible(self, url: str, is_router: bool = False) -> Dict[str, Any]:
        """测试 OpenAI 兼容的 /v1/completions 端点"""
        endpoint_url = f"{url}/v1/completions"
        print(f"\n{'='*60}")
        print(f"测试 OpenAI 兼容端点: {endpoint_url}")
        print(f"{'='*60}")
        
        test_data = {
            "model": "default",
            "prompt": "Hello, this is an OpenAI compatible test",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    endpoint_url,
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    end_time = time.time()
                    
                    status = resp.status
                    response_text = await resp.text()
                    
                    print(f"状态码: {status}")
                    print(f"响应时间: {end_time - start_time:.3f}s")
                    
                    # 尝试解析JSON
                    try:
                        response_json = json.loads(response_text)
                        print("\n✅ 响应是有效的 JSON")
                        print(f"响应结构: {list(response_json.keys())}")
                        
                        # OpenAI 格式通常没有 meta_info，但检查一下
                        if "meta_info" in response_json:
                            print("\n✅ 找到 meta_info（意外）")
                            print(json.dumps(response_json["meta_info"], indent=2))
                        
                        # 检查 usage 字段（OpenAI 格式）
                        if "usage" in response_json:
                            print("\n找到 usage 字段:")
                            print(json.dumps(response_json["usage"], indent=2))
                            
                        print("\n完整响应（前500字符）:")
                        print(json.dumps(response_json, indent=2)[:500])
                        
                    except json.JSONDecodeError:
                        print("\n❌ 响应不是 JSON 格式")
                        print(f"响应内容（前200字符）: {response_text[:200]}")
                        
                    return {
                        "server": "router" if is_router else url,
                        "endpoint": "/v1/completions",
                        "status": status,
                        "response_sample": response_text[:200]
                    }
                    
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
                return {
                    "server": "router" if is_router else url,
                    "endpoint": "/v1/completions",
                    "error": str(e)
                }
    
    async def test_stream_response(self, server_url: str) -> Dict[str, Any]:
        """测试流式响应"""
        print(f"\n{'='*60}")
        print(f"测试流式响应: {server_url}/generate")
        print(f"{'='*60}")
        
        test_data = {
            "text": "Hello, this is a stream test",
            "sampling_params": {
                "max_new_tokens": 20,
                "temperature": 0.0
            },
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                chunks = []
                async with session.post(
                    f"{server_url}/generate",
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    print(f"状态码: {resp.status}")
                    print(f"Content-Type: {resp.headers.get('Content-Type', '')}")
                    
                    if resp.status == 200:
                        print("\n接收到的数据块:")
                        async for line in resp.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]
                                    if data_str != '[DONE]':
                                        chunks.append(data_str)
                                        print(f"  块 {len(chunks)}: {data_str[:100]}...")
                                        
                                        # 尝试解析第一个块
                                        if len(chunks) == 1:
                                            try:
                                                chunk_json = json.loads(data_str)
                                                if "meta_info" in chunk_json:
                                                    print("\n✅ 第一个块包含 meta_info:")
                                                    print(json.dumps(chunk_json["meta_info"], indent=2))
                                            except:
                                                pass
                        
                        print(f"\n总共接收到 {len(chunks)} 个数据块")
                        
                        # 检查最后一个块
                        if chunks:
                            try:
                                last_chunk = json.loads(chunks[-1])
                                if "meta_info" in last_chunk:
                                    print("\n✅ 最后一个块包含 meta_info:")
                                    print(json.dumps(last_chunk["meta_info"], indent=2))
                            except:
                                pass
                    
                    return {
                        "server": server_url,
                        "endpoint": "/generate (stream)",
                        "status": resp.status,
                        "chunks_received": len(chunks)
                    }
                    
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
                return {
                    "server": server_url,
                    "endpoint": "/generate (stream)",
                    "error": str(e)
                }
    
    async def check_server_health(self, server_url: str) -> bool:
        """检查服务器健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'#'*60}")
        print(f"# SGLang 时间戳追踪诊断工具")
        print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 路由器: {self.router_url}")
        print(f"# 服务器: {self.server_urls}")
        print(f"{'#'*60}")
        
        # 1. 检查服务器健康状态
        print("\n检查服务器健康状态...")
        for server_url in self.server_urls:
            health = await self.check_server_health(server_url)
            print(f"  {server_url}: {'✅ 健康' if health else '❌ 不可达'}")
        
        router_health = await self.check_server_health(self.router_url)
        print(f"  {self.router_url} (路由器): {'✅ 健康' if router_health else '❌ 不可达'}")
        
        # 2. 测试直接服务器响应
        for server_url in self.server_urls:
            await self.test_direct_server_generate(server_url)
        
        # 3. 测试路由器响应
        await self.test_router_generate()
        
        # 4. 测试 OpenAI 兼容端点
        for server_url in self.server_urls:
            await self.test_openai_compatible(server_url)
        await self.test_openai_compatible(self.router_url, is_router=True)
        
        # 5. 测试流式响应
        if self.server_urls:
            await self.test_stream_response(self.server_urls[0])
        
        # 6. 生成诊断报告
        self.generate_report()
    
    def generate_report(self):
        """生成诊断报告"""
        print(f"\n{'#'*60}")
        print(f"# 诊断报告")
        print(f"{'#'*60}")
        
        print("\n## 可能的问题：")
        print("1. ❌ 如果直接服务器响应中没有 meta_info:")
        print("   - tokenizer_manager.py 的修改可能没有生效")
        print("   - 服务器可能使用了不同的响应路径")
        print("   - 需要检查服务器启动参数")
        
        print("\n2. ❌ 如果服务器有 meta_info 但路由器没有:")
        print("   - 路由器可能没有正确透传响应")
        print("   - 需要检查路由器代码")
        
        print("\n3. ❌ 如果 meta_info 存在但没有时间戳字段:")
        print("   - 代码修改可能不完整")
        print("   - 需要检查 BatchStrOut 类型的处理")
        
        print("\n## 建议的下一步：")
        print("1. 检查服务器日志，确认使用的代码版本")
        print("2. 在 tokenizer_manager.py 中添加调试日志")
        print("3. 检查请求是否走了预期的代码路径")
        print("4. 考虑使用不同的响应格式或端点")


async def main():
    parser = argparse.ArgumentParser(description="诊断 SGLang 时间戳追踪问题")
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:40009",
        help="路由器 URL"
    )
    parser.add_argument(
        "--server-urls",
        nargs="+",
        default=["http://localhost:40005", "http://localhost:40006"],
        help="服务器 URLs"
    )
    
    args = parser.parse_args()
    
    diagnostics = TimestampDiagnostics(args.router_url, args.server_urls)
    await diagnostics.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
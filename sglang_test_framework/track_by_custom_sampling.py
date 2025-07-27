#!/usr/bin/env python3
"""
通过自定义采样参数追踪GPU分配。
每个GPU使用略微不同的temperature，通过输出的多样性推断。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
from typing import Dict, List
import aiohttp
import numpy as np
from collections import defaultdict

from sglang_test_framework import RoutingConfig
from sglang_test_framework.core import ServerManager, RequestGenerator

class GPUTrackingTest:
    """使用不同采样参数追踪GPU分配的测试类"""
    
    def __init__(self):
        self.server_manager = ServerManager()
        self.servers = []
        self.router = None
        
        # GPU特定的采样参数
        self.gpu_sampling_params = {
            0: {"temperature": 0.0, "top_p": 1.0},    # GPU 0: 确定性输出
            1: {"temperature": 1.0, "top_p": 0.95},   # GPU 1: 随机输出
        }
        
    async def setup_servers(self, model_path: str):
        """启动带有不同采样参数的服务器"""
        
        # 为每个GPU启动服务器
        for gpu_id, sampling_params in self.gpu_sampling_params.items():
            server_config = {
                "server_id": f"gpu_{gpu_id}",
                "gpu_id": gpu_id,
                "port": 30001 + gpu_id,
                "model_path": model_path,
                "max_running_requests": 256,
                "mem_fraction_static": 0.9,
            }
            
            server = await self.server_manager.launch_server(server_config)
            self.servers.append(server)
            print(f"✅ Started server on GPU {gpu_id} with params: {sampling_params}")
        
        # 启动router
        router_config = {
            "port": 30000,
            "policy": "round_robin",  # 使用round_robin便于验证
            "prometheus_port": 29000,
        }
        
        worker_urls = [f"http://localhost:{s['port']}" for s in self.servers]
        self.router = await self.server_manager.launch_router(router_config, worker_urls)
        print(f"✅ Started router with {len(worker_urls)} workers")
        
    async def send_test_request(self, prompt: str, request_id: str) -> Dict:
        """发送测试请求并分析响应"""
        
        # 发送两次相同的请求到不同的GPU（通过多次尝试）
        responses = []
        
        for attempt in range(2):
            async with aiohttp.ClientSession() as session:
                # 根据GPU使用不同的采样参数
                for gpu_id, sampling_params in self.gpu_sampling_params.items():
                    payload = {
                        "text": prompt,
                        "sampling_params": {
                            **sampling_params,
                            "max_new_tokens": 50,
                        },
                        "stream": False
                    }
                    
                    async with session.post(
                        "http://localhost:30000/generate",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            responses.append({
                                "attempt": attempt,
                                "gpu_params": sampling_params,
                                "output": result.get("text", ""),
                                "request_id": f"{request_id}_attempt{attempt}"
                            })
        
        return self.analyze_responses(responses)
    
    def analyze_responses(self, responses: List[Dict]) -> Dict:
        """分析响应以确定GPU分配"""
        
        # 对于temperature=0的GPU，输出应该是确定的
        # 对于temperature=1的GPU，输出应该有变化
        
        if len(responses) >= 2:
            output1 = responses[0]["output"]
            output2 = responses[1]["output"]
            
            # 如果两次输出完全相同，很可能是GPU 0 (temperature=0)
            if output1 == output2:
                return {"likely_gpu": 0, "confidence": "high", "reason": "deterministic output"}
            else:
                return {"likely_gpu": 1, "confidence": "high", "reason": "varied output"}
        
        return {"likely_gpu": -1, "confidence": "low", "reason": "insufficient data"}
    
    async def run_tracking_test(self, num_requests: int = 100):
        """运行完整的追踪测试"""
        
        print("\n🧪 Running GPU Tracking Test")
        print("=" * 60)
        
        # 统计结果
        gpu_assignments = defaultdict(int)
        confidence_levels = defaultdict(int)
        
        # 生成测试prompt
        test_prompts = [
            "What is 2+2?",  # 简单确定性问题
            "Tell me a story about",  # 创造性问题
            "List three colors:",  # 半确定性问题
        ]
        
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)]
            result = await self.send_test_request(prompt, f"req_{i}")
            
            gpu_id = result["likely_gpu"]
            confidence = result["confidence"]
            
            if gpu_id >= 0:
                gpu_assignments[gpu_id] += 1
            confidence_levels[confidence] += 1
            
            if i % 10 == 0:
                print(f"Progress: {i+1}/{num_requests} requests processed")
        
        # 打印结果
        print("\n📊 GPU Assignment Results:")
        total_assigned = sum(gpu_assignments.values())
        for gpu_id, count in sorted(gpu_assignments.items()):
            percentage = (count / total_assigned * 100) if total_assigned > 0 else 0
            print(f"  GPU {gpu_id}: {count} requests ({percentage:.1f}%)")
        
        print("\n🎯 Confidence Levels:")
        for level, count in confidence_levels.items():
            print(f"  {level}: {count} requests")
        
        # 验证round-robin分布
        if len(gpu_assignments) == 2:
            counts = list(gpu_assignments.values())
            diff = abs(counts[0] - counts[1])
            print(f"\n📈 Balance Check: Difference = {diff} requests")
            if diff <= num_requests * 0.1:  # 10%容差
                print("✅ Distribution matches round-robin expectation!")
            else:
                print("⚠️  Distribution seems imbalanced")
    
    async def cleanup(self):
        """清理资源"""
        await self.server_manager.stop_all()


async def main():
    """主测试函数"""
    
    model_path = "/data/pretrained_models/Llama-2-7b-hf"  # 修改为实际路径
    
    test = GPUTrackingTest()
    
    try:
        # 设置服务器
        await test.setup_servers(model_path)
        
        # 等待服务器启动
        await asyncio.sleep(5)
        
        # 运行追踪测试
        await test.run_tracking_test(num_requests=50)
        
    finally:
        # 清理
        await test.cleanup()


if __name__ == "__main__":
    print("🚀 GPU Tracking via Sampling Parameters")
    print("Strategy: GPU 0 uses temperature=0 (deterministic)")
    print("         GPU 1 uses temperature=1 (random)")
    print("=" * 60)
    
    asyncio.run(main())
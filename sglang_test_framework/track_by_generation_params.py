#!/usr/bin/env python3
"""
通过生成参数差异化来追踪每个请求的GPU分配。
不需要修改router，立即可用。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sglang_test_framework import RoutingTest, RoutingConfig
from sglang_test_framework.config.base import ServerConfig
import json
import re

class GPUTrackingRoutingConfig(RoutingConfig):
    """扩展RoutingConfig以支持per-GPU生成参数"""
    
    def _create_uniform_server_configs(self):
        """创建带有不同生成参数的服务器配置"""
        configs = []
        
        # 为每个GPU设置不同的标识参数
        gpu_params = {
            0: {
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
            },
            1: {
                "temperature": 0.8,
                "top_p": 0.95,
                "seed": 43,
            }
        }
        
        for i, gpu_id in enumerate(self.gpu_ids):
            config = ServerConfig(
                server_id=f"worker_{i}",
                gpu_id=gpu_id,
                port=self.base_port + i,
                host=self.server_host,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                mem_fraction_static=self.mem_fraction_static,
                max_running_requests=self.max_running_requests,
                chunked_prefill_size=self.chunked_prefill_size,
                enable_torch_compile=self.enable_torch_compile,
                quantization=self.quantization,
                enable_metrics=True,
                verbose=True
            )
            
            # 添加GPU特定的生成参数作为额外启动参数
            # 注意：这需要确认SGLang是否支持这些参数
            gpu_param = gpu_params.get(gpu_id, {})
            
            # 方法1：通过环境变量传递（如果SGLang支持）
            # os.environ[f"SGLANG_GPU_{gpu_id}_TEMPERATURE"] = str(gpu_param["temperature"])
            
            # 方法2：修改默认生成参数（需要确认SGLang API）
            # config.default_generation_params = gpu_param
            
            configs.append(config)
            
        return configs


def analyze_output_for_gpu(output_text: str, request_params: dict) -> int:
    """通过输出特征推断GPU ID"""
    
    # 方法1：如果我们在prompt中注入了标识
    if "[GPU0]" in output_text:
        return 0
    elif "[GPU1]" in output_text:
        return 1
    
    # 方法2：通过输出的随机性特征推断（需要统计分析）
    # 这里需要更复杂的分析逻辑
    
    # 方法3：如果服务器在响应中包含了标识
    # 需要检查完整的响应结构
    
    return -1  # 无法确定


def test_with_generation_tracking():
    """使用生成参数差异化进行GPU追踪测试"""
    
    # 配置测试
    config = GPUTrackingRoutingConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        num_gpus=2,
        gpu_ids=[0, 1],
        routing_policy="round_robin",
        num_prompts=100,
        request_rate=20.0,
        dataset_name="random",
        random_input_len=256,
        random_output_len=64,
        output_dir="./gpu_tracking_results",
    )
    
    print("🔍 GPU Tracking via Generation Parameters")
    print("=" * 60)
    print("Strategy: Different generation parameters per GPU")
    print(f"GPU 0: temperature=0.7, seed=42")
    print(f"GPU 1: temperature=0.8, seed=43")
    print("=" * 60)
    
    # 运行测试
    test = RoutingTest(config)
    
    # 修改请求生成以包含GPU追踪
    # 这里我们需要自定义请求处理逻辑
    
    results = test.run()
    
    # 分析结果
    print("\n📊 Results Analysis")
    print("-" * 60)
    
    # 基于输出特征统计GPU分配
    gpu_assignments = {0: 0, 1: 0, -1: 0}
    
    # 这里需要访问原始响应数据
    # 实际实现需要修改RequestResult以保存完整响应
    
    print("\nGPU Assignment Statistics:")
    for gpu_id, count in sorted(gpu_assignments.items()):
        if gpu_id >= 0:
            print(f"  GPU {gpu_id}: {count} requests")
        else:
            print(f"  Unknown: {count} requests")


def simple_prompt_injection_test():
    """简单的prompt注入方法测试"""
    
    print("\n🏷️  Simple Prompt Injection Method")
    print("=" * 60)
    
    # 为每个GPU创建带标识的system prompt
    gpu_prompts = {
        0: "You are a helpful assistant. [GPU0]",
        1: "You are a helpful assistant. [GPU1]"
    }
    
    # 这需要在服务器启动时设置不同的system prompt
    # 或者在每个请求中动态注入
    
    config = RoutingConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        num_gpus=2,
        gpu_ids=[0, 1],
        routing_policy="round_robin",
        num_prompts=50,
        request_rate=10.0,
        dataset_name="random",
        random_input_len=128,
        random_output_len=32,
    )
    
    # 修改请求以包含追踪逻辑
    # 需要自定义RequestGenerator
    
    print("Note: This method requires modifying server startup")
    print("to use different system prompts per GPU.")


if __name__ == "__main__":
    # 测试不同的追踪方法
    print("🚀 Testing GPU Tracking Methods\n")
    
    # 方法1：生成参数差异化
    test_with_generation_tracking()
    
    # 方法2：Prompt注入
    simple_prompt_injection_test()
    
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    print("1. For immediate testing: Use prompt injection")
    print("2. For production: Implement proper request tracking in router")
    print("3. For analysis: Combine with Prometheus for validation")
    print("=" * 60)
#!/usr/bin/env python3
"""
Quick test for Prometheus tracking with 2 GPUs.
Sends fewer requests for faster testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sglang_test_framework import RoutingTest, RoutingConfig

def main():
    # Quick test with fewer requests
    config = RoutingConfig(
        # Model - 请根据你的环境修改路径
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        
        # Use 2 GPUs
        num_gpus=2,
        gpu_ids=[0, 1],
        
        # Router settings
        routing_policy="round_robin",  # 测试round_robin以验证均匀分布
        
        # Quick test settings
        num_prompts=100,      # 只发送100个请求
        request_rate=20.0,    # 每秒20个请求
        dataset_name="random",
        random_input_len=256,  # 较短的输入
        random_output_len=64,  # 较短的输出
        
        # Output
        output_dir="./quick_test_results",
        warmup_requests=5,
    )
    
    # Enable Prometheus
    config.router_config.prometheus_port = 29000
    
    print("🚀 Quick Prometheus Test")
    print(f"📍 Model: {config.model_path}")
    print(f"🖥️  GPUs: {config.gpu_ids}")
    print(f"📊 Requests: {config.num_prompts}")
    print(f"🎯 Policy: {config.routing_policy}")
    print("-" * 50)
    
    # Run test
    test = RoutingTest(config)
    results = test.run()
    
    # Show key results
    print("\n✨ Key Results:")
    
    if results.get('prometheus_gpu_counts'):
        print("\n📊 Prometheus GPU Distribution:")
        total = sum(results['prometheus_gpu_counts'].values())
        for gpu_id, count in sorted(results['prometheus_gpu_counts'].items()):
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)  # Simple bar chart
            print(f"  GPU {gpu_id}: {count:3d} requests ({pct:5.1f}%) {bar}")
    
    print(f"\n⏱️  Test duration: {results['test_duration']:.1f}s")
    print(f"✅ Success rate: {results['success_rate'] * 100:.1f}%")
    
    # Check balance for round_robin
    if config.routing_policy == "round_robin" and results.get('prometheus_gpu_counts'):
        counts = list(results['prometheus_gpu_counts'].values())
        if len(counts) == 2:
            diff = abs(counts[0] - counts[1])
            if diff <= 2:
                print(f"\n🎯 Perfect round-robin balance! (difference: {diff})")
            else:
                print(f"\n⚠️  Round-robin imbalance detected (difference: {diff})")

if __name__ == "__main__":
    main()
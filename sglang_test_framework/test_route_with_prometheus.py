#!/usr/bin/env python3
"""
Test routing with Prometheus metrics tracking on 2 GPUs.
This script will:
1. Launch 2 SGLang servers with Llama-2-7b-hf on GPU 0 and GPU 1
2. Start a router with Prometheus metrics enabled
3. Send test requests and track which GPU handles each request
4. Display results from both CSV tracking and Prometheus metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sglang_test_framework import RoutingTest, RoutingConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Test configuration
    config = RoutingConfig(
        # Model configuration
        model_path="/data/pretrained_models/Llama-2-7b-hf",  # 修改为你的模型路径
        tokenizer_path="/data/pretrained_models/Llama-2-7b-hf",
        
        # GPU configuration - 使用GPU 0和1
        num_gpus=2,
        gpu_ids=[0, 1],
        
        # Server configuration
        base_port=30001,  # 服务器起始端口
        max_running_requests=256,
        mem_fraction_static=0.9,
        
        # Router configuration with Prometheus
        routing_policy="round_robin",  # 可选: "round_robin", "random", "cache_aware", "shortest_queue"
        
        # Test configuration
        num_prompts=1000,  # 发送1000个请求
        request_rate=50.0,  # 每秒50个请求
        dataset_name="random",  # 使用随机数据集
        random_input_len=512,   # 输入长度
        random_output_len=128,  # 输出长度
        random_range_ratio=0.3, # 长度变化范围
        
        # Output configuration
        output_dir="./results/prometheus_test",
        save_detailed_results=True,
        
        # Warmup
        warmup_requests=10,
    )
    
    # 显式启用Prometheus指标（端口29000）
    config.router_config.prometheus_port = 29000
    config.router_config.prometheus_host = "0.0.0.0"
    
    print("=" * 80)
    print("SGLang Routing Test with Prometheus Metrics")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"GPUs: {config.gpu_ids}")
    print(f"Routing Policy: {config.routing_policy}")
    print(f"Prometheus Port: {config.router_config.prometheus_port}")
    print(f"Number of Requests: {config.num_prompts}")
    print(f"Request Rate: {config.request_rate} req/s")
    print("=" * 80)
    print()
    
    # Create and run test
    test = RoutingTest(config)
    
    try:
        print("Starting test...")
        results = test.run()
        
        # Analyze results
        test.analyze_results(results)
        
        # Create visualizations
        test.visualize_results(results)
        
        # Additional analysis
        print("\n" + "=" * 80)
        print("DETAILED PROMETHEUS ANALYSIS")
        print("=" * 80)
        
        # Check if Prometheus data was collected
        if results.get('prometheus_worker_counts'):
            print("\n✅ Prometheus metrics successfully collected!")
            
            # Worker-level analysis
            print("\n📊 Worker-Level Request Distribution:")
            total_prometheus = 0
            for worker_url, count in sorted(results['prometheus_worker_counts'].items()):
                print(f"  {worker_url}: {count} requests")
                total_prometheus += count
            
            print(f"\n  Total requests tracked by Prometheus: {total_prometheus}")
            
            # GPU-level analysis
            if results.get('prometheus_gpu_counts'):
                print("\n🖥️  GPU-Level Request Distribution (from Prometheus):")
                gpu_counts = results['prometheus_gpu_counts']
                
                for gpu_id in sorted(gpu_counts.keys()):
                    count = gpu_counts[gpu_id]
                    percentage = (count / total_prometheus * 100) if total_prometheus > 0 else 0
                    print(f"  GPU {gpu_id}: {count} requests ({percentage:.1f}%)")
                
                # Balance analysis for different policies
                if len(gpu_counts) == 2:
                    counts = list(gpu_counts.values())
                    diff = abs(counts[0] - counts[1])
                    balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
                    
                    print(f"\n📈 Load Balance Analysis:")
                    print(f"  Difference: {diff} requests")
                    print(f"  Balance Ratio: {balance_ratio:.2%}")
                    
                    if config.routing_policy == "round_robin":
                        if diff <= 2:
                            print("  ✅ Excellent balance for round-robin!")
                        else:
                            print("  ⚠️  Unexpected imbalance for round-robin")
                    elif config.routing_policy == "random":
                        if balance_ratio > 0.9:
                            print("  ✅ Good balance for random routing")
                        else:
                            print("  ℹ️  Some imbalance is normal for random routing")
                    elif config.routing_policy in ["cache_aware", "shortest_queue"]:
                        print(f"  ℹ️  Balance depends on workload for {config.routing_policy}")
        else:
            print("\n❌ No Prometheus data collected. Check if router was started with Prometheus support.")
        
        # Compare with CSV tracking
        if results.get('gpu_metrics'):
            print("\n📋 Comparison: CSV vs Prometheus Tracking")
            print("-" * 50)
            print("GPU | CSV Count | Prometheus Count | Difference")
            print("-" * 50)
            
            for gpu_id in sorted(set(list(results.get('gpu_metrics', {}).keys()) + 
                                    list(results.get('prometheus_gpu_counts', {}).keys()))):
                csv_count = results['gpu_metrics'].get(gpu_id, {}).get('completed_requests', 0)
                prom_count = results.get('prometheus_gpu_counts', {}).get(gpu_id, 0)
                diff = abs(csv_count - prom_count)
                print(f" {gpu_id}  |    {csv_count:4d}   |      {prom_count:4d}       |    {diff:4d}")
        
        # Save results summary
        print(f"\n💾 Results saved to: {config.output_dir}")
        print(f"  - CSV: {results.get('exported_files', {}).get('router_csv', 'N/A')}")
        print(f"  - JSON: {results.get('exported_files', {}).get('router_json', 'N/A')}")
        
        # Prometheus endpoint info
        print(f"\n🌐 Prometheus metrics endpoint: http://localhost:{config.router_config.prometheus_port}/metrics")
        print("   (You can access this URL in your browser while the test is running)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
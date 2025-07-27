#!/usr/bin/env python3
"""Test script to verify Prometheus-based request tracking."""

import asyncio
import logging
from sglang_test_framework import RoutingTest, RoutingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    # Configure test with Prometheus enabled
    config = RoutingConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        num_gpus=2,
        gpu_ids=[0, 1],
        routing_policy="round_robin",  # Use round_robin for predictable distribution
        request_rate=20.0,
        num_prompts=100,
        output_dir="./prometheus_test_results"
    )
    
    # Enable Prometheus metrics
    config.router_config.prometheus_port = 29000
    config.router_config.prometheus_host = "0.0.0.0"
    
    # Run test
    test = RoutingTest(config)
    results = test.run()
    
    # Analyze results
    test.analyze_results(results)
    
    # Print Prometheus data summary
    print("\n" + "="*80)
    print("PROMETHEUS TRACKING SUMMARY")
    print("="*80)
    
    if results.get('prometheus_worker_counts'):
        print("\nWorker-level counts from Prometheus:")
        for worker, count in sorted(results['prometheus_worker_counts'].items()):
            print(f"  {worker}: {count} requests")
    
    if results.get('prometheus_gpu_counts'):
        print("\nGPU-level counts from Prometheus:")
        total_requests = sum(results['prometheus_gpu_counts'].values())
        for gpu_id, count in sorted(results['prometheus_gpu_counts'].items()):
            percentage = (count / total_requests * 100) if total_requests > 0 else 0
            print(f"  GPU {gpu_id}: {count} requests ({percentage:.1f}%)")
        
        # Check distribution balance for round_robin
        if config.routing_policy == "round_robin" and len(results['prometheus_gpu_counts']) == 2:
            counts = list(results['prometheus_gpu_counts'].values())
            diff = abs(counts[0] - counts[1])
            print(f"\nBalance check: Difference = {diff} requests")
            if diff <= 2:  # Allow small difference due to startup/shutdown
                print("✅ Round-robin distribution is balanced!")
            else:
                print("⚠️ Round-robin distribution seems imbalanced")
    
    # Save visualization
    test.visualize_results(results)
    import matplotlib.pyplot as plt
    plt.savefig('./prometheus_test_results/request_distribution.png')
    print("\nVisualization saved to: ./prometheus_test_results/request_distribution.png")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test all routing policies with Prometheus tracking.
Compare how different policies distribute requests across GPUs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sglang_test_framework import RoutingTest, RoutingConfig
import matplotlib.pyplot as plt
import numpy as np

def test_policy(policy_name, num_requests=200):
    """Test a single routing policy."""
    config = RoutingConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        num_gpus=2,
        gpu_ids=[0, 1],
        routing_policy=policy_name,
        num_prompts=num_requests,
        request_rate=50.0,
        dataset_name="random",
        random_input_len=512,
        random_output_len=128,
        output_dir=f"./results/policy_{policy_name}",
        warmup_requests=10,
        verbose=False,  # Less verbose for multiple tests
    )
    
    # Enable Prometheus
    config.router_config.prometheus_port = 29000
    
    print(f"\n{'='*60}")
    print(f"Testing {policy_name.upper()} routing policy")
    print(f"{'='*60}")
    
    test = RoutingTest(config)
    results = test.run()
    
    # Extract key metrics
    prometheus_counts = results.get('prometheus_gpu_counts', {})
    
    if prometheus_counts:
        total = sum(prometheus_counts.values())
        distribution = {}
        for gpu_id in sorted(prometheus_counts.keys()):
            count = prometheus_counts[gpu_id]
            percentage = (count / total * 100) if total > 0 else 0
            distribution[gpu_id] = {
                'count': count,
                'percentage': percentage
            }
            print(f"  GPU {gpu_id}: {count} requests ({percentage:.1f}%)")
        
        # Calculate balance metric
        counts = list(prometheus_counts.values())
        if len(counts) == 2:
            balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
            print(f"  Balance ratio: {balance_ratio:.2%}")
        
        return {
            'policy': policy_name,
            'distribution': distribution,
            'total_requests': total,
            'success_rate': results['success_rate'],
            'duration': results['test_duration']
        }
    else:
        print("  ❌ No Prometheus data collected")
        return None

def visualize_comparison(results_list):
    """Create comparison visualization of all policies."""
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Routing Policy Comparison', fontsize=16)
    
    # 1. Request distribution bar chart
    policies = []
    gpu0_counts = []
    gpu1_counts = []
    
    for result in valid_results:
        policies.append(result['policy'])
        gpu0_counts.append(result['distribution'].get(0, {}).get('percentage', 0))
        gpu1_counts.append(result['distribution'].get(1, {}).get('percentage', 0))
    
    x = np.arange(len(policies))
    width = 0.35
    
    ax1.bar(x - width/2, gpu0_counts, width, label='GPU 0', alpha=0.8)
    ax1.bar(x + width/2, gpu1_counts, width, label='GPU 1', alpha=0.8)
    
    ax1.set_xlabel('Routing Policy')
    ax1.set_ylabel('Request Distribution (%)')
    ax1.set_title('Request Distribution by Policy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add 50% line for reference
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Perfect Balance')
    
    # 2. Balance ratio comparison
    balance_ratios = []
    for result in valid_results:
        dist = result['distribution']
        if 0 in dist and 1 in dist:
            counts = [dist[0]['count'], dist[1]['count']]
            ratio = min(counts) / max(counts) if max(counts) > 0 else 0
            balance_ratios.append(ratio * 100)
        else:
            balance_ratios.append(0)
    
    ax2.bar(policies, balance_ratios, alpha=0.8, color=['green' if r > 90 else 'orange' if r > 80 else 'red' for r in balance_ratios])
    ax2.set_xlabel('Routing Policy')
    ax2.set_ylabel('Balance Ratio (%)')
    ax2.set_title('Load Balance Quality')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add reference lines
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Balance')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Good Balance')
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Poor Balance')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./results/routing_policy_comparison.png', dpi=150)
    print("\n📊 Visualization saved to: ./results/routing_policy_comparison.png")

def main():
    print("🚀 Testing All Routing Policies with Prometheus Tracking")
    print("=" * 80)
    
    # Test each policy
    policies = ["round_robin", "random", "cache_aware", "shortest_queue"]
    results = []
    
    for policy in policies:
        try:
            result = test_policy(policy, num_requests=500)  # 500 requests per policy
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ❌ Failed to test {policy}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n📊 Policy Performance Comparison:")
    print("-" * 60)
    print(f"{'Policy':<15} {'GPU 0':<10} {'GPU 1':<10} {'Balance':<10} {'Duration':<10}")
    print("-" * 60)
    
    for result in results:
        policy = result['policy']
        gpu0_pct = result['distribution'].get(0, {}).get('percentage', 0)
        gpu1_pct = result['distribution'].get(1, {}).get('percentage', 0)
        
        counts = [result['distribution'].get(0, {}).get('count', 0),
                  result['distribution'].get(1, {}).get('count', 0)]
        balance = min(counts) / max(counts) * 100 if max(counts) > 0 else 0
        
        duration = result['duration']
        
        print(f"{policy:<15} {gpu0_pct:>6.1f}%    {gpu1_pct:>6.1f}%    {balance:>6.1f}%    {duration:>6.1f}s")
    
    # Create visualization
    visualize_comparison(results)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
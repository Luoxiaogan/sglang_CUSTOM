import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable server logs by setting environment variable before importing
# You can also set SGLANG_TEST_LOG_LEVEL=DEBUG for more detailed logs
os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'

import asyncio
import logging
import time
from sglang_test_framework import RoutingConfig
from sglang_test_framework.tests.routing_test import RoutingTest

# Get logger for this module
logger = logging.getLogger(__name__)

async def run_routing_test():
    logger.info("Starting SGLang routing test...")
    
    # 1. 配置路由测试
    logger.info("Setting up routing test configuration...")
    config = RoutingConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",  # Model path
        num_gpus=2,                          # Number of GPUs to use (reduced from 4 to avoid Triton errors)
        gpu_ids=[0, 1],                      # Specific GPU IDs
        routing_policy="cache_aware",        # Routing policy: cache_aware, round_robin, random, shortest_queue
        request_rate=50.0,                   # 50 requests per second
        num_prompts=1000,                    # Total number of requests
        dataset_name="random",
        random_input_len=512,
        random_output_len=128,
        random_range_ratio=0.5,              # Variation in request lengths
        
        # Server configuration (same for all servers)
        max_running_requests=256,
        mem_fraction_static=0.9,
        chunked_prefill_size=8192,
        
        # Metrics collection
        collect_per_node_metrics=True,       # Collect metrics for each server
        save_detailed_results=True,
        
        # Output directory
        output_dir="./results/routing_test",
    )
    logger.info(f"Test configuration: {config.num_prompts} prompts across {config.num_gpus} GPUs")
    logger.info(f"Routing policy: {config.routing_policy}")
    
    # 2. 创建路由测试实例
    logger.info("Initializing routing test...")
    routing_test = RoutingTest(config)
    
    # 3. 运行测试
    logger.info("Running routing test...")
    try:
        results = await routing_test._run_async()
        logger.info("Routing test completed successfully!")
        
        # 4. 分析结果
        logger.info("Analyzing results...")
        
        # Print summary
        print("\n" + "=" * 80)
        print(" " * 30 + "ROUTING TEST SUMMARY")
        print("=" * 80)
        
        router_metrics = results['router_metrics']
        print(f"\n📊 Overall Performance:")
        print(f"  Total Requests: {results['num_requests']}")
        print(f"  Success Rate: {results['success_rate'] * 100:.1f}%")
        print(f"  Test Duration: {results['test_duration']:.1f}s")
        print(f"  Request Throughput: {router_metrics.get('request_throughput', 0):.2f} req/s")
        
        # Token throughput
        print(f"\n  Token Throughput:")
        print(f"    Input: {router_metrics.get('input_token_throughput', 0):.0f} tok/s")
        print(f"    Output: {router_metrics.get('output_token_throughput', 0):.0f} tok/s")
        print(f"    Total: {router_metrics.get('total_token_throughput', 0):.0f} tok/s")
        
        # Latency metrics
        print(f"\n⏱  Latency Metrics:")
        print(f"  Server Latency (ms):")
        print(f"    Mean: {router_metrics.get('mean_server_latency', 0):.1f}")
        print(f"    P95: {router_metrics.get('p95_server_latency', 0):.1f}")
        print(f"    P99: {router_metrics.get('p99_server_latency', 0):.1f}")
        
        print(f"\n  Total Latency (ms):")
        print(f"    Mean: {router_metrics.get('mean_total_latency', 0):.1f}")
        print(f"    P95: {router_metrics.get('p95_total_latency', 0):.1f}")
        print(f"    P99: {router_metrics.get('p99_total_latency', 0):.1f}")
        
        # Queue metrics
        print(f"\n  Queue Metrics:")
        print(f"    Mean Queue Time: {router_metrics.get('mean_queue_time', 0):.1f} ms")
        print(f"    Mean Queue Length: {router_metrics.get('mean_queue_length', 0):.1f}")
        print(f"    Max Queue Length: {router_metrics.get('max_queue_length', 0)}")
        
        # Per-server distribution
        if results.get('server_metrics'):
            print(f"\n📈 Request Distribution Across Servers:")
            total_completed = router_metrics.get('completed_requests', 0)
            
            for server_id, metrics in sorted(results['server_metrics'].items()):
                completed = metrics.get('completed_requests', 0)
                percentage = (completed / total_completed * 100) if total_completed > 0 else 0
                
                # Create a simple bar chart
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                print(f"\n  {server_id}:")
                print(f"    Requests: {completed:4d} ({percentage:5.1f}%) |{bar}|")
                print(f"    Mean Latency: {metrics.get('mean_server_latency', 0):.1f} ms")
                print(f"    Throughput: {metrics.get('request_throughput', 0):.2f} req/s")
        
        # Routing policy effectiveness
        print(f"\n🎯 Routing Policy Analysis ({config.routing_policy}):")
        if results.get('server_metrics'):
            request_counts = [m.get('completed_requests', 0) for m in results['server_metrics'].values()]
            if request_counts:
                min_requests = min(request_counts)
                max_requests = max(request_counts)
                imbalance = (max_requests - min_requests) / max(min_requests, 1) * 100
                print(f"  Load Imbalance: {imbalance:.1f}%")
                print(f"  Min/Max Requests: {min_requests}/{max_requests}")
                
                if config.routing_policy == "cache_aware":
                    print(f"  Note: Cache-aware routing may show imbalance due to cache affinity")
                elif config.routing_policy == "round_robin":
                    print(f"  Note: Round-robin should show near-perfect balance")
        
        # Export locations
        if 'exported_files' in results:
            print(f"\n💾 Detailed Results Saved:")
            for name, path in results['exported_files'].items():
                print(f"  {name}: {path}")
        
        print("=" * 80 + "\n")
        
        # Create visualizations
        routing_test.visualize_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Routing test failed: {e}", exc_info=True)
        raise

async def run_routing_comparison():
    """Run tests with different routing policies for comparison."""
    logger.info("Running routing policy comparison...")
    
    policies = ["cache_aware", "round_robin", "random", "shortest_queue"]
    all_results = {}
    
    for policy in policies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {policy.upper()} routing policy...")
        logger.info(f"{'='*60}")
        
        config = RoutingConfig(
            model_path="/data/pretrained_models/Llama-2-7b-hf",
            num_gpus=2,
            gpu_ids=[0, 1],
            routing_policy=policy,
            request_rate=50.0,
            num_prompts=500,  # Smaller test for comparison
            dataset_name="random",
            random_input_len=512,
            random_output_len=128,
            collect_per_node_metrics=True,
            output_dir=f"./results/routing_comparison/{policy}",
        )
        
        routing_test = RoutingTest(config)
        
        try:
            results = await routing_test._run_async()
            all_results[policy] = results
            logger.info(f"✅ {policy} test completed")
        except Exception as e:
            logger.error(f"❌ {policy} test failed: {e}")
            all_results[policy] = None
    
    # Compare results
    print("\n" + "=" * 80)
    print(" " * 25 + "ROUTING POLICY COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for policy, results in all_results.items():
        if results:
            metrics = results['router_metrics']
            server_metrics = results.get('server_metrics', {})
            
            # Calculate load imbalance
            request_counts = [m.get('completed_requests', 0) for m in server_metrics.values()]
            if request_counts:
                min_req = min(request_counts)
                max_req = max(request_counts)
                imbalance = (max_req - min_req) / max(min_req, 1) * 100
            else:
                imbalance = 0
            
            comparison_data.append({
                'Policy': policy,
                'Success Rate': f"{results['success_rate'] * 100:.1f}%",
                'Throughput': f"{metrics.get('request_throughput', 0):.2f} req/s",
                'Mean Latency': f"{metrics.get('mean_total_latency', 0):.1f} ms",
                'P99 Latency': f"{metrics.get('p99_total_latency', 0):.1f} ms",
                'Load Imbalance': f"{imbalance:.1f}%"
            })
    
    # Print comparison table
    if comparison_data:
        headers = list(comparison_data[0].keys())
        col_widths = {h: max(len(h), max(len(str(row.get(h, ''))) for row in comparison_data)) + 2 
                     for h in headers}
        
        # Print header
        print("\n" + " ".join(h.ljust(col_widths[h]) for h in headers))
        print("-" * sum(col_widths.values()))
        
        # Print data
        for row in comparison_data:
            print(" ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers))
    
    print("\n" + "=" * 80 + "\n")

# 运行测试
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SGLang Testing Framework - Routing Test")
    logger.info("="*60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="SGLang Routing Test")
    parser.add_argument("--compare", action="store_true", 
                       help="Run comparison of different routing policies")
    parser.add_argument("--num-gpus", type=int, default=2,
                       help="Number of GPUs to use (default: 2)")
    parser.add_argument("--policy", type=str, default="cache_aware",
                       choices=["cache_aware", "round_robin", "random", "shortest_queue"],
                       help="Routing policy to use (default: cache_aware)")
    parser.add_argument("--num-prompts", type=int, default=1000,
                       help="Number of test prompts (default: 1000)")
    parser.add_argument("--request-rate", type=float, default=50.0,
                       help="Request rate in req/s (default: 50.0)")
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Run comparison test
            asyncio.run(run_routing_comparison())
        else:
            # Run single test with specified parameters
            # Update the config in run_routing_test() based on args
            async def run_custom_test():
                config = RoutingConfig(
                    model_path="/data/pretrained_models/Llama-2-7b-hf",
                    num_gpus=args.num_gpus,
                    routing_policy=args.policy,
                    request_rate=args.request_rate,
                    num_prompts=args.num_prompts,
                    dataset_name="random",
                    random_input_len=512,
                    random_output_len=128,
                    collect_per_node_metrics=True,
                    save_detailed_results=True,
                    output_dir=f"./results/routing_test_{args.policy}",
                )
                
                routing_test = RoutingTest(config)
                results = await routing_test._run_async()
                routing_test.visualize_results(results)
                return results
            
            asyncio.run(run_custom_test())
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
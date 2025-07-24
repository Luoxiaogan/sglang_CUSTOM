"""Multi-node routing test runner for SGLang.

Adapted from SGLang's benchmarking and router patterns.
Source references:
- python/sglang/bench_serving_new.py
- python/sglang_router/
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import sys

from ..config.routing_config import RoutingConfig
from ..core.server_manager import ServerManager, SGLangServer, RouterManager
from ..core.request_generator import RequestGenerator, RequestSender, generate_and_send_requests
from ..core.metrics_collector import MetricsCollector
from ..core.result_manager import ResultManager
from ..strategies.routing import RandomRouting, RoundRobinRouting, ShortestQueueRouting

logger = logging.getLogger(__name__)


def check_router_installed():
    """Check if sglang-router is installed."""
    if importlib.util.find_spec("sglang_router") is None:
        error_msg = """
ERROR: sglang-router is not installed!

The routing test requires the SGLang Router component to be installed.
Please install it using one of the following methods:

1. Install from PyPI:
   pip install sglang-router

2. Install from source (recommended for development):
   cd /path/to/sglang/sgl-router
   pip install -e .

For more information, see: https://github.com/sgl-project/sglang/tree/main/sgl-router
"""
        logger.error(error_msg)
        raise ImportError("sglang-router is not installed")
    else:
        logger.info("sglang-router is installed and available")


class RoutingTest:
    """Test runner for multi-node SGLang deployment with routing."""
    
    def __init__(self, config: RoutingConfig):
        """Initialize routing test.
        
        Args:
            config: Routing test configuration
        """
        # Check if router is installed before proceeding
        check_router_installed()
        
        self.config = config
        self.server_manager = ServerManager()
        self.request_generator = RequestGenerator(config.tokenizer_path)
        self.metrics_collectors: Dict[str, MetricsCollector] = {}
        self.router_metrics_collector: Optional[MetricsCollector] = None
        self.result_manager = ResultManager(config.output_dir)
        self.servers: List[SGLangServer] = []
        self.router: Optional[RouterManager] = None
        
    async def _run_async(self) -> Dict[str, Any]:
        """Async implementation of test run."""
        start_time = time.time()
        
        try:
            # 1. Launch multiple servers
            logger.info(f"Launching {self.config.num_gpus} SGLang servers")
            self.servers = await self.server_manager.launch_multiple_servers(
                self.config.servers_config
            )
            
            # 2. Initialize per-server metrics collectors
            if self.config.collect_per_node_metrics:
                for server in self.servers:
                    server_url = f"http://{server.config.host}:{server.config.port}"
                    # Convert MetricsConfig to dict manually since it doesn't have to_dict()
                    metrics_config_dict = {
                        "collection_interval": self.config.metrics_config.collection_interval,
                        "collect_throughput": self.config.metrics_config.collect_throughput,
                        "collect_latency": self.config.metrics_config.collect_latency,
                        "collect_queue_metrics": self.config.metrics_config.collect_queue_metrics,
                        "collect_resource_usage": self.config.metrics_config.collect_resource_usage,
                        "collect_cache_metrics": self.config.metrics_config.collect_cache_metrics,
                        "latency_percentiles": self.config.metrics_config.latency_percentiles,
                        "export_format": self.config.metrics_config.export_format,
                        "export_interval": self.config.metrics_config.export_interval,
                        "enable_dashboard": self.config.metrics_config.enable_dashboard,
                        "dashboard_port": self.config.metrics_config.dashboard_port
                    }
                    collector = MetricsCollector(
                        config=metrics_config_dict,
                        server_url=server_url
                    )
                    self.metrics_collectors[server.config.server_id] = collector
            
            # 3. Launch router
            worker_urls = self.config.get_worker_urls()
            logger.info(f"Launching router with policy: {self.config.routing_policy}")
            self.router = await self.server_manager.launch_router(
                self.config.router_config,
                worker_urls
            )
            
            router_url = f"http://{self.config.router_config.host}:{self.config.router_config.port}"
            
            # 4. Initialize router metrics collector
            # Convert MetricsConfig to dict manually since it doesn't have to_dict()
            metrics_config_dict = {
                "collection_interval": self.config.metrics_config.collection_interval,
                "collect_throughput": self.config.metrics_config.collect_throughput,
                "collect_latency": self.config.metrics_config.collect_latency,
                "collect_queue_metrics": self.config.metrics_config.collect_queue_metrics,
                "collect_resource_usage": self.config.metrics_config.collect_resource_usage,
                "collect_cache_metrics": self.config.metrics_config.collect_cache_metrics,
                "latency_percentiles": self.config.metrics_config.latency_percentiles,
                "export_format": self.config.metrics_config.export_format,
                "export_interval": self.config.metrics_config.export_interval,
                "enable_dashboard": self.config.metrics_config.enable_dashboard,
                "dashboard_port": self.config.metrics_config.dashboard_port
            }
            self.router_metrics_collector = MetricsCollector(
                config=metrics_config_dict
            )
            
            # 5. Warm up servers
            if self.config.warmup_requests > 0:
                logger.info(f"Warming up servers with {self.config.warmup_requests} requests")
                await self._warmup_servers(router_url)
            
            # 6. Start metrics collection
            self.router_metrics_collector.start_collection()
            for collector in self.metrics_collectors.values():
                collector.start_collection()
            
            # 7. Handle node failures if enabled
            if self.config.enable_node_failures and self.config.failure_schedule:
                failure_task = asyncio.create_task(
                    self._handle_node_failures()
                )
            else:
                failure_task = None
            
            # 8. Generate and send requests through router
            logger.info(f"Starting test with {self.config.num_prompts} requests")
            api_url = f"{router_url}/generate"
            
            request_config = {
                "num_prompts": self.config.num_prompts,
                "dataset_name": self.config.dataset_name,
                "dataset_path": self.config.dataset_path,
                "random_input_len": self.config.random_input_len,
                "random_output_len": self.config.random_output_len,
                "random_range_ratio": self.config.random_range_ratio,
                "seed": self.config.seed,
                "request_rate": self.config.request_rate
            }
            
            # Send requests and collect results
            async with RequestSender() as sender:
                async for result in generate_and_send_requests(
                    self.request_generator, 
                    sender, 
                    api_url, 
                    request_config
                ):
                    self.router_metrics_collector.record_request_complete(result)
            
            # Cancel failure task if running
            if failure_task:
                failure_task.cancel()
                try:
                    await failure_task
                except asyncio.CancelledError:
                    pass
            
            # 9. Stop metrics collection
            self.router_metrics_collector.stop_collection()
            for collector in self.metrics_collectors.values():
                collector.stop_collection()
            
            # 10. Calculate final metrics
            end_time = time.time()
            router_metrics = self.router_metrics_collector.get_aggregated_metrics()
            
            # Collect per-server metrics
            server_metrics = {}
            if self.config.collect_per_node_metrics:
                for server_id, collector in self.metrics_collectors.items():
                    server_metrics[server_id] = collector.get_aggregated_metrics().__dict__
            
            # 11. Prepare results
            results = {
                "config": self.config.to_dict(),
                "router_metrics": router_metrics.__dict__,
                "server_metrics": server_metrics,
                "test_duration": end_time - start_time,
                "router_url": router_url,
                "num_servers": len(self.servers),
                "num_requests": self.config.num_prompts,
                "success_rate": router_metrics.completed_requests / router_metrics.total_requests
                if router_metrics.total_requests > 0 else 0
            }
            
            # 12. Export detailed results
            if self.config.save_detailed_results:
                csv_path = self.router_metrics_collector.export_metrics("csv")
                json_path = self.router_metrics_collector.export_metrics("json")
                results["exported_files"] = {
                    "router_csv": str(csv_path),
                    "router_json": str(json_path)
                }
                
                # Export per-server metrics
                for server_id, collector in self.metrics_collectors.items():
                    server_csv = collector.export_metrics("csv", 
                        f"server_{server_id}_metrics.csv")
                    results["exported_files"][f"{server_id}_csv"] = str(server_csv)
            
            return results
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            # Cleanup
            if self.router_metrics_collector:
                self.router_metrics_collector.stop_collection()
            for collector in self.metrics_collectors.values():
                collector.stop_collection()
            self.server_manager.stop_all()
    
    async def _warmup_servers(self, router_url: str):
        """Warm up all servers through the router."""
        warmup_config = {
            "num_prompts": self.config.warmup_requests,
            "dataset_name": "random",
            "random_input_len": 128,
            "random_output_len": 16,
            "random_range_ratio": 0.1,
            "seed": 42,
            "request_rate": float('inf')  # Send all at once
        }
        
        api_url = f"{router_url}/generate"
        
        async with RequestSender() as sender:
            warmup_results = []
            async for result in generate_and_send_requests(
                self.request_generator,
                sender,
                api_url,
                warmup_config
            ):
                warmup_results.append(result)
        
        successful = sum(1 for r in warmup_results if r.success)
        logger.info(f"Warmup complete: {successful}/{len(warmup_results)} successful")
    
    async def _handle_node_failures(self):
        """Simulate node failures according to schedule."""
        start_time = time.time()
        
        for failure_time, failed_nodes in sorted(self.config.failure_schedule.items()):
            # Wait until failure time
            wait_time = failure_time - (time.time() - start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Simulate failures/recoveries
            if failed_nodes:
                # Fail nodes
                for node_idx in failed_nodes:
                    if node_idx < len(self.servers):
                        server = self.servers[node_idx]
                        logger.warning(f"Simulating failure of {server.config.server_id}")
                        worker_url = f"http://{server.config.host}:{server.config.port}"
                        await self.router.remove_worker(worker_url)
            else:
                # Recover all nodes
                logger.info("Recovering all failed nodes")
                for server in self.servers:
                    worker_url = f"http://{server.config.host}:{server.config.port}"
                    await self.router.add_worker(worker_url)
    
    def run(self) -> Dict[str, Any]:
        """Run the routing test.
        
        Returns:
            Dictionary containing test results and metrics
        """
        return asyncio.run(self._run_async())
    
    def visualize_results(self, results: Dict[str, Any]):
        """Create visualizations for test results.
        
        Args:
            results: Test results dictionary
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Routing Test Results - {self.config.routing_policy.upper()} Policy", 
                     fontsize=16)
        
        # 1. Request distribution across servers
        if results['server_metrics']:
            ax = axes[0, 0]
            server_ids = list(results['server_metrics'].keys())
            request_counts = [results['server_metrics'][sid]['completed_requests'] 
                            for sid in server_ids]
            
            ax.bar(server_ids, request_counts)
            ax.set_xlabel("Server ID")
            ax.set_ylabel("Completed Requests")
            ax.set_title("Request Distribution Across Servers")
            
            # Add percentage labels
            total_requests = sum(request_counts)
            for i, (sid, count) in enumerate(zip(server_ids, request_counts)):
                percentage = (count / total_requests * 100) if total_requests > 0 else 0
                ax.text(i, count, f"{percentage:.1f}%", ha='center', va='bottom')
        
        # 2. Latency comparison
        ax = axes[0, 1]
        router_metrics = results['router_metrics']
        
        latency_types = ['Server', 'Total', 'Queue']
        mean_latencies = [
            router_metrics.get('mean_server_latency', 0),
            router_metrics.get('mean_total_latency', 0),
            router_metrics.get('mean_queue_time', 0)
        ]
        p95_latencies = [
            router_metrics.get('p95_server_latency', 0),
            router_metrics.get('p95_total_latency', 0),
            router_metrics.get('p95_queue_time', 0)
        ]
        
        x = np.arange(len(latency_types))
        width = 0.35
        
        ax.bar(x - width/2, mean_latencies, width, label='Mean', alpha=0.8)
        ax.bar(x + width/2, p95_latencies, width, label='P95', alpha=0.8)
        
        ax.set_xlabel("Latency Type")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(latency_types)
        ax.legend()
        
        # 3. Throughput over time (if available)
        ax = axes[1, 0]
        if hasattr(self.router_metrics_collector, 'get_concurrency_over_time'):
            concurrency_df = self.router_metrics_collector.get_concurrency_over_time()
            if not concurrency_df.empty:
                # Plot concurrency over time
                server_data = concurrency_df[concurrency_df['event_type'] == 'server_poll']
                if not server_data.empty:
                    ax.plot(server_data['timestamp'] - server_data['timestamp'].min(),
                           server_data.get('server_running', 0), 
                           label='Running Requests')
                    ax.plot(server_data['timestamp'] - server_data['timestamp'].min(),
                           server_data.get('server_queued', 0), 
                           label='Queued Requests')
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Request Count")
                    ax.set_title("Concurrency Over Time")
                    ax.legend()
        
        # 4. Performance summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""Performance Summary:
        
Total Requests: {results['num_requests']}
Success Rate: {results['success_rate'] * 100:.1f}%
Test Duration: {results['test_duration']:.1f}s

Request Throughput: {router_metrics.get('request_throughput', 0):.2f} req/s
Token Throughput:
  Input: {router_metrics.get('input_token_throughput', 0):.0f} tok/s
  Output: {router_metrics.get('output_token_throughput', 0):.0f} tok/s

Mean Latencies:
  Server: {router_metrics.get('mean_server_latency', 0):.1f} ms
  Total: {router_metrics.get('mean_total_latency', 0):.1f} ms
  Queue: {router_metrics.get('mean_queue_time', 0):.1f} ms

P99 Latencies:
  Server: {router_metrics.get('p99_server_latency', 0):.1f} ms
  Total: {router_metrics.get('p99_total_latency', 0):.1f} ms"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(self.config.output_dir) / f"routing_test_results_{int(time.time())}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        
        # Show plot
        plt.show()
        
        # Also print results
        self._print_results(results)
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "=" * 80)
        print(" " * 30 + "ROUTING TEST RESULTS")
        print("=" * 80)
        
        # Configuration summary
        print("\n📋 Test Configuration:")
        print(f"  Model: {self.config.model_path}")
        print(f"  Number of GPUs: {self.config.num_gpus}")
        print(f"  Routing Policy: {self.config.routing_policy}")
        print(f"  Request Rate: {self.config.request_rate} req/s")
        print(f"  Total Requests: {results['num_requests']}")
        print(f"  Dataset: {self.config.dataset_name}")
        
        # Router metrics
        router_metrics = results['router_metrics']
        print("\n📊 Router Performance Metrics:")
        print(f"  Success Rate: {results['success_rate'] * 100:.1f}%")
        print(f"  Request Throughput: {router_metrics.get('request_throughput', 0):.2f} req/s")
        print(f"  Token Throughput:")
        print(f"    Input: {router_metrics.get('input_token_throughput', 0):.0f} tok/s")
        print(f"    Output: {router_metrics.get('output_token_throughput', 0):.0f} tok/s")
        
        print("\n⏱  Latency Metrics:")
        print(f"  Server Latency (ms):")
        print(f"    Mean: {router_metrics.get('mean_server_latency', 0):.1f}")
        print(f"    P95: {router_metrics.get('p95_server_latency', 0):.1f}")
        print(f"    P99: {router_metrics.get('p99_server_latency', 0):.1f}")
        
        print(f"\n  Total Latency (ms):")
        print(f"    Mean: {router_metrics.get('mean_total_latency', 0):.1f}")
        print(f"    P95: {router_metrics.get('p95_total_latency', 0):.1f}")
        print(f"    P99: {router_metrics.get('p99_total_latency', 0):.1f}")
        
        print(f"\n  Queue Time (ms):")
        print(f"    Mean: {router_metrics.get('mean_queue_time', 0):.1f}")
        print(f"    P95: {router_metrics.get('p95_queue_time', 0):.1f}")
        
        # Per-server metrics
        if results['server_metrics']:
            print("\n📈 Per-Server Metrics:")
            for server_id, metrics in results['server_metrics'].items():
                completed = metrics.get('completed_requests', 0)
                total = metrics.get('total_requests', 0)
                percentage = (completed / results['router_metrics']['completed_requests'] * 100) \
                            if results['router_metrics']['completed_requests'] > 0 else 0
                
                print(f"\n  {server_id}:")
                print(f"    Completed Requests: {completed} ({percentage:.1f}%)")
                print(f"    Mean Server Latency: {metrics.get('mean_server_latency', 0):.1f} ms")
                print(f"    Request Throughput: {metrics.get('request_throughput', 0):.2f} req/s")
        
        # Test duration
        print(f"\n⏲  Test Duration: {results['test_duration']:.1f} seconds")
        
        # Exported files
        if 'exported_files' in results:
            print(f"\n💾 Exported Files:")
            for name, path in results['exported_files'].items():
                print(f"  {name}: {path}")
        
        print("=" * 80 + "\n")
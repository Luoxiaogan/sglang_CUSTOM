"""Single-node (single GPU) test runner for SGLang.

Adapted from SGLang's benchmarking patterns.
Source references:
- python/sglang/bench_serving_new.py
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from ..config.node_config import NodeConfig
from ..core.server_manager import ServerManager, SGLangServer
from ..core.request_generator import RequestGenerator, RequestSender, generate_and_send_requests
from ..core.metrics_collector import MetricsCollector
from ..core.result_manager import ResultManager

logger = logging.getLogger(__name__)


class NodeTest:
    """Test runner for single-node SGLang deployment."""
    
    def __init__(self, config: NodeConfig):
        """Initialize node test.
        
        Args:
            config: Node test configuration
        """
        self.config = config
        self.server_manager = ServerManager()
        self.request_generator = RequestGenerator(config.tokenizer_path)
        self.metrics_collector = None
        self.result_manager = ResultManager(config.output_dir)
        self.server: Optional[SGLangServer] = None
        
    async def _run_async(self) -> Dict[str, Any]:
        """Async implementation of test run."""
        start_time = time.time()
        
        try:
            # 1. Launch server
            logger.info(f"Launching SGLang server on GPU {self.config.gpu_id}")
            server_config = self.config.get_server_config()
            self.server = await self.server_manager.launch_server(server_config)
            server_url = f"http://{server_config.host}:{server_config.port}"
            
            # 2. Initialize metrics collector with server URL
            self.metrics_collector = MetricsCollector(
                config=self.config.metrics_config.to_dict() if self.config.metrics_config else {},
                server_url=server_url
            )
            
            # 3. Warm up server
            if self.config.warmup_requests > 0:
                logger.info(f"Warming up server with {self.config.warmup_requests} requests")
                await self._warmup_server(server_url)
            
            # 4. Start metrics collection
            self.metrics_collector.start_collection()
            
            # 5. Generate and send requests
            logger.info(f"Starting test with {self.config.num_prompts} requests")
            api_url = f"{server_url}/generate"
            
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
            
            # Handle dynamic parameter updates if enabled
            if self.config.enable_dynamic_params and self.config.param_update_schedule:
                update_task = asyncio.create_task(
                    self._handle_dynamic_updates()
                )
            else:
                update_task = None
            
            # Send requests and collect results
            async with RequestSender() as sender:
                async for result in generate_and_send_requests(
                    self.request_generator, 
                    sender, 
                    api_url, 
                    request_config
                ):
                    self.metrics_collector.record_request_complete(result)
            
            # Cancel update task if running
            if update_task:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
            
            # 6. Stop metrics collection
            self.metrics_collector.stop_collection()
            
            # 7. Calculate final metrics
            end_time = time.time()
            aggregated_metrics = self.metrics_collector.get_aggregated_metrics()
            
            # 8. Prepare results
            results = {
                "config": self.config.to_dict(),
                "metrics": aggregated_metrics.__dict__,
                "test_duration": end_time - start_time,
                "server_url": server_url,
                "num_requests": self.config.num_prompts,
                "success_rate": aggregated_metrics.completed_requests / aggregated_metrics.total_requests
                if aggregated_metrics.total_requests > 0 else 0
            }
            
            # 9. Export detailed results
            if self.config.save_detailed_results:
                csv_path = self.metrics_collector.export_metrics("csv")
                json_path = self.metrics_collector.export_metrics("json")
                results["exported_files"] = {
                    "csv": str(csv_path),
                    "json": str(json_path)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            # Cleanup
            if self.metrics_collector:
                self.metrics_collector.stop_collection()
            self.server_manager.stop_all()
    
    async def _warmup_server(self, server_url: str):
        """Warm up the server with initial requests."""
        warmup_config = {
            "num_prompts": self.config.warmup_requests,
            "dataset_name": "random",
            "random_input_len": 128,
            "random_output_len": 16,
            "random_range_ratio": 0.1,
            "seed": 42,
            "request_rate": float('inf')  # Send all at once
        }
        
        api_url = f"{server_url}/generate"
        
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
    
    async def _handle_dynamic_updates(self):
        """Handle dynamic parameter updates during test."""
        start_time = time.time()
        
        for update_time, params in sorted(self.config.param_update_schedule.items()):
            # Wait until update time
            wait_time = update_time - (time.time() - start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Apply updates
            for param, value in params.items():
                logger.info(f"Updating {param} to {value} at t={update_time}s")
                # Note: Actual update would require SGLang API support
                # For now, we log the intention
                logger.warning(f"Dynamic update of {param} not yet supported by SGLang")
    
    def run(self) -> Dict[str, Any]:
        """Run the node test.
        
        Returns:
            Dictionary containing test results and metrics
        """
        return asyncio.run(self._run_async())
    
    def analyze_results(self, results: Dict[str, Any]):
        """Analyze and display test results.
        
        Args:
            results: Test results dictionary
        """
        print("\n" + "=" * 60)
        print(" " * 20 + "NODE TEST RESULTS")
        print("=" * 60)
        
        # Configuration summary
        print("\n=Ë Test Configuration:")
        print(f"  Model: {self.config.model_path}")
        print(f"  GPU ID: {self.config.gpu_id}")
        print(f"  Max Running Requests: {self.config.max_running_requests}")
        print(f"  Request Rate: {self.config.request_rate} req/s")
        print(f"  Total Requests: {results['num_requests']}")
        print(f"  Dataset: {self.config.dataset_name}")
        
        # Metrics summary
        metrics = results['metrics']
        print("\n=Ê Performance Metrics:")
        print(f"  Success Rate: {results['success_rate'] * 100:.1f}%")
        print(f"  Request Throughput: {metrics.get('request_throughput', 0):.2f} req/s")
        print(f"  Token Throughput:")
        print(f"    Input: {metrics.get('input_token_throughput', 0):.0f} tok/s")
        print(f"    Output: {metrics.get('output_token_throughput', 0):.0f} tok/s")
        
        print("\nñ  Latency Metrics:")
        print(f"  Server Latency (ms):")
        print(f"    Mean: {metrics.get('mean_server_latency', 0):.1f}")
        print(f"    P95: {metrics.get('p95_server_latency', 0):.1f}")
        print(f"    P99: {metrics.get('p99_server_latency', 0):.1f}")
        
        print(f"\n  Total Latency (ms):")
        print(f"    Mean: {metrics.get('mean_total_latency', 0):.1f}")
        print(f"    P95: {metrics.get('p95_total_latency', 0):.1f}")
        print(f"    P99: {metrics.get('p99_total_latency', 0):.1f}")
        
        print(f"\n  Queue Time (ms):")
        print(f"    Mean: {metrics.get('mean_queue_time', 0):.1f}")
        print(f"    P95: {metrics.get('p95_queue_time', 0):.1f}")
        
        # GPU metrics if available
        if metrics.get('gpu_utilization', 0) > 0:
            print(f"\n=¥  GPU Metrics:")
            print(f"  Utilization: {metrics.get('gpu_utilization', 0):.1f}%")
            print(f"  Memory: {metrics.get('gpu_memory_used', 0):.1f} / "
                  f"{metrics.get('gpu_memory_total', 0):.1f} GB")
        
        # Test duration
        print(f"\nò  Test Duration: {results['test_duration']:.1f} seconds")
        
        # Exported files
        if 'exported_files' in results:
            print(f"\n=¾ Exported Files:")
            for fmt, path in results['exported_files'].items():
                print(f"  {fmt.upper()}: {path}")
        
        print("=" * 60 + "\n")
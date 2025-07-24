"""Metrics collection and analysis for SGLang testing.

Enhanced from SGLang's benchmarking patterns.
Source references:
- python/sglang/bench_serving_new.py
- python/sglang/bench_serving.py
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Deque
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

from .request_generator import RequestResult

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """A snapshot of metrics at a point in time."""
    timestamp: float
    metrics: Dict[str, Any]
    

@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time period."""
    # Request metrics
    completed_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    
    # Throughput metrics
    request_throughput: float = 0.0  # requests/sec
    input_token_throughput: float = 0.0  # tokens/sec
    output_token_throughput: float = 0.0  # tokens/sec
    total_token_throughput: float = 0.0  # tokens/sec
    
    # Latency metrics (in ms)
    mean_server_latency: float = 0.0
    median_server_latency: float = 0.0
    p95_server_latency: float = 0.0
    p99_server_latency: float = 0.0
    max_server_latency: float = 0.0
    
    mean_total_latency: float = 0.0
    median_total_latency: float = 0.0
    p95_total_latency: float = 0.0
    p99_total_latency: float = 0.0
    max_total_latency: float = 0.0
    
    mean_queue_time: float = 0.0
    median_queue_time: float = 0.0
    p95_queue_time: float = 0.0
    p99_queue_time: float = 0.0
    max_queue_time: float = 0.0
    
    mean_ttft: float = 0.0
    median_ttft: float = 0.0
    p95_ttft: float = 0.0
    p99_ttft: float = 0.0
    
    mean_itl: float = 0.0
    median_itl: float = 0.0
    p95_itl: float = 0.0
    p99_itl: float = 0.0
    max_itl: float = 0.0
    
    # Queue metrics
    mean_queue_depth: float = 0.0
    max_queue_depth: int = 0
    
    # System metrics
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    
    # Time period
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0


class MetricsCollector:
    """Collects and analyzes metrics during testing."""
    
    def __init__(self, config: Dict[str, Any] = None, server_url: str = None):
        self.config = config or {}
        self.collection_interval = self.config.get("collection_interval", 1.0)
        self.poll_interval = self.config.get("poll_interval", 0.1)  # Server polling interval
        self.server_url = server_url
        
        # Storage for results
        self.results: List[RequestResult] = []
        self.snapshots: List[MetricSnapshot] = []
        
        # Real-time metrics
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.queue_depths: Deque[Tuple[float, int]] = deque(maxlen=10000)
        
        # Server metrics time series
        self.server_metrics_history: List[Dict[str, Any]] = []
        
        # Collection control
        self.collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.polling_task: Optional[asyncio.Task] = None
        self.start_time: Optional[float] = None
        
        # Incremental saving
        self.incremental_save_path: Optional[str] = None
        self.incremental_save_interval: int = 100  # Save every N requests
        
    def start_collection(self):
        """Start metrics collection."""
        if self.collecting:
            logger.warning("Metrics collection already started")
            return
            
        self.collecting = True
        self.start_time = time.time()
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        # Start server polling if URL provided
        if self.server_url:
            self.polling_task = asyncio.create_task(self._server_polling_loop())
            
        logger.info("Started metrics collection")
        
    def stop_collection(self):
        """Stop metrics collection."""
        if not self.collecting:
            logger.warning("Metrics collection not started")
            return
            
        self.collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            self.collection_task = None
        if self.polling_task:
            self.polling_task.cancel()
            self.polling_task = None
        logger.info("Stopped metrics collection")
    
    def record_request_start(self, request_id: str):
        """Record that a request has started."""
        self.active_requests[request_id] = time.time()
        self.queue_depths.append((time.time(), len(self.active_requests)))
        
    def record_request_complete(self, result: RequestResult):
        """Record a completed request."""
        self.results.append(result)
        
        if result.request.request_id in self.active_requests:
            del self.active_requests[result.request.request_id]
            self.queue_depths.append((time.time(), len(self.active_requests)))
            
        # Log progress every 100 requests
        if len(self.results) % 100 == 0:
            success_rate = sum(1 for r in self.results if r.success) / len(self.results) * 100
            logger.info(f"Completed {len(self.results)} requests, success rate: {success_rate:.1f}%")
            
        # Incremental save if enabled
        if self.incremental_save_path and len(self.results) % self.incremental_save_interval == 0:
            self._save_incremental()
    
    async def _collection_loop(self):
        """Background loop for collecting system metrics."""
        while self.collecting:
            try:
                # Collect current metrics
                snapshot = await self._collect_snapshot()
                self.snapshots.append(snapshot)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def _server_polling_loop(self):
        """Poll server for real-time metrics."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            while self.collecting:
                try:
                    url = f"{self.server_url}/get_server_info"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            
                            # Extract key metrics from server
                            server_data = {
                                "timestamp": time.time(),
                                "num_running_reqs": 0,
                                "num_queue_reqs": 0,
                                "gen_throughput": 0.0,
                                "prompt_tokens_total": 0,
                                "generation_tokens_total": 0
                            }
                            
                            # Parse internal states if available
                            if "internal_states" in data:
                                states = data["internal_states"]
                                if isinstance(states, str):
                                    # Parse string format from bench_serving_new.py pattern
                                    import re
                                    running_match = re.search(r"#running-req: (\d+)", states)
                                    queue_match = re.search(r"#queue-req: (\d+)", states)
                                    if running_match:
                                        server_data["num_running_reqs"] = int(running_match.group(1))
                                    if queue_match:
                                        server_data["num_queue_reqs"] = int(queue_match.group(1))
                                
                            # Get throughput from proper fields
                            server_data["gen_throughput"] = data.get("gen_throughput", 0.0)
                            server_data["prompt_tokens_total"] = data.get("prompt_tokens_total", 0)
                            server_data["generation_tokens_total"] = data.get("generation_tokens_total", 0)
                            
                            self.server_metrics_history.append(server_data)
                            
                    await asyncio.sleep(self.poll_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error polling server metrics: {e}")
                    await asyncio.sleep(self.poll_interval)
    
    async def _collect_snapshot(self) -> MetricSnapshot:
        """Collect a snapshot of current metrics."""
        metrics = {
            "timestamp": time.time(),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.results),
        }
        
        # Add latest server metrics if available
        if self.server_metrics_history:
            latest_server = self.server_metrics_history[-1]
            metrics["server_running_reqs"] = latest_server["num_running_reqs"]
            metrics["server_queue_reqs"] = latest_server["num_queue_reqs"]
            metrics["server_throughput"] = latest_server["gen_throughput"]
        
        # Collect GPU metrics if available
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Assuming GPU 0 for now, can be extended
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization"] = util.gpu
            
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_memory_used"] = mem_info.used / (1024**3)  # GB
            metrics["gpu_memory_total"] = mem_info.total / (1024**3)  # GB
            metrics["gpu_memory_utilization"] = (mem_info.used / mem_info.total) * 100
            
        except Exception as e:
            logger.debug(f"Could not collect GPU metrics: {e}")
            
        return MetricSnapshot(timestamp=time.time(), metrics=metrics)
    
    def get_concurrency_over_time(self) -> pd.DataFrame:
        """Calculate concurrency over time from request data.
        
        Returns DataFrame with columns: timestamp, concurrency, queue_depth
        """
        events = []
        
        # Create events for each request
        for r in self.results:
            if r.request.send_time:
                events.append((r.request.send_time, 1, "start"))
            if r.request.completion_time:
                events.append((r.request.completion_time, -1, "end"))
        
        # Sort events by time
        events.sort(key=lambda x: x[0])
        
        # Calculate running concurrency
        concurrency_data = []
        current_concurrency = 0
        
        for timestamp, delta, event_type in events:
            current_concurrency += delta
            concurrency_data.append({
                "timestamp": timestamp,
                "concurrency": current_concurrency,
                "event_type": event_type
            })
        
        # Add server metrics if available
        if self.server_metrics_history:
            for metric in self.server_metrics_history:
                concurrency_data.append({
                    "timestamp": metric["timestamp"],
                    "server_running": metric["num_running_reqs"],
                    "server_queued": metric["num_queue_reqs"],
                    "event_type": "server_poll"
                })
        
        df = pd.DataFrame(concurrency_data).sort_values("timestamp")
        return df
    
    def get_aggregated_metrics(self, 
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> AggregatedMetrics:
        """Get aggregated metrics for a time period."""
        metrics = AggregatedMetrics()
        
        # Filter results by time period
        filtered_results = self.results
        if start_time or end_time:
            filtered_results = []
            for result in self.results:
                completion_time = result.request.completion_time or 0
                if start_time and completion_time < start_time:
                    continue
                if end_time and completion_time > end_time:
                    continue
                filtered_results.append(result)
        
        if not filtered_results:
            return metrics
            
        # Calculate time bounds
        completion_times = [r.request.completion_time for r in filtered_results if r.request.completion_time]
        if completion_times:
            metrics.start_time = min(completion_times)
            metrics.end_time = max(completion_times)
            metrics.duration = metrics.end_time - metrics.start_time
        
        # Count requests
        metrics.total_requests = len(filtered_results)
        metrics.completed_requests = sum(1 for r in filtered_results if r.success)
        metrics.failed_requests = metrics.total_requests - metrics.completed_requests
        
        # Extract successful results
        successful_results = [r for r in filtered_results if r.success]
        
        if not successful_results:
            return metrics
            
        # Calculate throughput
        if metrics.duration > 0:
            metrics.request_throughput = metrics.completed_requests / metrics.duration
            
            total_input_tokens = sum(r.request.prompt_len for r in successful_results)
            total_output_tokens = sum(r.request.output_len for r in successful_results)
            
            metrics.input_token_throughput = total_input_tokens / metrics.duration
            metrics.output_token_throughput = total_output_tokens / metrics.duration
            metrics.total_token_throughput = (total_input_tokens + total_output_tokens) / metrics.duration
        
        # Calculate latencies
        server_latencies = [r.server_latency * 1000 for r in successful_results if r.server_latency > 0]
        total_latencies = [r.total_latency * 1000 for r in successful_results if r.total_latency > 0]
        queue_times = [r.queue_time * 1000 for r in successful_results if r.queue_time >= 0]
        ttfts = [r.ttft * 1000 for r in successful_results if r.ttft > 0]
        
        # Flatten ITLs
        all_itls = []
        for r in successful_results:
            all_itls.extend([itl * 1000 for itl in r.itl])
        
        # Server latency stats
        if server_latencies:
            metrics.mean_server_latency = np.mean(server_latencies)
            metrics.median_server_latency = np.median(server_latencies)
            metrics.p95_server_latency = np.percentile(server_latencies, 95)
            metrics.p99_server_latency = np.percentile(server_latencies, 99)
            metrics.max_server_latency = np.max(server_latencies)
        
        # Total latency stats
        if total_latencies:
            metrics.mean_total_latency = np.mean(total_latencies)
            metrics.median_total_latency = np.median(total_latencies)
            metrics.p95_total_latency = np.percentile(total_latencies, 95)
            metrics.p99_total_latency = np.percentile(total_latencies, 99)
            metrics.max_total_latency = np.max(total_latencies)
        
        # Queue time stats
        if queue_times:
            metrics.mean_queue_time = np.mean(queue_times)
            metrics.median_queue_time = np.median(queue_times)
            metrics.p95_queue_time = np.percentile(queue_times, 95)
            metrics.p99_queue_time = np.percentile(queue_times, 99)
            metrics.max_queue_time = np.max(queue_times)
        
        # TTFT stats
        if ttfts:
            metrics.mean_ttft = np.mean(ttfts)
            metrics.median_ttft = np.median(ttfts)
            metrics.p95_ttft = np.percentile(ttfts, 95)
            metrics.p99_ttft = np.percentile(ttfts, 99)
        
        # ITL stats
        if all_itls:
            metrics.mean_itl = np.mean(all_itls)
            metrics.median_itl = np.median(all_itls)
            metrics.p95_itl = np.percentile(all_itls, 95)
            metrics.p99_itl = np.percentile(all_itls, 99)
            metrics.max_itl = np.max(all_itls)
        
        # Queue depth stats
        if self.queue_depths:
            queue_values = [depth for _, depth in self.queue_depths]
            metrics.mean_queue_depth = np.mean(queue_values)
            metrics.max_queue_depth = max(queue_values)
        
        # GPU metrics from snapshots
        if self.snapshots:
            gpu_utils = [s.metrics.get("gpu_utilization", 0) for s in self.snapshots]
            gpu_mems = [s.metrics.get("gpu_memory_used", 0) for s in self.snapshots]
            
            if gpu_utils:
                metrics.gpu_utilization = np.mean(gpu_utils)
            if gpu_mems:
                metrics.gpu_memory_used = np.mean(gpu_mems)
                
            # Get total memory from last snapshot
            if "gpu_memory_total" in self.snapshots[-1].metrics:
                metrics.gpu_memory_total = self.snapshots[-1].metrics["gpu_memory_total"]
        
        return metrics
    
    def export_metrics(self, format: str = "json", path: str = None) -> str:
        """Export metrics to file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"metrics_{timestamp}.{format}"
        
        if format == "json":
            self._export_json(path)
        elif format == "csv":
            self._export_csv(path)
        elif format == "parquet":
            self._export_parquet(path)
        else:
            raise ValueError(f"Unknown export format: {format}")
            
        logger.info(f"Exported metrics to {path}")
        return path
    
    def _export_json(self, path: str):
        """Export metrics as JSON."""
        data = {
            "summary": self.get_aggregated_metrics().__dict__,
            "results": [
                {
                    "request_id": r.request.request_id,
                    "success": r.success,
                    "prompt_len": r.request.prompt_len,
                    "output_len": r.request.output_len,
                    "server_latency_ms": r.server_latency_ms,
                    "total_latency_ms": r.total_latency_ms,
                    "queue_time_ms": r.queue_time_ms,
                    "ttft_ms": r.ttft * 1000,
                    "arrival_time": r.request.arrival_time,
                    "send_time": r.request.send_time,
                    "completion_time": r.request.completion_time,
                    "error": r.error
                }
                for r in self.results
            ],
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    **s.metrics
                }
                for s in self.snapshots
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, path: str):
        """Export results as CSV with required format.
        
        Format: req_id, input_length, decode_length, arrival_time, to_server_time, 
                finish_time, server_latency, total_latency, ttft
        """
        # Find the earliest timestamp to calculate relative times
        if self.results:
            min_time = min(r.request.arrival_time for r in self.results if r.request.arrival_time is not None)
        else:
            min_time = 0
        
        records = []
        for r in self.results:
            # Convert times to relative values from test start
            arrival_relative = (r.request.arrival_time - min_time) if r.request.arrival_time else None
            send_relative = (r.request.send_time - min_time) if r.request.send_time else None
            finish_relative = (r.request.completion_time - min_time) if r.request.completion_time else None
            
            record = {
                "req_id": r.request.request_id,
                "input_length": r.request.prompt_len,
                "decode_length": r.request.output_len,
                "arrival_time": arrival_relative,
                "to_server_time": send_relative,
                "finish_time": finish_relative,
                "server_latency": r.server_latency,  # Keep in seconds
                "total_latency": r.total_latency,    # Keep in seconds
                "ttft": r.ttft,                      # Keep in seconds
                # Additional fields for analysis
                "queue_time": r.queue_time,
                "success": r.success,
                "error": r.error if not r.success else ""
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        # Ensure column order matches requirements
        columns = ["req_id", "input_length", "decode_length", "arrival_time", 
                  "to_server_time", "finish_time", "server_latency", "total_latency", "ttft",
                  "queue_time", "success", "error"]
        df = df[columns]
        df.to_csv(path, index=False)
    
    def _export_parquet(self, path: str):
        """Export results as Parquet."""
        records = []
        for r in self.results:
            record = {
                "request_id": r.request.request_id,
                "success": r.success,
                "prompt_len": r.request.prompt_len,
                "output_len": r.request.output_len,
                "server_latency_ms": r.server_latency * 1000,
                "total_latency_ms": r.total_latency * 1000,
                "queue_time_ms": r.queue_time * 1000,
                "ttft_ms": r.ttft * 1000,
                "itl_ms": r.itl,  # Parquet supports arrays
                "arrival_time": r.request.arrival_time,
                "send_time": r.request.send_time,
                "completion_time": r.request.completion_time,
                "error": r.error
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
    
    def print_summary(self):
        """Print a summary of collected metrics."""
        metrics = self.get_aggregated_metrics()
        
        logger.info("Generating metrics summary...")
        
        print("\n" + "=" * 60)
        print(" " * 20 + "METRICS SUMMARY")
        print("=" * 60)
        
        print(f"\n=� Request Statistics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful: {metrics.completed_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Success Rate: {metrics.completed_requests / metrics.total_requests * 100:.1f}%")
        
        print(f"\n� Throughput Metrics:")
        print(f"  Request Throughput: {metrics.request_throughput:.2f} req/s")
        print(f"  Input Token Throughput: {metrics.input_token_throughput:.0f} tok/s")
        print(f"  Output Token Throughput: {metrics.output_token_throughput:.0f} tok/s")
        print(f"  Total Token Throughput: {metrics.total_token_throughput:.0f} tok/s")
        
        print(f"\n�  Latency Metrics:")
        print(f"  Server Latency (ms):")
        print(f"    Mean: {metrics.mean_server_latency:.1f}")
        print(f"    Median: {metrics.median_server_latency:.1f}")
        print(f"    P95: {metrics.p95_server_latency:.1f}")
        print(f"    P99: {metrics.p99_server_latency:.1f}")
        print(f"    Max: {metrics.max_server_latency:.1f}")
        
        print(f"\n  Total Latency (ms):")
        print(f"    Mean: {metrics.mean_total_latency:.1f}")
        print(f"    Median: {metrics.median_total_latency:.1f}")
        print(f"    P95: {metrics.p95_total_latency:.1f}")
        print(f"    P99: {metrics.p99_total_latency:.1f}")
        print(f"    Max: {metrics.max_total_latency:.1f}")
        
        print(f"\n  Queue Time (ms):")
        print(f"    Mean: {metrics.mean_queue_time:.1f}")
        print(f"    Median: {metrics.median_queue_time:.1f}")
        print(f"    P95: {metrics.p95_queue_time:.1f}")
        print(f"    P99: {metrics.p99_queue_time:.1f}")
        
        print(f"\n  Token Generation:")
        print(f"    Mean TTFT: {metrics.mean_ttft:.1f} ms")
        print(f"    Mean ITL: {metrics.mean_itl:.1f} ms")
        
        print(f"\n=� Queue Metrics:")
        print(f"  Mean Queue Depth: {metrics.mean_queue_depth:.1f}")
        print(f"  Max Queue Depth: {metrics.max_queue_depth}")
        
        if metrics.gpu_utilization > 0:
            print(f"\n=�  GPU Metrics:")
            print(f"  GPU Utilization: {metrics.gpu_utilization:.1f}%")
            print(f"  GPU Memory: {metrics.gpu_memory_used:.1f} / {metrics.gpu_memory_total:.1f} GB")
        
        print("=" * 60 + "\n")
    
    def enable_incremental_save(self, base_path: str, interval: int = 100):
        """Enable incremental saving of results.
        
        Args:
            base_path: Base path for saving incremental results (without extension)
            interval: Save every N requests (default: 100)
        """
        self.incremental_save_path = base_path
        self.incremental_save_interval = interval
        logger.info(f"Enabled incremental saving to {base_path} every {interval} requests")
        
    def _save_incremental(self):
        """Save current results incrementally."""
        if not self.incremental_save_path:
            return
            
        try:
            # Save CSV with current results
            csv_path = f"{self.incremental_save_path}_incremental.csv"
            self._export_csv(csv_path)
            
            # Also save a lightweight JSON summary
            summary_path = f"{self.incremental_save_path}_incremental_summary.json"
            summary_data = {
                "timestamp": time.time(),
                "total_requests": len(self.results),
                "successful_requests": sum(1 for r in self.results if r.success),
                "failed_requests": sum(1 for r in self.results if not r.success),
                "aggregated_metrics": self.get_aggregated_metrics().__dict__
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            logger.info(f"Incremental save completed: {len(self.results)} results saved")
        except Exception as e:
            logger.error(f"Failed to save incremental results: {e}")
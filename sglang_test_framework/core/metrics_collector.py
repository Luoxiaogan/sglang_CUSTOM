"""Metrics collection and analysis for SGLang testing."""

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
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.collection_interval = self.config.get("collection_interval", 1.0)
        
        # Storage for results
        self.results: List[RequestResult] = []
        self.snapshots: List[MetricSnapshot] = []
        
        # Real-time metrics
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.queue_depths: Deque[Tuple[float, int]] = deque(maxlen=10000)
        
        # Collection control
        self.collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.start_time: Optional[float] = None
        
    def start_collection(self):
        """Start metrics collection."""
        if self.collecting:
            logger.warning("Metrics collection already started")
            return
            
        self.collecting = True
        self.start_time = time.time()
        self.collection_task = asyncio.create_task(self._collection_loop())
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
    
    async def _collect_snapshot(self) -> MetricSnapshot:
        """Collect a snapshot of current metrics."""
        metrics = {
            "timestamp": time.time(),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.results),
        }
        
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
                    "server_latency_ms": r.server_latency * 1000,
                    "total_latency_ms": r.total_latency * 1000,
                    "queue_time_ms": r.queue_time * 1000,
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
        """Export results as CSV."""
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
                "mean_itl_ms": np.mean(r.itl) * 1000 if r.itl else 0,
                "arrival_time": r.request.arrival_time,
                "send_time": r.request.send_time,
                "completion_time": r.request.completion_time,
                "error": r.error
            }
            records.append(record)
            
        df = pd.DataFrame(records)
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
        
        print("\n" + "=" * 60)
        print(" " * 20 + "METRICS SUMMARY")
        print("=" * 60)
        
        print(f"\n=Ê Request Statistics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful: {metrics.completed_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Success Rate: {metrics.completed_requests / metrics.total_requests * 100:.1f}%")
        
        print(f"\n¡ Throughput Metrics:")
        print(f"  Request Throughput: {metrics.request_throughput:.2f} req/s")
        print(f"  Input Token Throughput: {metrics.input_token_throughput:.0f} tok/s")
        print(f"  Output Token Throughput: {metrics.output_token_throughput:.0f} tok/s")
        print(f"  Total Token Throughput: {metrics.total_token_throughput:.0f} tok/s")
        
        print(f"\nñ  Latency Metrics:")
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
        
        print(f"\n=È Queue Metrics:")
        print(f"  Mean Queue Depth: {metrics.mean_queue_depth:.1f}")
        print(f"  Max Queue Depth: {metrics.max_queue_depth}")
        
        if metrics.gpu_utilization > 0:
            print(f"\n=¥  GPU Metrics:")
            print(f"  GPU Utilization: {metrics.gpu_utilization:.1f}%")
            print(f"  GPU Memory: {metrics.gpu_memory_used:.1f} / {metrics.gpu_memory_total:.1f} GB")
        
        print("=" * 60 + "\n")
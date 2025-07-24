"""Base routing policy for SGLang testing framework.

Adapted from SGLang Router patterns.
Source references:
- python/sglang_router/
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    url: str
    active_requests: int = 0
    queue_depth: int = 0
    last_update: float = 0.0
    is_healthy: bool = True
    total_requests: int = 0
    error_count: int = 0
    

@dataclass
class ServerMetrics:
    """Metrics from a server."""
    worker_id: str
    throughput: float
    latency: float
    queue_depth: int
    memory_usage: float
    num_running_reqs: int
    num_queue_reqs: int
    timestamp: float


class BaseRoutingPolicy(ABC):
    """Abstract base class for routing policies."""
    
    def __init__(self, workers: List[Dict[str, Any]]):
        """Initialize routing policy.
        
        Args:
            workers: List of worker configurations with 'id' and 'url'
        """
        self.workers = {}
        for w in workers:
            worker_id = w.get("id", f"worker_{len(self.workers)}")
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                url=w["url"],
                last_update=time.time()
            )
    
    @abstractmethod
    def route_request(self, request: Any, server_metrics: Dict[str, ServerMetrics]) -> str:
        """Route a request to a worker.
        
        Args:
            request: The incoming request
            server_metrics: Current metrics for all servers
            
        Returns:
            worker_id: ID of the selected worker
        """
        pass
    
    def update_worker_state(self, worker_id: str, metrics: ServerMetrics):
        """Update worker state based on metrics.
        
        Args:
            worker_id: Worker to update
            metrics: Latest metrics from the worker
        """
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.active_requests = metrics.num_running_reqs
            worker.queue_depth = metrics.num_queue_reqs
            worker.last_update = time.time()
    
    def mark_worker_unhealthy(self, worker_id: str):
        """Mark a worker as unhealthy."""
        if worker_id in self.workers:
            self.workers[worker_id].is_healthy = False
            self.workers[worker_id].error_count += 1
    
    def mark_worker_healthy(self, worker_id: str):
        """Mark a worker as healthy."""
        if worker_id in self.workers:
            self.workers[worker_id].is_healthy = True
    
    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs."""
        return [w_id for w_id, w in self.workers.items() if w.is_healthy]
    
    def record_request_routed(self, worker_id: str):
        """Record that a request was routed to a worker."""
        if worker_id in self.workers:
            self.workers[worker_id].total_requests += 1
    
    def get_worker_url(self, worker_id: str) -> Optional[str]:
        """Get URL for a worker."""
        worker = self.workers.get(worker_id)
        return worker.url if worker else None
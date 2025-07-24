"""Round-robin routing policy implementation."""

from typing import Dict, Any

from .base import BaseRoutingPolicy, ServerMetrics


class RoundRobinRouting(BaseRoutingPolicy):
    """Route requests in round-robin fashion."""
    
    def __init__(self, workers: list):
        super().__init__(workers)
        self.current_index = 0
    
    def route_request(self, request: Any, server_metrics: Dict[str, ServerMetrics]) -> str:
        """Route request to next worker in round-robin order.
        
        Args:
            request: The incoming request
            server_metrics: Current metrics for all servers (ignored)
            
        Returns:
            worker_id: ID of the selected worker
        """
        healthy_workers = self.get_healthy_workers()
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")
        
        # Find next healthy worker
        attempts = 0
        while attempts < len(self.workers):
            worker_ids = list(self.workers.keys())
            worker_id = worker_ids[self.current_index % len(worker_ids)]
            self.current_index = (self.current_index + 1) % len(worker_ids)
            
            if worker_id in healthy_workers:
                self.record_request_routed(worker_id)
                return worker_id
            
            attempts += 1
        
        raise RuntimeError("No healthy workers found in round-robin")
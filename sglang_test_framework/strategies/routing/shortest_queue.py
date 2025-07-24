"""Shortest queue routing policy implementation."""

from typing import Dict, Any

from .base import BaseRoutingPolicy, ServerMetrics


class ShortestQueueRouting(BaseRoutingPolicy):
    """Route to the worker with shortest queue."""
    
    def route_request(self, request: Any, server_metrics: Dict[str, ServerMetrics]) -> str:
        """Route request to worker with shortest queue.
        
        Args:
            request: The incoming request
            server_metrics: Current metrics for all servers
            
        Returns:
            worker_id: ID of the selected worker
        """
        healthy_workers = self.get_healthy_workers()
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")
        
        # Update worker states from metrics
        for worker_id, metrics in server_metrics.items():
            if worker_id in self.workers:
                self.update_worker_state(worker_id, metrics)
        
        # Find worker with minimum queue depth
        min_queue_depth = float('inf')
        selected_worker = None
        
        for worker_id in healthy_workers:
            worker = self.workers[worker_id]
            total_load = worker.active_requests + worker.queue_depth
            
            if total_load < min_queue_depth:
                min_queue_depth = total_load
                selected_worker = worker_id
        
        if selected_worker is None:
            # Fallback to first healthy worker
            selected_worker = healthy_workers[0]
        
        self.record_request_routed(selected_worker)
        return selected_worker
"""Random routing policy implementation."""

import random
from typing import Dict, Any

from .base import BaseRoutingPolicy, ServerMetrics


class RandomRouting(BaseRoutingPolicy):
    """Randomly select a healthy worker."""
    
    def route_request(self, request: Any, server_metrics: Dict[str, ServerMetrics]) -> str:
        """Route request to a random healthy worker.
        
        Args:
            request: The incoming request
            server_metrics: Current metrics for all servers (ignored)
            
        Returns:
            worker_id: ID of the randomly selected worker
        """
        healthy_workers = self.get_healthy_workers()
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")
        
        selected = random.choice(healthy_workers)
        self.record_request_routed(selected)
        return selected
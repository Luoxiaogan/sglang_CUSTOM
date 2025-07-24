"""Configuration for multi-node routing testing.

Adapted from SGLang Router's configuration patterns.
Source references:
- python/sglang_router/
- .DUCUMENT/SGLang_Router_详解.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import BaseConfig, ServerConfig, MetricsConfig


@dataclass
class RouterConfig:
    """Configuration for SGLang router.
    
    Based on SGLang Router arguments from:
    - python/sglang_router/launch_router.py  
    - .DUCUMENT/SGLang_Router_详解.md
    """
    
    # Router identification
    port: int = 30000
    host: str = "0.0.0.0"
    
    # Router mode
    router_mode: str = "separate"  # Options: "joint", "separate"
    
    # Data parallelism (for joint mode)
    dp_size: Optional[int] = None  # Number of data parallel replicas
    
    # Model configuration (for joint mode)
    model_path: Optional[str] = None
    
    # Routing policy
    policy: str = "cache_aware"  # Options: "cache_aware", "round_robin", "random", "shortest_queue"
    
    # Cache-aware routing parameters
    cache_threshold: float = 0.5
    balance_abs_threshold: int = 32
    balance_rel_threshold: float = 1.0001
    eviction_interval: int = 60  # seconds
    max_tree_size: int = 16777216  # 16MB default
    
    # Request handling
    max_outstanding_requests: Optional[int] = None
    
    # Custom policy configuration
    custom_policy_class: Optional[str] = None
    custom_policy_config: Optional[Dict[str, Any]] = None
    
    def get_launch_args(self) -> List[str]:
        """Get command line arguments for launching the router.
        
        Note: In joint mode, the model path and dp-size are required.
        In separate mode, worker URLs are passed separately.
        """
        args = [
            "--port", str(self.port),
            "--host", self.host,
            "--policy", self.policy,
        ]
        
        # Add model and dp-size for joint mode
        if self.router_mode == "joint":
            if not self.model_path or not self.dp_size:
                raise ValueError("model_path and dp_size are required for joint mode")
            args.extend(["--model-path", self.model_path])
            args.extend(["--dp-size", str(self.dp_size)])
        
        # Cache-aware routing parameters
        if self.policy == "cache_aware":
            args.extend([
                "--cache-threshold", str(self.cache_threshold),
                "--balance-abs-threshold", str(self.balance_abs_threshold),
                "--balance-rel-threshold", str(self.balance_rel_threshold),
                "--eviction-interval", str(self.eviction_interval),
                "--max-tree-size", str(self.max_tree_size),
            ])
        
        if self.max_outstanding_requests:
            args.extend(["--max-outstanding-requests", str(self.max_outstanding_requests)])
            
        if self.custom_policy_class:
            args.extend(["--custom-policy-class", self.custom_policy_class])
            
        return args


@dataclass
class RoutingConfig(BaseConfig):
    """Configuration for routing-level testing with multiple GPUs."""
    
    # Multi-GPU configuration
    num_gpus: int = 4
    gpu_ids: Optional[List[int]] = None  # If None, use [0, 1, ..., num_gpus-1]
    
    # Server configuration
    base_port: int = 30001  # Starting port for worker servers
    server_host: str = "0.0.0.0"
    
    # Router configuration
    router_config: Optional[RouterConfig] = None
    
    # Routing policy
    routing_policy: str = "cache_aware"  # Convenience field, overrides router_config.policy
    
    # Per-server configuration
    servers_config: Optional[List[ServerConfig]] = None
    
    # Uniform server parameters (used if servers_config is None)
    max_running_requests: int = 256
    mem_fraction_static: float = 0.9
    chunked_prefill_size: int = 8192
    enable_torch_compile: bool = False
    quantization: Optional[str] = None
    
    # Dynamic parameter updates
    enable_dynamic_params: bool = False
    param_update_schedule: Optional[Dict[int, Dict[str, Any]]] = None
    per_node_update_schedule: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None
    
    # Load balancing test configuration
    load_distribution: str = "uniform"  # Options: "uniform", "skewed", "pattern"
    skew_factor: float = 1.0  # For skewed distribution
    
    # Metrics configuration
    metrics_config: Optional[MetricsConfig] = None
    collect_per_node_metrics: bool = True
    
    # Test scenarios
    enable_node_failures: bool = False
    failure_schedule: Optional[Dict[int, List[int]]] = None  # time -> [node_ids]
    
    def __post_init__(self):
        """Initialize configurations."""
        super().__post_init__()
        
        # Set GPU IDs if not provided
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.num_gpus))
        elif len(self.gpu_ids) != self.num_gpus:
            raise ValueError(f"Length of gpu_ids ({len(self.gpu_ids)}) must match num_gpus ({self.num_gpus})")
            
        # Create default router config if not provided
        if self.router_config is None:
            self.router_config = RouterConfig(policy=self.routing_policy)
        else:
            # Override policy if specified
            self.router_config.policy = self.routing_policy
            
        # Create default metrics config if not provided
        if self.metrics_config is None:
            self.metrics_config = MetricsConfig()
            
        # Create server configurations if not provided
        if self.servers_config is None:
            self.servers_config = self._create_uniform_server_configs()
    
    def validate(self):
        """Validate routing configuration."""
        super().validate()
        
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
            
        if self.base_port < 1024 or self.base_port + self.num_gpus > 65535:
            raise ValueError("Invalid port range")
            
        valid_policies = ["cache_aware", "round_robin", "random", "shortest_queue", 
                         "uniform", "shortest_job", "param_aware", "custom"]
        if self.routing_policy not in valid_policies:
            raise ValueError(f"Unknown routing policy: {self.routing_policy}")
            
        if self.enable_node_failures and not self.failure_schedule:
            raise ValueError("failure_schedule required when enable_node_failures is True")
            
        if self.load_distribution not in ["uniform", "skewed", "pattern"]:
            raise ValueError(f"Unknown load distribution: {self.load_distribution}")
    
    def _create_uniform_server_configs(self) -> List[ServerConfig]:
        """Create uniform server configurations for all GPUs."""
        configs = []
        for i, gpu_id in enumerate(self.gpu_ids):
            config = ServerConfig(
                server_id=f"worker_{i}",
                gpu_id=gpu_id,
                port=self.base_port + i,
                host=self.server_host,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                mem_fraction_static=self.mem_fraction_static,
                max_running_requests=self.max_running_requests,
                chunked_prefill_size=self.chunked_prefill_size,
                enable_torch_compile=self.enable_torch_compile,
                quantization=self.quantization,
                enable_metrics=True
            )
            configs.append(config)
        return configs
    
    def get_worker_urls(self) -> List[str]:
        """Get URLs for all worker servers."""
        urls = []
        for config in self.servers_config:
            url = f"http://{config.host}:{config.port}"
            urls.append(url)
        return urls
    
    @classmethod
    def create_default(cls, model_path: str, num_gpus: int = 4) -> "RoutingConfig":
        """Create a default configuration for quick testing."""
        return cls(
            model_path=model_path,
            num_gpus=num_gpus,
            num_prompts=5000,
            request_rate=40.0,
            routing_policy="cache_aware",
            max_running_requests=256
        )
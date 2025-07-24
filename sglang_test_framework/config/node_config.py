"""Configuration for single node (single GPU) testing.

Adapted from SGLang's configuration patterns.
Note: SGLang handles batching automatically, no batch strategy configuration needed.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .base import BaseConfig, ServerConfig, MetricsConfig


@dataclass
class NodeConfig(BaseConfig):
    """Configuration for node-level testing on a single GPU."""
    
    # GPU configuration
    gpu_id: int = 0
    
    # Server configuration
    port: int = 30000
    host: str = "0.0.0.0"
    
    # Memory configuration
    mem_fraction_static: float = 0.9
    
    # Request handling
    max_running_requests: int = 256
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    
    # Parallelism
    tp_size: int = 1  # Tensor parallelism size
    
    # Performance tuning
    enable_torch_compile: bool = False
    disable_radix_cache: bool = False
    schedule_conservativeness: float = 1.0
    
    # Quantization
    quantization: Optional[str] = None
    
    # Attention backend
    attention_backend: Optional[str] = None
    
    # Metrics configuration
    metrics_config: Optional[MetricsConfig] = None
    
    # Dynamic parameter update
    enable_dynamic_params: bool = False
    param_update_schedule: Optional[Dict[int, Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Initialize server and metrics configurations."""
        super().__post_init__()
        
        # Create default metrics config if not provided
        if self.metrics_config is None:
            self.metrics_config = MetricsConfig()
    
    def validate(self):
        """Validate node configuration."""
        super().validate()
        
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
            
        if self.port < 1024 or self.port > 65535:
            raise ValueError("port must be between 1024 and 65535")
            
        if not 0 < self.mem_fraction_static <= 1:
            raise ValueError("mem_fraction_static must be between 0 and 1")
            
        if self.max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
            
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
            
        if self.enable_dynamic_params and not self.param_update_schedule:
            raise ValueError("param_update_schedule required when enable_dynamic_params is True")
    
    def get_server_config(self) -> ServerConfig:
        """Create ServerConfig from NodeConfig."""
        return ServerConfig(
            server_id=f"node_{self.gpu_id}",
            gpu_id=self.gpu_id,
            port=self.port,
            host=self.host,
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            mem_fraction_static=self.mem_fraction_static,
            max_running_requests=self.max_running_requests,
            max_total_tokens=self.max_total_tokens,
            chunked_prefill_size=self.chunked_prefill_size,
            max_prefill_tokens=self.max_prefill_tokens,
            tp_size=self.tp_size,
            enable_torch_compile=self.enable_torch_compile,
            disable_radix_cache=self.disable_radix_cache,
            schedule_conservativeness=self.schedule_conservativeness,
            quantization=self.quantization,
            attention_backend=self.attention_backend,
            enable_metrics=True,
            verbose=self.verbose
        )
    
    @classmethod
    def create_default(cls, model_path: str, gpu_id: int = 0) -> "NodeConfig":
        """Create a default configuration for quick testing."""
        return cls(
            model_path=model_path,
            gpu_id=gpu_id,
            num_prompts=1000,
            request_rate=10.0,
            max_running_requests=256
        )
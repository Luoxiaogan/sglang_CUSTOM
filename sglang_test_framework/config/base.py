"""Base configuration classes for SGLang testing framework.

Adapted from SGLang's configuration patterns.
Source references:
- python/sglang/srt/server_args.py
- .DUCUMENT/Server_Arguments.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


@dataclass
class BaseConfig(ABC):
    """Base configuration class for all test configurations."""
    
    # Model configuration
    model_path: str
    tokenizer_path: Optional[str] = None
    
    # Test configuration
    num_prompts: int = 1000
    request_rate: float = float('inf')  # Requests per second
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    
    # Output configuration
    output_dir: str = "./results"
    output_format: str = "json"
    save_detailed_results: bool = True
    
    # Random dataset configuration
    random_input_len: int = 1024
    random_output_len: int = 128
    random_range_ratio: float = 0.5
    
    # Runtime configuration
    seed: int = 42
    warmup_requests: int = 10
    timeout: int = 3600  # seconds
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
        
    @abstractmethod
    def validate(self):
        """Validate the configuration parameters."""
        if self.num_prompts <= 0:
            raise ValueError("num_prompts must be positive")
        
        if self.request_rate <= 0:
            raise ValueError("request_rate must be positive")
            
        if self.dataset_name not in ["sharegpt", "random", "random-ids", "custom"]:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ServerConfig:
    """Configuration for a single SGLang server.
    
    Based on SGLang server arguments from:
    - python/sglang/srt/server_args.py
    - .DUCUMENT/Server_Arguments.md
    """
    
    # Server identification
    server_id: str
    gpu_id: int  # Used for CUDA_VISIBLE_DEVICES env var
    port: int = 30000
    host: str = "0.0.0.0"
    
    # Model configuration
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    
    # Model loading
    dtype: str = "auto"  # Options: "auto", "float16", "bfloat16", "float32"
    load_format: str = "auto"  # Options: "auto", "pt", "safetensors", etc.
    
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
    disable_regex_jump_forward: bool = False
    disable_flashinfer: bool = False
    schedule_conservativeness: float = 1.0
    
    # Quantization
    quantization: Optional[str] = None  # Options: "fp8", "int8", "awq", "gptq", etc.
    
    # Attention backend
    attention_backend: Optional[str] = None  # Auto-selected if None
    
    # Logging and monitoring
    log_level: str = "info"
    enable_metrics: bool = True
    
    # OpenAI API compatibility
    api_key: Optional[str] = None
    
    def get_launch_args(self) -> List[str]:
        """Get command line arguments for launching the server.
        
        Note: CUDA_VISIBLE_DEVICES should be set as an environment variable,
        not as a command line argument. The server_manager handles this.
        """
        args = [
            "--model-path", self.model_path,
            "--port", str(self.port),
            "--host", self.host,
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--max-running-requests", str(self.max_running_requests),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--max-prefill-tokens", str(self.max_prefill_tokens),
            "--schedule-conservativeness", str(self.schedule_conservativeness),
            "--tp-size", str(self.tp_size),
            "--tokenizer-mode", self.tokenizer_mode,
            "--dtype", self.dtype,
            "--load-format", self.load_format,
            "--log-level", self.log_level,
        ]
        
        if self.tokenizer_path:
            args.extend(["--tokenizer-path", self.tokenizer_path])
            
        if self.max_total_tokens:
            args.extend(["--max-total-tokens", str(self.max_total_tokens)])
            
        if self.trust_remote_code:
            args.append("--trust-remote-code")
            
        if self.enable_torch_compile:
            args.append("--enable-torch-compile")
            
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
            
        if self.disable_regex_jump_forward:
            args.append("--disable-regex-jump-forward")
            
        if self.disable_flashinfer:
            args.append("--disable-flashinfer")
            
        if self.quantization:
            args.extend(["--quantization", self.quantization])
            
        if self.attention_backend:
            args.extend(["--attention-backend", self.attention_backend])
            
        if self.enable_metrics:
            args.append("--enable-metrics")
            
        if self.api_key:
            args.extend(["--api-key", self.api_key])
            
        return args


@dataclass 
class MetricsConfig:
    """Configuration for metrics collection."""
    
    # Collection intervals
    collection_interval: float = 1.0  # seconds
    
    # Metrics to collect
    collect_throughput: bool = True
    collect_latency: bool = True
    collect_queue_metrics: bool = True
    collect_resource_usage: bool = True
    collect_cache_metrics: bool = True
    
    # Latency percentiles to calculate
    latency_percentiles: List[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.95, 0.99]
    )
    
    # Export settings
    export_format: str = "json"  # Options: "json", "csv", "parquet"
    export_interval: Optional[int] = None  # Export every N seconds
    
    # Real-time monitoring
    enable_dashboard: bool = False
    dashboard_port: int = 8080
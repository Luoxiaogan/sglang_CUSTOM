"""SGLang Testing Framework - A comprehensive testing solution for SGLang."""

import os
import logging
from .utils.logging import setup_logging

# Initialize logging when the package is imported
log_level = os.environ.get('SGLANG_TEST_LOG_LEVEL', 'INFO')
setup_logging(log_level)

# Log framework initialization
logger = logging.getLogger(__name__)
logger.info(f"SGLang Testing Framework v1.0.0 initialized with log level: {log_level}")

from .config import NodeConfig, RoutingConfig
from .core import (
    ServerManager, RequestGenerator, MetricsCollector, ResultManager,
    Request, RequestResult, RequestSender
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "NodeConfig",
    "RoutingConfig",
    
    # Core components
    "ServerManager",
    "RequestGenerator", 
    "MetricsCollector",
    "ResultManager",
    
    # Data types
    "Request",
    "RequestResult",
    "RequestSender"
]
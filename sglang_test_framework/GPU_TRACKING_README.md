# GPU Tracking in SGLang Routing Tests

## Overview

The SGLang routing test framework tracks which GPU processes each request to provide per-GPU performance metrics. This document explains how GPU tracking works and how to troubleshoot issues.

## How GPU Tracking Works

### 1. Primary Method: Worker URL Headers
The sgl-router adds an `X-SGLang-Worker` header to each response containing the worker URL that processed the request. The test framework maps these URLs to GPU IDs.

**Status**: Implemented in router code but requires router rebuild and deployment.

### 2. Port-Based Mapping
Worker URLs are mapped to GPU IDs based on port numbers:
- Port 30001 → GPU 0
- Port 30002 → GPU 1
- etc.

The framework creates multiple URL format mappings to handle different formats the router might return.

### 3. Fallback Policy-Based Tracking
When worker URLs are not available, the framework uses routing policy logic:

- **round_robin**: Assigns GPU based on request counter modulo number of GPUs
- **random**: Uses hash of request ID to deterministically assign GPU
- **cache_aware/shortest_queue**: Requires router information (no fallback available)

## Troubleshooting

### Empty gpu_id in CSV/JSON

1. **Check router logs**: Look for "Response headers for request" messages
2. **Verify worker URL mapping**: Check logs for "Mapped port X to GPU Y" messages
3. **Routing policy**: Use round_robin or random for better fallback tracking

### Building Modified Router

To enable proper GPU tracking with headers:

```bash
cd sgl-router
cargo build --release
pip install -e .
```

### Current Limitations

1. The X-SGLang-Worker header requires a modified router build
2. Fallback tracking is approximate for cache_aware and shortest_queue policies
3. Remote deployments may not have the modified router

## Metrics Output

When GPU tracking works correctly, you'll see:

1. **CSV**: `gpu_id` column populated with GPU assignments
2. **JSON**: `gpu_metrics` section with per-GPU performance breakdowns
3. **Logs**: "GPU metrics available for GPUs: [0, 1]" message

## Example Usage

```python
# Use round_robin for predictable GPU tracking
config = RoutingConfig(
    routing_policy="round_robin",  # Best for fallback tracking
    num_gpus=2,
    # ... other config
)
```
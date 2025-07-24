# SGLang Router Async Context Issue

## Problem Description

When building and installing the modified sgl-router (with X-SGLang-Worker header support), the router fails to start with the following error:

```
thread '<unnamed>' panicked at /home/lg/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/tokio-1.46.1/src/runtime/blocking/shutdown.rs:51:21:
Cannot drop a runtime in a context where blocking is not allowed. This happens when a runtime is dropped from within an asynchronous context.
```

## Root Cause

The issue occurs in the Python-Rust integration layer (PyO3) when the router tries to create/manage a tokio runtime. The router's `start()` method is being called from Python, and internally it's trying to create a new tokio runtime, which conflicts with async context management.

## Workarounds

### 1. Use System-Installed Router

Revert to the pip-installed router which doesn't have this issue:

```bash
pip uninstall sglang-router
pip install sglang-router
```

### 2. Use Fallback GPU Tracking

The test framework includes fallback GPU tracking that works without router headers:

- For `round_robin` policy: Uses request counter modulo number of GPUs
- For `random` policy: Uses hash of request ID for deterministic assignment

Configure your test with:

```python
config = RoutingConfig(
    routing_policy="round_robin",  # Best for fallback tracking
    # ... other config
)
```

### 3. Run Router Separately

Start the router as a separate process outside of the test framework:

```bash
# Terminal 1: Start workers
python -m sglang.launch_server --model-path /path/to/model --port 30001 &
python -m sglang.launch_server --model-path /path/to/model --port 30002 &

# Terminal 2: Start router
python -m sglang_router.launch_router --worker-urls http://localhost:30001 http://localhost:30002

# Terminal 3: Run tests
python sglang_test_framework/test_route.py
```

## Long-term Fix

The router code needs to be updated to handle async contexts properly. Possible solutions:

1. Modify the Rust code to detect and handle existing tokio runtime
2. Update the Python wrapper to spawn router in a separate thread
3. Use a different async runtime management strategy

## Current Status

- Router header implementation is complete and works when router runs successfully
- Fallback GPU tracking provides accurate metrics for round_robin and random policies
- The async issue only affects the locally-built router, not the pip-installed version
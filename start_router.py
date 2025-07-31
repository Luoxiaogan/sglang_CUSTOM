"""
General router launcher with configurable policy types.

Usage:
    python start_a_router_general.py --policy cache_aware
    python start_a_router_general.py --policy round_robin --port 40010
    python start_a_router_general.py --policy random --workers http://localhost:30001 http://localhost:30002

Policy types:
    - cache_aware: Routes based on cache hits and load balancing (default)
    - round_robin: Distributes requests in circular order
    - random: Randomly selects workers
    - power_of_two: Selects best of two random workers (PD mode only, requires --pd-mode)
"""

import argparse
import json
import sys
from sglang_router_rs import Router, PolicyType


def main():
    parser = argparse.ArgumentParser(description="Start SGLang router with configurable policy")
    
    # Policy configuration
    parser.add_argument(
        "--policy",
        type=str,
        default="cache_aware",
        choices=["cache_aware", "round_robin", "random", "power_of_two", "marginal_utility"],
        help="Routing policy type (default: cache_aware)"
    )
    
    # Worker configuration
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["http://localhost:40005", "http://localhost:40006"],
        help="Worker URLs (default: http://localhost:40005 http://localhost:40006)"
    )
    
    # Router configuration
    parser.add_argument(
        "--port",
        type=int,
        default=40009,
        help="Router port (default: 40009)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Router host (default: 0.0.0.0)"
    )
    
    # Request tracking configuration
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=True,
        help="Enable request tracking (default: True)"
    )
    
    parser.add_argument(
        "--max-trace-entries",
        type=int,
        default=100000,
        help="Maximum number of trace entries to keep (default: 100000)"
    )
    
    parser.add_argument(
        "--trace-ttl",
        type=int,
        default=3600,
        help="Time to live for trace entries in seconds (default: 3600)"
    )
    
    # PD mode for power_of_two policy
    parser.add_argument(
        "--pd-mode",
        action="store_true",
        help="Enable PD mode (required for power_of_two policy)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        help="Log level (default: INFO)"
    )
    
    # Port-GPU mapping configuration
    parser.add_argument(
        "--port-gpu-mapping",
        type=str,
        help="JSON string or file path containing port to GPU mapping"
    )
    
    args = parser.parse_args()
    
    # Validate power_of_two requires PD mode
    if args.policy == "power_of_two" and not args.pd_mode:
        print("Error: power_of_two policy requires --pd-mode flag")
        sys.exit(1)
    
    # Map policy name to PolicyType
    policy_map = {
        "cache_aware": PolicyType.CacheAware,
        "round_robin": PolicyType.RoundRobin,
        "random": PolicyType.Random,
        "power_of_two": PolicyType.PowerOfTwo,
        "marginal_utility": PolicyType.MarginalUtility,
    }
    
    # Handle port-GPU mapping
    port_gpu_mapping = {}
    if args.port_gpu_mapping:
        try:
            # Try parsing as JSON string first
            port_gpu_mapping = json.loads(args.port_gpu_mapping)
        except json.JSONDecodeError:
            # Try loading from file
            try:
                with open(args.port_gpu_mapping, 'r') as f:
                    port_gpu_mapping = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load port-GPU mapping: {e}")
    else:
        # Default mapping based on common setup
        # Extract port numbers from worker URLs
        for i, worker in enumerate(args.workers):
            try:
                port = int(worker.split(":")[-1])
                # Map to GPU based on position (this is a simple heuristic)
                port_gpu_mapping[port] = f"cuda:{i}"
            except:
                pass
    
    # Save port-GPU mapping for other scripts
    if port_gpu_mapping:
        mapping_file = "/tmp/sglang_port_gpu_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(port_gpu_mapping, f)
        print(f"Port-GPU mapping saved to: {mapping_file}")
        print("Mapping:")
        for port, gpu in port_gpu_mapping.items():
            print(f"  Port {port} -> {gpu}")
        print()
    
    # Create router configuration
    router_config = {
        "worker_urls": args.workers,
        "policy": policy_map[args.policy],
        "port": args.port,
        "host": args.host,
        "log_level": args.log_level,
    }
    
    # Add request tracking if enabled
    if args.enable_tracking:
        router_config.update({
            "enable_request_tracking": True,
            "max_trace_entries": args.max_trace_entries,
            "trace_ttl_seconds": args.trace_ttl,
        })
    
    # Create and start router
    print(f"Starting router with configuration:")
    print(f"  Policy: {args.policy}")
    print(f"  Workers: {args.workers}")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Request tracking: {'Enabled' if args.enable_tracking else 'Disabled'}")
    if args.enable_tracking:
        print(f"  Max trace entries: {args.max_trace_entries}")
        print(f"  Trace TTL: {args.trace_ttl}s")
    print(f"  Log level: {args.log_level}")
    print()
    
    try:
        router = Router(**router_config)
        print("Router started successfully! Press Ctrl+C to stop.")
        router.start()
    except KeyboardInterrupt:
        print("\nRouter stopped.")
    except Exception as e:
        print(f"Error starting router: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
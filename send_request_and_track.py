"""
Send requests to router and track their execution.

Usage:
    python send_request_and_track.py --num-requests 100 --dataset random
    python send_request_and_track.py --dataset sharegpt --request-rate 10
    python send_request_and_track.py --dataset custom --dataset-path /path/to/data.json
"""

import argparse
import asyncio
import aiohttp
import json
import time
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import random
import string
from typing import List, Dict, Any


class RequestGenerator:
    """Generate requests for testing."""
    
    def __init__(self, dataset_name: str, dataset_path: str = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        
    def generate_requests(self, num_requests: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate test requests based on dataset type."""
        if self.dataset_name == "random":
            return self._generate_random_requests(num_requests, **kwargs)
        elif self.dataset_name == "sharegpt":
            return self._generate_sharegpt_requests(num_requests)
        elif self.dataset_name == "custom":
            return self._load_custom_requests(num_requests)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _generate_random_requests(self, num_requests: int, 
                                  input_len: int = 512, 
                                  output_len: int = 128,
                                  range_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """Generate random requests with configurable lengths."""
        requests = []
        
        for i in range(num_requests):
            # Add variability to lengths
            actual_input_len = int(np.random.uniform(
                input_len * (1 - range_ratio),
                input_len * (1 + range_ratio)
            ))
            actual_output_len = int(np.random.uniform(
                output_len * (1 - range_ratio),
                output_len * (1 + range_ratio)
            ))
            
            # Generate random prompt
            prompt = f"Request {i}: " + " ".join(
                ''.join(random.choices(string.ascii_letters, k=5))
                for _ in range(actual_input_len // 6)
            )
            
            requests.append({
                "request_id": f"req_{i}_{int(time.time()*1000)}",
                "prompt": prompt,
                "max_new_tokens": actual_output_len,
                "temperature": 0.7,
                "input_len": actual_input_len,
                "expected_output_len": actual_output_len
            })
            
        return requests
    
    def _generate_sharegpt_requests(self, num_requests: int) -> List[Dict[str, Any]]:
        """Load requests from ShareGPT dataset."""
        # This is a placeholder - in real implementation, you would load from actual dataset
        # For now, generate some realistic-looking requests
        prompts = [
            "Write a Python function to sort a list of integers.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Write a short story about a robot.",
        ]
        
        requests = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            requests.append({
                "request_id": f"req_{i}_{int(time.time()*1000)}",
                "prompt": prompt,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "input_len": len(prompt.split()),
                "expected_output_len": 100
            })
            
        return requests
    
    def _load_custom_requests(self, num_requests: int) -> List[Dict[str, Any]]:
        """Load requests from custom dataset file."""
        if not self.dataset_path:
            raise ValueError("dataset_path required for custom dataset")
            
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
            
        # Assume data is a list of {"prompt": ..., "completion": ...} objects
        requests = []
        for i, item in enumerate(data[:num_requests]):
            requests.append({
                "request_id": f"req_{i}_{int(time.time()*1000)}",
                "prompt": item["prompt"],
                "max_new_tokens": len(item.get("completion", "").split()) or 100,
                "temperature": 0.7,
                "input_len": len(item["prompt"].split()),
                "expected_output_len": len(item.get("completion", "").split()) or 100
            })
            
        return requests


class RequestTracker:
    """Track requests through the router."""
    
    def __init__(self, router_url: str):
        self.router_url = router_url
        self.results = []
        
    async def send_and_track_requests(self, requests: List[Dict[str, Any]], 
                                      request_rate: float) -> List[Dict[str, Any]]:
        """Send requests and track their execution."""
        results = []
        
        # Calculate Poisson arrival times
        if request_rate == float('inf'):
            arrival_times = [0] * len(requests)
        else:
            inter_arrival_times = np.random.exponential(1.0 / request_rate, len(requests))
            arrival_times = np.cumsum(inter_arrival_times)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # First, check router health
            try:
                async with session.get(f"{self.router_url}/health") as resp:
                    if resp.status != 200:
                        raise Exception("Router health check failed")
                print("✅ Router health check passed")
            except Exception as e:
                print(f"❌ Router health check failed: {e}")
                return []
            
            # Send requests
            print(f"\nSending {len(requests)} requests at {request_rate} req/s...")
            
            for i, (request, arrival_time) in enumerate(zip(requests, arrival_times)):
                # Wait until arrival time
                current_time = time.time() - start_time
                if arrival_time > current_time:
                    await asyncio.sleep(arrival_time - current_time)
                
                # Record actual arrival time
                actual_arrival_time = time.time()
                
                # Send request
                data = {
                    "text": request["prompt"],
                    "sampling_params": {
                        "max_new_tokens": request["max_new_tokens"],
                        "temperature": request["temperature"]
                    },
                    "stream": False
                }
                
                try:
                    send_time = time.time()
                    async with session.post(
                        f"{self.router_url}/generate",
                        json=data
                    ) as resp:
                        completion_time = time.time()
                        
                        if resp.status == 200:
                            response_text = await resp.text()
                            try:
                                response_data = json.loads(response_text)
                            except json.JSONDecodeError:
                                response_data = {"text": response_text}
                            
                            result = {
                                "request_id": request["request_id"],
                                "success": True,
                                "arrival_time": actual_arrival_time,
                                "send_time": send_time,
                                "completion_time": completion_time,
                                "response": response_data,
                                "input_length": request["input_len"],
                                "expected_output_length": request["expected_output_len"],
                                "status_code": resp.status
                            }
                            
                            print(f"✅ Request {i+1}/{len(requests)} completed")
                        else:
                            result = {
                                "request_id": request["request_id"],
                                "success": False,
                                "arrival_time": actual_arrival_time,
                                "send_time": send_time,
                                "completion_time": completion_time,
                                "error": f"HTTP {resp.status}",
                                "input_length": request["input_len"],
                                "expected_output_length": request["expected_output_len"],
                                "status_code": resp.status
                            }
                            print(f"❌ Request {i+1}/{len(requests)} failed: HTTP {resp.status}")
                            
                except Exception as e:
                    completion_time = time.time()
                    result = {
                        "request_id": request["request_id"],
                        "success": False,
                        "arrival_time": actual_arrival_time,
                        "send_time": send_time,
                        "completion_time": completion_time,
                        "error": str(e),
                        "input_length": request["input_len"],
                        "expected_output_length": request["expected_output_len"],
                        "status_code": 0
                    }
                    print(f"❌ Request {i+1}/{len(requests)} error: {e}")
                
                results.append(result)
            
            # Wait a bit for all requests to complete
            print("\nWaiting for requests to complete...")
            await asyncio.sleep(5)
            
            # Query tracking information
            print("\nQuerying request tracking information...")
            try:
                async with session.get(
                    f"{self.router_url}/v1/traces?limit={len(requests)*2}"
                ) as resp:
                    if resp.status == 200:
                        traces_text = await resp.text()
                        traces = json.loads(traces_text)
                        
                        # Since router tracks requests on arrival (not completion),
                        # and we send requests sequentially, traces should be in the same order
                        # Get the most recent traces (equal to number of requests sent)
                        recent_traces = traces[:len(results)]
                        
                        # Reverse to get chronological order (oldest first)
                        recent_traces.reverse()
                        
                        # Match by order - the nth request corresponds to the nth trace
                        for i, result in enumerate(results):
                            if i < len(recent_traces):
                                trace = recent_traces[i]
                                result["host"] = trace["worker_url"]
                                result["node_id"] = trace.get("node_id", "")
                                result["routing_policy"] = trace.get("routing_policy", "")
                                result["trace_status"] = trace.get("status", "")
                                result["input_tokens"] = trace.get("input_tokens", 0)
                                result["output_tokens"] = trace.get("output_tokens", 0)
                            else:
                                # Not enough traces found
                                result["host"] = "unknown"
                                result["node_id"] = "unknown"
                                result["routing_policy"] = "unknown"
                                result["trace_status"] = "not_found"
                                result["input_tokens"] = 0
                                result["output_tokens"] = 0
                                    
                        print(f"✅ Found tracking info for {sum(1 for r in results if r.get('host') != 'unknown')} requests")
                    else:
                        print(f"❌ Failed to get tracking info: HTTP {resp.status}")
                        for result in results:
                            result["host"] = "error"
                            result["node_id"] = "error"
                            
            except Exception as e:
                print(f"❌ Error getting tracking info: {e}")
                for result in results:
                    result["host"] = "error"
                    result["node_id"] = "error"
        
        return results
    
    def export_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Export results to CSV format matching test_timing_fix.py structure."""
        
        # Calculate relative times from test start
        if results:
            min_time = min(r["arrival_time"] for r in results)
        else:
            min_time = 0
        
        # Prepare records for CSV
        records = []
        for r in results:
            # Calculate latencies in seconds (matching test_timing_fix.py)
            server_latency = r["completion_time"] - r["send_time"] if r["success"] else None
            total_latency = r["completion_time"] - r["arrival_time"] if r["success"] else None
            queue_time = r["send_time"] - r["arrival_time"]
            
            # For TTFT, we don't have streaming data, so approximate as a fraction of server latency
            ttft = server_latency * 0.3 if server_latency else None
            
            record = {
                "req_id": r["request_id"],
                "input_length": r["input_length"],
                "decode_length": r.get("output_tokens", r["expected_output_length"]),
                "arrival_time": r["arrival_time"] - min_time,
                "to_server_time": r["send_time"] - min_time,
                "finish_time": r["completion_time"] - min_time,
                "server_latency": server_latency,
                "total_latency": total_latency,
                "ttft": ttft,
                "queue_time": queue_time,
                "success": r["success"],
                "error": r.get("error", ""),
                "host": r.get("host", "unknown")  # This replaces worker_url and gpu_id
            }
            records.append(record)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        
        # Ensure column order
        columns = ["req_id", "input_length", "decode_length", "arrival_time", 
                  "to_server_time", "finish_time", "server_latency", "total_latency", 
                  "ttft", "queue_time", "success", "error", "host"]
        df = df[columns]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results exported to: {output_path}")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"Total requests: {len(df)}")
        print(f"Successful requests: {df['success'].sum()}")
        print(f"Failed requests: {(~df['success']).sum()}")
        
        if df['success'].any():
            successful_df = df[df['success']]
            print(f"\nLatency Statistics (successful requests):")
            print(f"  Server latency: mean={successful_df['server_latency'].mean():.3f}s, "
                  f"p50={successful_df['server_latency'].median():.3f}s, "
                  f"p99={successful_df['server_latency'].quantile(0.99):.3f}s")
            print(f"  Total latency: mean={successful_df['total_latency'].mean():.3f}s, "
                  f"p50={successful_df['total_latency'].median():.3f}s, "
                  f"p99={successful_df['total_latency'].quantile(0.99):.3f}s")
            print(f"  Queue time: mean={successful_df['queue_time'].mean():.3f}s, "
                  f"p50={successful_df['queue_time'].median():.3f}s, "
                  f"p99={successful_df['queue_time'].quantile(0.99):.3f}s")
        
        # Host distribution
        print(f"\nHost Distribution:")
        host_counts = df['host'].value_counts()
        for host, count in host_counts.items():
            print(f"  {host}: {count} requests ({count/len(df)*100:.1f}%)")


async def main():
    parser = argparse.ArgumentParser(description="Send requests to router and track execution")
    
    # Request generation
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send (default: 100)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        choices=["random", "sharegpt", "custom"],
        help="Dataset type (default: random)"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to custom dataset file (required for custom dataset)"
    )
    
    # Random dataset parameters
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Average input length for random dataset (default: 512)"
    )
    
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Average output length for random dataset (default: 128)"
    )
    
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.5,
        help="Length variation ratio for random dataset (default: 0.5)"
    )
    
    # Request rate
    parser.add_argument(
        "--request-rate",
        type=float,
        default=10.0,
        help="Request rate in requests per second (default: 10.0, inf for max rate)"
    )
    
    # Router configuration
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:40009",
        help="Router URL (default: http://localhost:40009)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: router_test_<timestamp>.csv)"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"router_test_{timestamp}.csv"
    
    # Handle inf request rate
    if args.request_rate == float('inf'):
        print("Using maximum request rate (no delay between requests)")
    
    # Create request generator
    generator = RequestGenerator(args.dataset, args.dataset_path)
    
    # Generate requests
    print(f"Generating {args.num_requests} {args.dataset} requests...")
    requests = generator.generate_requests(
        args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        range_ratio=args.range_ratio
    )
    
    # Create tracker and send requests
    tracker = RequestTracker(args.router_url)
    results = await tracker.send_and_track_requests(requests, args.request_rate)
    
    # Export results
    if results:
        tracker.export_to_csv(results, args.output)
    else:
        print("❌ No results to export")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
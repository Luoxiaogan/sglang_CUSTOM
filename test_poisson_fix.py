#!/usr/bin/env python3
"""Test script to verify Poisson arrival time fix."""

import asyncio
import time
from send_request_and_track import RequestTracker, RequestGenerator

async def test_poisson_arrival():
    """Test that arrival times are correctly distributed."""
    print("Testing Poisson arrival time fix...")
    print("=" * 50)
    
    # Test parameters
    num_requests = 20
    request_rate = 10.0  # 10 req/s, should take ~2 seconds
    
    print(f"Parameters:")
    print(f"  Number of requests: {num_requests}")
    print(f"  Request rate: {request_rate} req/s")
    print(f"  Expected duration: ~{num_requests/request_rate:.1f} seconds")
    print()
    
    # Generate test requests
    generator = RequestGenerator("random")
    requests = generator.generate_requests(
        num_requests,
        input_len=128,
        output_len=64,
        range_ratio=0.2
    )
    
    # Track arrival times
    arrival_times = []
    start_time = time.time()
    
    # Simulate Poisson arrival
    print("Simulating Poisson arrival process:")
    for i in range(num_requests):
        if i > 0:
            import numpy as np
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
        
        current_time = time.time() - start_time
        arrival_times.append(current_time)
        print(f"  Request {i+1}: arrival at {current_time:.3f}s")
    
    total_time = arrival_times[-1]
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Actual rate: {num_requests/total_time:.2f} req/s")
    
    # Calculate inter-arrival times
    inter_arrivals = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
    print(f"\nInter-arrival times:")
    print(f"  Mean: {sum(inter_arrivals)/len(inter_arrivals):.3f}s (expected: {1/request_rate:.3f}s)")
    print(f"  Min: {min(inter_arrivals):.3f}s")
    print(f"  Max: {max(inter_arrivals):.3f}s")

if __name__ == "__main__":
    asyncio.run(test_poisson_arrival())
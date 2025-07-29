#!/usr/bin/env python3
"""
Test script to verify queue time tracking functionality.
This script sends a small number of requests and checks if the new timestamps are properly recorded.
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime


async def test_single_request(router_url: str = "http://localhost:40009"):
    """Send a single request and examine the timestamps."""
    
    print(f"\n{'='*60}")
    print(f"Testing queue timestamp tracking - {datetime.now()}")
    print(f"Router URL: {router_url}")
    print(f"{'='*60}\n")
    
    # Prepare test request
    test_request = {
        "text": "Once upon a time, there was a",
        "sampling_params": {
            "max_new_tokens": 50,
            "temperature": 0.7
        },
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        # Check router health first
        try:
            async with session.get(f"{router_url}/health") as resp:
                if resp.status != 200:
                    print(f"❌ Router health check failed: HTTP {resp.status}")
                    return
                print("✅ Router health check passed")
        except Exception as e:
            print(f"❌ Failed to connect to router: {e}")
            return
        
        # Send test request
        print("\n📤 Sending test request...")
        send_time = time.time()
        
        try:
            async with session.post(f"{router_url}/generate", json=test_request) as resp:
                completion_time = time.time()
                
                if resp.status == 200:
                    response_text = await resp.text()
                    response_data = json.loads(response_text)
                    
                    print(f"✅ Request completed in {completion_time - send_time:.3f}s")
                    
                    # Extract and display timestamps
                    if "meta_info" in response_data:
                        meta_info = response_data["meta_info"]
                        print("\n📊 Timestamp Analysis:")
                        print(f"{'─'*40}")
                        
                        # Server created time
                        server_created = meta_info.get("server_created_time")
                        print(f"server_created_time: {server_created}")
                        
                        # Queue timestamps
                        queue_start = meta_info.get("queue_time_start")
                        queue_end = meta_info.get("queue_time_end")
                        print(f"queue_time_start: {queue_start}")
                        print(f"queue_time_end: {queue_end}")
                        
                        # First token time
                        first_token = meta_info.get("server_first_token_time")
                        print(f"server_first_token_time: {first_token}")
                        
                        print(f"\n⏱️  Time Durations:")
                        print(f"{'─'*40}")
                        
                        # Calculate durations if timestamps available
                        if queue_start and queue_end:
                            pure_queue_time = queue_end - queue_start
                            print(f"Pure queue time: {pure_queue_time:.3f}s")
                        else:
                            print(f"Pure queue time: Not available (queue_start or queue_end missing)")
                        
                        if server_created and first_token:
                            total_server_time = first_token - server_created
                            print(f"Total server time (tokenize + queue + prefill): {total_server_time:.3f}s")
                            
                            if queue_start and queue_end:
                                tokenize_time = queue_start - server_created if queue_start > server_created else 0
                                prefill_time = first_token - queue_end if first_token > queue_end else 0
                                print(f"Estimated tokenize time: {tokenize_time:.3f}s")
                                print(f"Estimated prefill time: {prefill_time:.3f}s")
                        
                        # Check timestamp ordering
                        print(f"\n✔️  Timestamp Validation:")
                        print(f"{'─'*40}")
                        
                        timestamps = []
                        if server_created: timestamps.append(("server_created", server_created))
                        if queue_start: timestamps.append(("queue_start", queue_start))
                        if queue_end: timestamps.append(("queue_end", queue_end))
                        if first_token: timestamps.append(("first_token", first_token))
                        
                        # Check ordering
                        is_ordered = all(timestamps[i][1] <= timestamps[i+1][1] 
                                       for i in range(len(timestamps)-1))
                        
                        if is_ordered:
                            print("✅ Timestamps are in correct chronological order")
                        else:
                            print("❌ Timestamps are NOT in correct order!")
                            for name, ts in timestamps:
                                print(f"  {name}: {ts}")
                    else:
                        print("❌ No meta_info in response")
                    
                    # Show response text
                    print(f"\n📝 Generated text:")
                    print(f"{'─'*40}")
                    if "text" in response_data:
                        print(response_data["text"][:200] + "..." if len(response_data["text"]) > 200 else response_data["text"])
                        
                else:
                    print(f"❌ Request failed: HTTP {resp.status}")
                    error_text = await resp.text()
                    print(f"Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Error sending request: {e}")
            import traceback
            traceback.print_exc()


async def test_multiple_requests(router_url: str = "http://localhost:40009", num_requests: int = 5):
    """Send multiple requests to test queue behavior."""
    
    print(f"\n\n{'='*60}")
    print(f"Testing with {num_requests} concurrent requests")
    print(f"{'='*60}\n")
    
    # Create test requests
    requests = []
    for i in range(num_requests):
        requests.append({
            "text": f"Request {i}: Tell me a fact about the number {i}",
            "sampling_params": {
                "max_new_tokens": 30,
                "temperature": 0.7
            },
            "stream": False
        })
    
    async with aiohttp.ClientSession() as session:
        # Send all requests concurrently
        tasks = []
        send_time = time.time()
        
        for i, request in enumerate(requests):
            task = session.post(f"{router_url}/generate", json=request)
            tasks.append(task)
        
        print(f"📤 Sent {num_requests} concurrent requests")
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        completion_time = time.time()
        
        print(f"✅ All requests completed in {completion_time - send_time:.3f}s")
        
        # Analyze queue times
        queue_times = []
        pure_queue_times = []
        
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"  Request {i}: ❌ Error - {resp}")
                continue
                
            if resp.status == 200:
                data = json.loads(await resp.text())
                if "meta_info" in data:
                    meta = data["meta_info"]
                    
                    # Collect queue times
                    if meta.get("queue_time_start") and meta.get("queue_time_end"):
                        pure_queue = meta["queue_time_end"] - meta["queue_time_start"]
                        pure_queue_times.append(pure_queue)
                        
                    if meta.get("server_created_time") and meta.get("server_first_token_time"):
                        total_queue = meta["server_first_token_time"] - meta["server_created_time"]
                        queue_times.append(total_queue)
                        
                    print(f"  Request {i}: ✅ Success")
            else:
                print(f"  Request {i}: ❌ HTTP {resp.status}")
        
        # Show statistics
        if pure_queue_times:
            print(f"\n📊 Pure Queue Time Statistics ({len(pure_queue_times)} samples):")
            print(f"  Min: {min(pure_queue_times):.3f}s")
            print(f"  Max: {max(pure_queue_times):.3f}s")
            print(f"  Avg: {sum(pure_queue_times)/len(pure_queue_times):.3f}s")
        
        if queue_times:
            print(f"\n📊 Total Server Time Statistics ({len(queue_times)} samples):")
            print(f"  Min: {min(queue_times):.3f}s")
            print(f"  Max: {max(queue_times):.3f}s")
            print(f"  Avg: {sum(queue_times)/len(queue_times):.3f}s")


async def main():
    # Check command line arguments
    router_url = "http://localhost:40009"
    if len(sys.argv) > 1:
        router_url = sys.argv[1]
    
    # Run tests
    await test_single_request(router_url)
    await test_multiple_requests(router_url, num_requests=5)
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
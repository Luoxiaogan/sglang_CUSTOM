#!/usr/bin/env python3
"""
Test SGLang server directly (without router) to isolate timestamp issues.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime


async def test_single_server(server_url: str):
    """Test a single server directly."""
    
    print(f"\n{'='*60}")
    print(f"Testing Server Directly: {server_url}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Simple test request
    test_request = {
        "text": "Hello, this is a direct server test",
        "sampling_params": {
            "max_new_tokens": 20,
            "temperature": 0.7
        },
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        # Check server health
        try:
            health_url = f"{server_url}/health"
            async with session.get(health_url) as resp:
                if resp.status == 200:
                    print(f"✅ Server health check passed")
                else:
                    print(f"❌ Server health check failed: HTTP {resp.status}")
                    return
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return
        
        # Check metrics endpoint
        try:
            metrics_url = f"{server_url}/metrics"
            async with session.get(metrics_url) as resp:
                if resp.status == 200:
                    print(f"✅ Metrics endpoint accessible")
                else:
                    print(f"⚠️  Metrics endpoint returned: HTTP {resp.status}")
        except Exception as e:
            print(f"⚠️  Metrics check failed: {e}")
        
        # Send test request
        print(f"\n📤 Sending test request to {server_url}/generate...")
        send_time = time.time()
        
        try:
            async with session.post(f"{server_url}/generate", json=test_request) as resp:
                recv_time = time.time()
                
                if resp.status == 200:
                    response_text = await resp.text()
                    response_data = json.loads(response_text)
                    
                    print(f"✅ Response received in {recv_time - send_time:.3f}s")
                    
                    # Pretty print the full response
                    print(f"\n📋 Full Response:")
                    print(json.dumps(response_data, indent=2))
                    
                    # Analyze meta_info
                    if "meta_info" in response_data:
                        meta_info = response_data["meta_info"]
                        
                        print(f"\n📊 Timestamp Analysis:")
                        print(f"{'─'*50}")
                        
                        # List all timestamp fields
                        timestamp_fields = [
                            "server_created_time",
                            "server_first_token_time",
                            "queue_time_start", 
                            "queue_time_end",
                            "e2e_latency"
                        ]
                        
                        missing_count = 0
                        for field in timestamp_fields:
                            value = meta_info.get(field)
                            if value is None:
                                print(f"  ❌ {field}: None/Missing")
                                missing_count += 1
                            else:
                                print(f"  ✅ {field}: {value}")
                        
                        # Additional analysis
                        if missing_count > 0:
                            print(f"\n⚠️  {missing_count} timestamp fields are missing")
                            
                            # Check if it's a known issue
                            if meta_info.get("server_first_token_time") is None:
                                print("\n💡 Possible cause: --enable-metrics not used when starting server")
                            
                            if meta_info.get("queue_time_start") is None:
                                print("💡 Possible cause: queue timestamp modifications not in effect")
                        else:
                            print("\n🎉 All timestamps present!")
                    else:
                        print("\n❌ No meta_info in response")
                        
                else:
                    print(f"❌ Request failed: HTTP {resp.status}")
                    error_text = await resp.text()
                    print(f"Error response: {error_text}")
                    
        except Exception as e:
            print(f"❌ Error sending request: {e}")
            import traceback
            traceback.print_exc()


async def test_concurrent_requests(server_url: str, num_requests: int = 5):
    """Send concurrent requests to create queue pressure."""
    
    print(f"\n\n{'='*60}")
    print(f"Testing Concurrent Requests ({num_requests} requests)")
    print(f"{'='*60}\n")
    
    requests = []
    for i in range(num_requests):
        requests.append({
            "text": f"Concurrent request {i}: Tell me about",
            "sampling_params": {
                "max_new_tokens": 50,
                "temperature": 0.7
            },
            "stream": False
        })
    
    async with aiohttp.ClientSession() as session:
        # Send all requests at once
        print(f"📤 Sending {num_requests} concurrent requests...")
        tasks = []
        start_time = time.time()
        
        for i, req in enumerate(requests):
            task = session.post(f"{server_url}/generate", json=req)
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"✅ All requests completed in {end_time - start_time:.3f}s")
        
        # Analyze queue times
        queue_times_found = 0
        queue_durations = []
        
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"  Request {i}: ❌ Error - {resp}")
                continue
                
            if resp.status == 200:
                data = json.loads(await resp.text())
                if "meta_info" in data:
                    meta = data["meta_info"]
                    
                    # Check queue timestamps
                    if meta.get("queue_time_start") and meta.get("queue_time_end"):
                        queue_times_found += 1
                        duration = meta["queue_time_end"] - meta["queue_time_start"]
                        queue_durations.append(duration)
                        print(f"  Request {i}: ✅ Queue duration: {duration:.3f}s")
                    else:
                        print(f"  Request {i}: ⚠️  No queue timestamps")
            else:
                print(f"  Request {i}: ❌ HTTP {resp.status}")
        
        if queue_times_found > 0:
            print(f"\n📊 Queue Time Statistics:")
            print(f"  Found queue times: {queue_times_found}/{num_requests}")
            print(f"  Average duration: {sum(queue_durations)/len(queue_durations):.3f}s")
            print(f"  Max duration: {max(queue_durations):.3f}s")
        else:
            print(f"\n❌ No queue timestamps found in any response")


async def main():
    # Test servers directly
    server_urls = [
        "http://localhost:60005",
        "http://localhost:60006"
    ]
    
    print("="*60)
    print("Direct Server Testing (Bypassing Router)")
    print("="*60)
    
    # Test each server individually
    for url in server_urls:
        await test_single_server(url)
    
    # Test with concurrent requests to create queue pressure
    print("\n" + "="*60)
    print("Creating Queue Pressure")
    print("="*60)
    
    # Pick first server for concurrent test
    await test_concurrent_requests(server_urls[0], num_requests=10)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)
    print("\n💡 If timestamps are still missing:")
    print("1. Run: python check_server_code_version.py")
    print("2. Apply debug patches from debug_scheduler_timestamps.py")
    print("3. Check server logs for [DEBUG] messages")
    print("4. Ensure servers started with --enable-metrics")


if __name__ == "__main__":
    asyncio.run(main())
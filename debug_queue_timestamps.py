#!/usr/bin/env python3
"""
Debug script to check why queue timestamps are not appearing.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime


async def debug_single_request(router_url: str = "http://localhost:40009"):
    """Send a request and debug timestamp issues."""
    
    print(f"\n{'='*60}")
    print(f"Debugging Queue Timestamps - {datetime.now()}")
    print(f"Router URL: {router_url}")
    print(f"{'='*60}\n")
    
    # Simple test request
    test_request = {
        "text": "Hello, this is a test",
        "sampling_params": {
            "max_new_tokens": 20,
            "temperature": 0.7
        },
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        print("📤 Sending test request...")
        
        try:
            async with session.post(f"{router_url}/generate", json=test_request) as resp:
                if resp.status == 200:
                    response_text = await resp.text()
                    response_data = json.loads(response_text)
                    
                    print(f"\n📋 Full Response:")
                    print(json.dumps(response_data, indent=2))
                    
                    if "meta_info" in response_data:
                        meta_info = response_data["meta_info"]
                        print(f"\n📊 Meta Info Fields:")
                        for key, value in meta_info.items():
                            print(f"  {key}: {value}")
                        
                        # Check specific timestamps
                        print(f"\n🔍 Timestamp Analysis:")
                        fields = [
                            "server_created_time",
                            "server_first_token_time", 
                            "queue_time_start",
                            "queue_time_end"
                        ]
                        
                        for field in fields:
                            value = meta_info.get(field)
                            if value is None:
                                print(f"  ❌ {field}: Missing!")
                            else:
                                print(f"  ✅ {field}: {value}")
                        
                        # Check if we can calculate queue times
                        if meta_info.get("queue_time_start") and meta_info.get("queue_time_end"):
                            pure_queue = meta_info["queue_time_end"] - meta_info["queue_time_start"]
                            print(f"\n  ✅ Pure queue time: {pure_queue:.3f}s")
                        else:
                            print(f"\n  ❌ Cannot calculate pure queue time")
                            
                else:
                    print(f"❌ Request failed: HTTP {resp.status}")
                    error_text = await resp.text()
                    print(f"Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def check_metrics_endpoint(server_urls: list):
    """Check if metrics are enabled on the servers."""
    
    print(f"\n\n{'='*60}")
    print(f"Checking Metrics Status")
    print(f"{'='*60}\n")
    
    async with aiohttp.ClientSession() as session:
        for url in server_urls:
            try:
                # Try to access metrics endpoint
                metrics_url = f"{url}/metrics"
                async with session.get(metrics_url) as resp:
                    if resp.status == 200:
                        print(f"✅ {url}: Metrics enabled")
                    else:
                        print(f"❌ {url}: Metrics endpoint returned {resp.status}")
            except Exception as e:
                print(f"❌ {url}: Failed to check metrics - {e}")


async def main():
    router_url = "http://localhost:60009"
    server_urls = ["http://localhost:60005", "http://localhost:60006"]
    
    # Check metrics on servers
    await check_metrics_endpoint(server_urls)
    
    # Debug request
    await debug_single_request(router_url)
    
    print(f"\n💡 Troubleshooting Tips:")
    print(f"1. Ensure servers were started with --enable-metrics")
    print(f"2. Check if queue_time_start/end are being set in scheduler")
    print(f"3. Verify BatchTokenIDOut is passing the timestamps correctly")
    print(f"4. Check if tokenizer_manager is receiving the timestamps")


if __name__ == "__main__":
    asyncio.run(main())
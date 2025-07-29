#!/usr/bin/env python3
"""
Verify that queue timestamps are now working correctly.
"""

import asyncio
import aiohttp
import json
import pandas as pd
from datetime import datetime


async def test_queue_timestamps():
    """Test queue timestamp functionality."""
    
    router_url = "http://localhost:40009"
    
    print(f"\n{'='*60}")
    print(f"Verifying Queue Timestamp Fix - {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Test request
    test_request = {
        "text": "Test request for queue timestamps",
        "sampling_params": {
            "max_new_tokens": 30,
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
                    
                    if "meta_info" in response_data:
                        meta_info = response_data["meta_info"]
                        
                        print("\n✅ Response received successfully")
                        print("\n📊 Timestamp Report:")
                        print(f"{'─'*50}")
                        
                        # Check all timestamps
                        timestamps = {
                            "server_created_time": meta_info.get("server_created_time"),
                            "queue_time_start": meta_info.get("queue_time_start"),
                            "queue_time_end": meta_info.get("queue_time_end"),
                            "server_first_token_time": meta_info.get("server_first_token_time")
                        }
                        
                        all_present = True
                        for name, value in timestamps.items():
                            if value is None:
                                print(f"  ❌ {name}: Missing")
                                all_present = False
                            else:
                                print(f"  ✅ {name}: {value:.6f}")
                        
                        if all_present:
                            print(f"\n🎉 All timestamps present!")
                            
                            # Calculate durations
                            tokenize_time = timestamps["queue_time_start"] - timestamps["server_created_time"]
                            pure_queue_time = timestamps["queue_time_end"] - timestamps["queue_time_start"]
                            prefill_time = timestamps["server_first_token_time"] - timestamps["queue_time_end"]
                            total_time = timestamps["server_first_token_time"] - timestamps["server_created_time"]
                            
                            print(f"\n⏱️  Calculated Durations:")
                            print(f"{'─'*50}")
                            print(f"  Tokenize time: {tokenize_time:.3f}s")
                            print(f"  Pure queue time: {pure_queue_time:.3f}s")
                            print(f"  Prefill time: {prefill_time:.3f}s")
                            print(f"  Total server time: {total_time:.3f}s")
                            
                            # Verify time ordering
                            if (timestamps["server_created_time"] <= timestamps["queue_time_start"] <= 
                                timestamps["queue_time_end"] <= timestamps["server_first_token_time"]):
                                print(f"\n✅ Timestamps are in correct order")
                            else:
                                print(f"\n❌ Timestamps are NOT in correct order!")
                        else:
                            print(f"\n❌ Some timestamps are missing. Fix not fully working.")
                            
                            # Additional debugging info
                            print(f"\n🔍 Full meta_info:")
                            print(json.dumps(meta_info, indent=2))
                    else:
                        print("❌ No meta_info in response")
                else:
                    print(f"❌ Request failed: HTTP {resp.status}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def test_with_csv(num_requests: int = 5):
    """Run a small test and check the CSV output."""
    
    print(f"\n\n{'='*60}")
    print(f"Running CSV Test with {num_requests} requests")
    print(f"{'='*60}\n")
    
    import subprocess
    import os
    from datetime import datetime
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"queue_test_{timestamp}.csv"
    
    # Run send_req.py
    cmd = [
        "python", "send_req.py",
        "--num-requests", str(num_requests),
        "--request-rate", "5",
        "--output", csv_file
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            
            # Check CSV file
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                print(f"\n📊 CSV Analysis:")
                print(f"Total rows: {len(df)}")
                
                # Check queue time columns
                queue_cols = ['queue_time_start', 'queue_time_end', 'pure_queue_time']
                for col in queue_cols:
                    if col in df.columns:
                        non_null = df[col].notna().sum()
                        print(f"  {col}: {non_null}/{len(df)} non-null values")
                        if non_null > 0:
                            print(f"    Mean: {df[col].mean():.3f}s")
                    else:
                        print(f"  {col}: Column not found!")
                
                # Show first few rows
                print(f"\n📋 First 3 rows (queue time columns):")
                cols_to_show = ['req_id'] + queue_cols
                cols_to_show = [c for c in cols_to_show if c in df.columns]
                print(df[cols_to_show].head(3).to_string())
                
                # Clean up
                os.remove(csv_file)
                print(f"\n🧹 Cleaned up {csv_file}")
            else:
                print(f"❌ CSV file {csv_file} not found")
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error running test: {e}")


async def main():
    # Single request test
    await test_queue_timestamps()
    
    # CSV test
    await test_with_csv()
    
    print(f"\n{'='*60}")
    print("Verification completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
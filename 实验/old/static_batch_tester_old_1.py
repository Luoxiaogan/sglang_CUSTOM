import argparse
import asyncio
import random
import time
import aiohttp
import numpy as np

# =================================================================
# --- 核心请求函数 ---
# =================================================================
async def send_request(session: aiohttp.ClientSession, url: str, request_id: int, prompt: str, max_new_tokens: int):
    """异步发送单个HTTP请求到SGLang服务器"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": False,
    }
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return True
            else:
                print(f"Request {request_id} failed with status: {response.status}")
                return False
    except Exception as e:
        print(f"Request {request_id} failed with exception: {e}")
        return False

# =================================================================
# --- 主测试逻辑 ---
# =================================================================
async def main(args):
    print("--- Static Batching with Poisson Arrivals Benchmark ---")
    print(f"Server URL: {args.url}")
    print(f"Target Poisson Rate: {args.rate} req/s")
    print(f"Static Batch Size: {args.batch_size}")
    print(f"Total Batches to Test: {args.num_batches}")
    print("-----------------------------------------------------")

    prompt = "San Francisco is a" 

    async with aiohttp.ClientSession() as session:
        request_queue = asyncio.Queue()
        latencies = []
        request_id_counter = 0

        async def request_generator():
            nonlocal request_id_counter
            total_requests = args.num_batches * args.batch_size
            while request_id_counter < total_requests:
                inter_arrival_time = random.expovariate(args.rate)
                await asyncio.sleep(inter_arrival_time)
                request_id_counter += 1
                await request_queue.put(request_id_counter)

        async def batch_processor():
            for i in range(args.num_batches):
                batch_to_send_ids = []
                for _ in range(args.batch_size):
                    req_id = await request_queue.get()
                    batch_to_send_ids.append(req_id)
                
                print(f"Batch #{i+1}: Formed a batch of {len(batch_to_send_ids)}. Sending concurrently...")
                
                tasks = [send_request(session, args.url, req_id, prompt, args.output_len) for req_id in batch_to_send_ids]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                batch_latency = end_time - start_time
                success_count = sum(1 for r in results if r)

                if success_count == args.batch_size:
                    latencies.append(batch_latency)
                    print(f"Batch #{i+1}: Processed successfully in {batch_latency:.4f} seconds.\n")
                else:
                    print(f"Batch #{i+1}: Failed with {args.batch_size - success_count} errors.\n")

        gen_task = asyncio.create_task(request_generator())
        proc_task = asyncio.create_task(batch_processor())
        await proc_task
        gen_task.cancel()

    print("\n--- Benchmark Finished ---")
    if latencies:
        print(f"Number of successful batches: {len(latencies)}")
        print(f"Average static batch latency: {np.mean(latencies):.4f} seconds")
        print(f"Median static batch latency: {np.median(latencies):.4f} seconds")
        print(f"P95 static batch latency: {np.percentile(latencies, 95):.4f} seconds")
        
        req_throughput = args.batch_size / np.mean(latencies)
        token_throughput = (args.batch_size * args.output_len) / np.mean(latencies)
        print(f"Equivalent Request Throughput: {req_throughput:.2f} req/s")
        print(f"Equivalent Token Throughput: {token_throughput:.2f} tokens/s")
    else:
        print("No batches were processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A custom benchmark tool for static batching with Poisson arrivals.")
    parser.add_argument("--url", type=str, default="http://localhost:31000/generate", help="SGLang server generate endpoint URL.")
    parser.add_argument("--rate", type=float, default=20, help="Poisson arrival rate (requests per second).")
    parser.add_argument("--batch-size", type=int, default=10, help="The fixed size of each static batch.")
    parser.add_argument("--num-batches", type=int, default=10, help="Total number of batches to test.")
    parser.add_argument("--input-len", type=int, default=100, help="Approximate number of input tokens (for information only).")
    parser.add_argument("--output-len", type=int, default=300, help="Number of output tokens (decode steps).")
    
    args = parser.parse_args()
    
    # 删除了 sgl.set_default_backend(...) 和相关的 SGLANG_URL 全局变量设置
    
    asyncio.run(main(args))
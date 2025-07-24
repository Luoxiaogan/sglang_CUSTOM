import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import List

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# =================================================================
# --- 数据类 & Prompt生成函数 (无变化) ---
# =================================================================
@dataclass
class RequestResult:
    success: bool
    latency: float = 0.0
    error: str = ""
    prompt_len: int = 0
    output_len: int = 0

def generate_random_prompt(tokenizer: PreTrainedTokenizerBase, token_len: int) -> str:
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_len)
    return tokenizer.decode(selected_tokens)


# =================================================================
# --- 核心请求函数 (无变化) ---
# =================================================================
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> RequestResult:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            end_time = time.time()
            latency = end_time - start_time
            if response.status == 200:
                return RequestResult(
                    success=True,
                    latency=latency,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )
            else:
                error_msg = f"Request {request_id} failed with status: {response.status}"
                # (日志优化) 失败时只返回，让上层统一打印
                return RequestResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Request {request_id} failed with exception: {e}"
        return RequestResult(success=False, error=error_msg)


# =================================================================
# --- 指标计算与保存函数 (无变化) ---
# =================================================================
def calculate_and_save_metrics(
    args: argparse.Namespace,
    results: List[RequestResult],
    duration_s: float,
    output_file: str,
):
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        print("No successful requests to analyze.")
        return

    latencies_ms = [r.latency * 1000 for r in successful_results]
    
    metrics = {
        "backend": "sglang-static-batch",
        "dataset_name": "random",
        "request_rate": args.rate,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "duration": duration_s,
        "completed": len(successful_results),
        "total_input_tokens": sum(r.prompt_len for r in successful_results),
        "total_output_tokens": sum(r.output_len for r in successful_results),
        "request_throughput": len(successful_results) / duration_s,
        "input_throughput": sum(r.prompt_len for r in successful_results) / duration_s,
        "output_throughput": sum(r.output_len for r in successful_results) / duration_s,
        "mean_e2e_latency_ms": np.mean(latencies_ms),
        "median_e2e_latency_ms": np.median(latencies_ms),
        "std_e2e_latency_ms": np.std(latencies_ms),
        "p99_e2e_latency_ms": np.percentile(latencies_ms, 99),
    }

    print(f"\nWriting results for rate {args.rate} to {output_file}...")
    with open(output_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print("Done.")


# =================================================================
# --- 主测试逻辑 (增加日志) ---
# =================================================================
async def main(args):
    print("--- Static Batching with Poisson Arrivals Benchmark ---")
    print(f"Server URL: {args.url}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target Input Length: {args.input_len} tokens")
    print(f"Target Output Length: {args.output_len} tokens")
    print(f"Target Poisson Rate: {args.rate} req/s")
    print(f"Static Batch Size: {args.batch_size}")
    print(f"Total Batches to Test: {args.num_batches}")
    print(f"Running test for rate: {args.rate} req/s")
    print("-----------------------------------------------------")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    total_requests_to_generate = args.num_batches * args.batch_size
    prompts = [generate_random_prompt(tokenizer, args.input_len) for _ in range(total_requests_to_generate)]
    
    all_results: List[RequestResult] = []
    
    async with aiohttp.ClientSession() as session:
        request_queue = asyncio.Queue()
        
        async def request_generator():
            # (日志新增) 每10%打印一次生成进度
            log_interval = max(1, len(prompts) // 10)
            for i, prompt in enumerate(prompts):
                inter_arrival_time = random.expovariate(args.rate)
                await asyncio.sleep(inter_arrival_time)
                await request_queue.put((i + 1, prompt))
                if (i + 1) % log_interval == 0:
                    print(f"[Generator] Queued {i + 1}/{len(prompts)} requests...")
            print("[Generator] All requests have been generated and queued.")


        async def batch_processor():
            for i in range(args.num_batches):
                batch_to_send = []
                # (日志新增) 表明开始等待收集批次
                print(f"\n[Processor] Batch #{i+1}: Waiting to collect {args.batch_size} requests...")
                
                for _ in range(args.batch_size):
                    req_id, prompt = await request_queue.get()
                    batch_to_send.append((req_id, prompt))
                
                # (日志新增) 表明批次已凑齐，即将发送
                print(f"[Processor] Batch #{i+1}: Batch collected. Sending {args.batch_size} requests concurrently...")
                
                tasks = [
                    send_request(session, args.url, req_id, prompt, args.input_len, args.output_len)
                    for req_id, prompt in batch_to_send
                ]
                
                batch_start_time = time.time()
                batch_results = await asyncio.gather(*tasks)
                batch_end_time = time.time()
                batch_latency = batch_end_time - batch_start_time
                all_results.extend(batch_results)

                # (日志新增) 表明SGLang服务器已完成处理
                print(f"[Processor] Batch #{i+1}: All {args.batch_size} requests completed by server.")

                # (日志优化) 优化最终的批处理结果日志
                success_count = sum(1 for r in batch_results if r.success)
                if success_count == args.batch_size:
                    print(f"[Processor] Batch #{i+1}: Result: SUCCESS | Batch Latency: {batch_latency:.4f}s")
                else:
                    print(f"[Processor] Batch #{i+1}: Result: FAILED with {args.batch_size - success_count} errors.")
                    # 打印具体的错误信息
                    for res in batch_results:
                        if not res.success:
                            print(f"  - Error: {res.error}")


        benchmark_start_time = time.time()
        
        gen_task = asyncio.create_task(request_generator())
        proc_task = asyncio.create_task(batch_processor())
        
        await proc_task
        gen_task.cancel()
        
        benchmark_end_time = time.time()

    total_duration = benchmark_end_time - benchmark_start_time
    print(f"\n--- Benchmark Finished for Rate {args.rate} ---")
    print(f"Total duration: {total_duration:.2f} seconds")
    
    calculate_and_save_metrics(args, all_results, total_duration, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A custom benchmark tool for static batching with Poisson arrivals.")
    parser.add_argument("--url", type=str, default="http://localhost:30000/generate", help="SGLang server generate endpoint URL.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Name or path of the tokenizer to use for prompt generation.")
    parser.add_argument("--rate", type=float, required=True, help="Poisson arrival rate (requests per second).")
    parser.add_argument("--batch-size", type=int, default=10, help="The fixed size of each static batch.")
    parser.add_argument("--num-batches", type=int, default=20, help="Total number of batches to test.")
    parser.add_argument("--input-len", type=int, default=128, help="Approximate number of input tokens.")
    parser.add_argument("--output-len", type=int, default=128, help="Number of output tokens (decode steps).")
    parser.add_argument("--output-file", type=str, default="results.jsonl", help="File to append results to.")
    
    args = parser.parse_args()
    asyncio.run(main(args))
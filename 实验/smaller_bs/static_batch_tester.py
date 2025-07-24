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
# --- 数据类 & Prompt生成函数 ---
# =================================================================
@dataclass
class RequestResult:
    """用于存储单个请求的所有结果和延迟信息"""
    success: bool
    # 新增：拆分为两种延迟
    latency_on_server: float = 0.0  # 服务器处理延迟
    latency_total: float = 0.0      # 包含排队等待的总延迟
    error: str = ""
    prompt_len: int = 0
    output_len: int = 0

def generate_random_prompt(tokenizer: PreTrainedTokenizerBase, token_len: int) -> str:
    """生成指定token长度的随机prompt"""
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_len)
    return tokenizer.decode(selected_tokens)


# =================================================================
# --- 核心请求函数 ---
# =================================================================
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> RequestResult:
    """异步发送单个HTTP请求并只返回服务器处理延迟"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }
    # 这个函数内的计时，只衡量服务器处理和网络往返时间
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            end_time = time.time()
            server_latency = end_time - start_time
            if response.status == 200:
                # 注意：这里只填充了 server_latency
                return RequestResult(
                    success=True,
                    latency_on_server=server_latency,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )
            else:
                error_msg = f"Request {request_id} failed with status: {response.status}"
                return RequestResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Request {request_id} failed with exception: {e}"
        return RequestResult(success=False, error=error_msg)


# =================================================================
# --- 指标计算与保存函数 (已更新) ---
# =================================================================
def calculate_and_save_metrics(
    args: argparse.Namespace,
    results: List[RequestResult],
    duration_s: float,
    output_file: str,
):
    """计算并保存两种延迟的统计数据"""
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        print("No successful requests to analyze.")
        return

    # 分别提取两种延迟数据
    server_latencies_ms = [r.latency_on_server * 1000 for r in successful_results]
    total_latencies_ms = [r.latency_total * 1000 for r in successful_results]
    
    # 构造包含两种延迟指标的 metrics 字典
    metrics = {
        "backend": "sglang-static-batch",
        "dataset_name": "random",
        "request_rate": args.rate,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "duration_from_first_request": duration_s, # 明确duration的含义
        "completed": len(successful_results),
        "total_input_tokens": sum(r.prompt_len for r in successful_results),
        "total_output_tokens": sum(r.output_len for r in successful_results),
        "request_throughput": len(successful_results) / duration_s,
        "input_throughput": sum(r.prompt_len for r in successful_results) / duration_s,
        "output_throughput": sum(r.output_len for r in successful_results) / duration_s,
        
        "mean_server_latency_ms": np.mean(server_latencies_ms),
        "median_server_latency_ms": np.median(server_latencies_ms),
        "std_server_latency_ms": np.std(server_latencies_ms),
        "p99_server_latency_ms": np.percentile(server_latencies_ms, 99),
        
        "mean_total_latency_ms": np.mean(total_latencies_ms),
        "median_total_latency_ms": np.median(total_latencies_ms),
        "std_total_latency_ms": np.std(total_latencies_ms),
        "p99_total_latency_ms": np.percentile(total_latencies_ms, 99),
    }

    print(f"\nWriting results for rate {args.rate} to {output_file}...")
    with open(output_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print("Done.")


# =================================================================
# --- 主测试逻辑 (已更新) ---
# =================================================================
async def main(args):
    print("--- Static Batching with Poisson Arrivals Benchmark ---")
    # ... (打印参数部分无变化) ...
    print(f"Running test for rate: {args.rate} req/s")
    print("-----------------------------------------------------")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    total_requests_to_generate = args.num_batches * args.batch_size
    prompts = [generate_random_prompt(tokenizer, args.input_len) for _ in range(total_requests_to_generate)]
    
    all_results: List[RequestResult] = []
    
    async with aiohttp.ClientSession() as session:
        # 队列中现在存放元组 (id, prompt, arrival_time)
        request_queue = asyncio.Queue()
        # 新增：用于记录第一个请求的到达时间
        first_request_arrival_time = None
        
        async def request_generator():
            nonlocal first_request_arrival_time
            log_interval = max(1, len(prompts) // 10)
            for i, prompt in enumerate(prompts):
                inter_arrival_time = random.expovariate(args.rate)
                await asyncio.sleep(inter_arrival_time)
                
                # 核心修改(1): 记录请求的“出生”时间
                arrival_time = time.time()
                
                # 记录第一个请求的到达时间，用于计算总时长
                if first_request_arrival_time is None:
                    first_request_arrival_time = arrival_time
                    print(f"First request arrived at {arrival_time:.2f}")
                
                # 将到达时间与请求一起放入队列
                await request_queue.put((i + 1, prompt, arrival_time))

                if (i + 1) % log_interval == 0:
                    print(f"[Generator] Queued {i + 1}/{len(prompts)} requests...")
            print("[Generator] All requests have been generated and queued.")


        async def batch_processor():
            for i in range(args.num_batches):
                batch_to_process = []
                print(f"\n[Processor] Batch #{i+1}: Waiting to collect {args.batch_size} requests...")
                
                for _ in range(args.batch_size):
                    # 从队列中解包出到达时间
                    req_id, prompt, arrival_time = await request_queue.get()
                    batch_to_process.append((req_id, prompt, arrival_time))
                
                print(f"[Processor] Batch #{i+1}: Batch collected. Sending {args.batch_size} requests concurrently...")
                
                tasks = [
                    send_request(session, args.url, req_id, prompt, args.input_len, args.output_len)
                    for req_id, prompt, _ in batch_to_process
                ]
                
                # 服务器返回的结果只包含 server_latency
                server_results = await asyncio.gather(*tasks)
                
                # 核心修改(2): 在收到批处理结果后，计算每个请求的 total_latency
                # 所有请求在同一时间完成，这是静态批处理的定义
                batch_completion_time = time.time()

                final_results_for_batch = []
                for idx, server_res in enumerate(server_results):
                    if server_res.success:
                        # 从我们保存的批次信息中取出对应的到达时间
                        arrival_time = batch_to_process[idx][2]
                        # 计算并填充 total_latency
                        server_res.latency_total = batch_completion_time - arrival_time
                    final_results_for_batch.append(server_res)

                all_results.extend(final_results_for_batch)

                print(f"[Processor] Batch #{i+1}: All {args.batch_size} requests completed by server.")
                # ... (日志部分无变化) ...
                success_count = sum(1 for r in final_results_for_batch if r.success)
                if success_count == args.batch_size:
                    batch_duration = batch_completion_time - min(t for _,_,t in batch_to_process)
                    print(f"[Processor] Batch #{i+1}: Result: SUCCESS | Batch Completion Time: {batch_duration:.4f}s")

        
        proc_task = asyncio.create_task(batch_processor())
        gen_task = asyncio.create_task(request_generator())
        
        await proc_task
        gen_task.cancel()
        
        # 核心修改(3): 使用第一个请求的到达时间作为总时长的起点
        last_request_completion_time = time.time()
        
    if first_request_arrival_time:
        total_duration = last_request_completion_time - first_request_arrival_time
    else:
        total_duration = 0

    print(f"\n--- Benchmark Finished for Rate {args.rate} ---")
    print(f"Total duration (from first request arrival): {total_duration:.2f} seconds")
    
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
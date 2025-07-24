import argparse
import asyncio
import random
import time
import aiohttp
import numpy as np
# 新增: 引入 AutoTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# =================================================================
# --- (新增) 学习bench_serving.py，用于生成指定长度的Prompt ---
# =================================================================
def generate_random_prompt(tokenizer: PreTrainedTokenizerBase, token_len: int) -> str:
    """
    使用分词器生成一个指定Token长度的随机Prompt.
    这个逻辑借鉴自 bench_serving.py 中的 gen_prompt 函数。
    """
    # 从分词器的词汇表中获取所有可用的token ID
    all_available_tokens = list(tokenizer.get_vocab().values())
    # 随机选择 token_len 个 token ID
    selected_tokens = random.choices(all_available_tokens, k=token_len)
    # 将这些ID解码成文本字符串
    return tokenizer.decode(selected_tokens)

# =================================================================
# --- 核心请求函数 (无变化) ---
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
                return (True, "")
            else:
                error_msg = f"Request {request_id} failed with status: {response.status}"
                print(error_msg)
                return (False, error_msg)
    except Exception as e:
        error_msg = f"Request {request_id} failed with exception: {e}"
        print(error_msg)
        return (False, error_msg)

# =================================================================
# --- 主测试逻辑 (有修改) ---
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
    print("-----------------------------------------------------")

    # --- (修改) 1. 加载分词器 ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # --- (修改) 2. 在测试开始前，预先生成所有Prompts ---
    total_requests_to_generate = args.num_batches * args.batch_size
    print(f"Generating {total_requests_to_generate} prompts with length ~{args.input_len} tokens...")
    prompts = [generate_random_prompt(tokenizer, args.input_len) for _ in range(total_requests_to_generate)]
    print("Prompt generation finished.")
    
    # -----------------------------------------------------

    async with aiohttp.ClientSession() as session:
        # (修改) 队列现在将存储 (request_id, prompt) 元组
        request_queue = asyncio.Queue()
        latencies = []
        request_id_counter = 0

        # (修改) 请求生成器现在从预生成的列表中获取prompt
        async def request_generator():
            nonlocal request_id_counter
            for i in range(len(prompts)):
                inter_arrival_time = random.expovariate(args.rate)
                await asyncio.sleep(inter_arrival_time)
                
                request_id_counter += 1
                prompt_to_send = prompts[i]
                await request_queue.put((request_id_counter, prompt_to_send))

        # (修改) 批处理器现在从队列中解包prompt
        async def batch_processor():
            for i in range(args.num_batches):
                batch_to_send = []
                for _ in range(args.batch_size):
                    # 从队列中获取 (req_id, prompt)
                    req_id, prompt = await request_queue.get()
                    batch_to_send.append((req_id, prompt))
                
                print(f"Batch #{i+1}: Formed a batch of {len(batch_to_send)}. Sending concurrently...")
                
                # 创建任务时，传入对应的prompt
                tasks = [send_request(session, args.url, req_id, prompt, args.output_len) for req_id, prompt in batch_to_send]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                batch_latency = end_time - start_time
                # results现在是(success, error_msg)的元组列表
                success_count = sum(1 for r, _ in results if r)

                if success_count == args.batch_size:
                    latencies.append(batch_latency)
                    print(f"Batch #{i+1}: Processed successfully in {batch_latency:.4f} seconds.\n")
                else:
                    print(f"Batch #{i+1}: Failed with {args.batch_size - success_count} errors.\n")

        gen_task = asyncio.create_task(request_generator())
        proc_task = asyncio.create_task(batch_processor())
        
        # 等待所有批次处理完毕
        await proc_task
        # 取消可能还在等待sleep的生成器任务
        gen_task.cancel()

    print("\n--- Benchmark Finished ---")
    if latencies:
        print(f"Number of successful batches: {len(latencies)}")
        print(f"Average static batch latency: {np.mean(latencies):.4f} seconds")
        print(f"Median static batch latency: {np.median(latencies):.4f} seconds")
        print(f"P95 static batch latency: {np.percentile(latencies, 95):.4f} seconds")
        
        # 计算吞吐量
        req_throughput = args.batch_size / np.mean(latencies)
        # (修改) 计算Token吞吐量时，现在输入和输出长度都是准确的
        token_throughput = (args.batch_size * (args.input_len + args.output_len)) / np.mean(latencies)
        output_token_throughput = (args.batch_size * args.output_len) / np.mean(latencies)

        print(f"Equivalent Request Throughput: {req_throughput:.2f} req/s")
        print(f"Equivalent Output Token Throughput: {output_token_throughput:.2f} tokens/s")
        print(f"Equivalent Total Token Throughput: {token_throughput:.2f} tokens/s (Input + Output)")

    else:
        print("No batches were processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A custom benchmark tool for static batching with Poisson arrivals.")
    parser.add_argument("--url", type=str, default="http://localhost:31001/generate", help="SGLang server generate endpoint URL.")
    # (新增) 增加tokenizer参数
    parser.add_argument("--tokenizer", type=str, default="/home/lg/arc/model", help="Name or path of the tokenizer to use for prompt generation.")
    parser.add_argument("--rate", type=float, default=20, help="Poisson arrival rate (requests per second).")
    parser.add_argument("--batch-size", type=int, default=10, help="The fixed size of each static batch.")
    parser.add_argument("--num-batches", type=int, default=20, help="Total number of batches to test.")
    # (修改) 参数现在可以被正确使用了
    parser.add_argument("--input-len", type=int, default=128, help="Approximate number of input tokens.")
    parser.add_argument("--output-len", type=int, default=128, help="Number of output tokens (decode steps).")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
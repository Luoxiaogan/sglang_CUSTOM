import subprocess
import time
import asyncio
import pandas as pd
from test_marginal_utility import send_request
import aiohttp

async def benchmark_policy(policy_name, num_requests=100):
    """对单个策略进行基准测试"""
    print(f"\n=== 测试 {policy_name} 策略 ===")
    
    # 启动路由器
    router_process = subprocess.Popen([
        "python", "start_router.py",
        "--host", "0.0.0.0",
        "--port", "30009",
        "--policy", policy_name,
        # "--prometheus-port", "29001",
        "--enable-tracking",
        "--workers", "http://localhost:30001 http://localhost:30002 http://localhost:30003"
    ])
    
    # 等待路由器启动
    time.sleep(5)
    
    try:
        # 发送测试请求
        router_url = "http://localhost:30009"
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            tasks = []
            
            for i in range(num_requests):
                prompt = f"Generate a random number and explain why you chose it."
                max_tokens = 100
                tasks.append(send_request(session, router_url, prompt, max_tokens))
            
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            success_count = len([r for r in results if r])
            
            metrics = {
                "policy": policy_name,
                "total_requests": num_requests,
                "successful_requests": success_count,
                "duration": duration,
                "throughput": success_count / duration,
                "success_rate": success_count / num_requests
            }
            
            return metrics
            
    finally:
        # 停止路由器
        router_process.terminate()
        router_process.wait()
        time.sleep(2)

async def run_comparison():
    """运行策略对比测试"""
    policies = ["round_robin", "random", "cache_aware", "marginal_utility"]
    results = []
    
    for policy in policies:
        metrics = await benchmark_policy(policy)
        results.append(metrics)
        time.sleep(5)  # 策略之间的间隔
    
    # 生成报告
    df = pd.DataFrame(results)
    print("\n=== 性能对比结果 ===")
    print(df.to_string(index=False))
    
    # 保存结果
    df.to_csv("policy_comparison_results.csv", index=False)
    print("\n结果已保存到 policy_comparison_results.csv")

if __name__ == "__main__":
    asyncio.run(run_comparison())
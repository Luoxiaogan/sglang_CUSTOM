import asyncio
import aiohttp
import json
import time
import random

async def send_request(session, router_url, prompt, max_tokens):
    """发送单个请求到路由器"""
    data = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.8,
        }
    }
    
    try:
        async with session.post(f"{router_url}/generate", json=data) as resp:
            result = await resp.json()
            return result
    except Exception as e:
        print(f"Request failed: {e}")
        return None

async def test_marginal_utility_routing():
    """测试边际效用路由策略"""
    router_url = "http://localhost:29000"
    
    # 测试阶段 1: 冷启动（数据不足，应该使用负载均衡）
    print("=== 阶段 1: 冷启动测试 ===")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(15):
            prompt = f"Tell me a story about {i}"
            max_tokens = random.randint(50, 200)
            tasks.append(send_request(session, router_url, prompt, max_tokens))
        
        results = await asyncio.gather(*tasks)
        print(f"完成 {len([r for r in results if r])} 个请求")
    
    # 等待一段时间让系统处理
    await asyncio.sleep(5)
    
    # 测试阶段 2: 梯度路由（有足够历史数据）
    print("\n=== 阶段 2: 梯度路由测试 ===")
    async with aiohttp.ClientSession() as session:
        # 发送更多请求，观察路由决策
        for batch in range(5):
            print(f"\n批次 {batch + 1}:")
            tasks = []
            for i in range(20):
                prompt = f"Explain the concept of {random.choice(['AI', 'ML', 'DL', 'NLP'])}"
                max_tokens = random.randint(100, 300)
                tasks.append(send_request(session, router_url, prompt, max_tokens))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            success_count = len([r for r in results if r])
            print(f"  完成: {success_count}/20 请求")
            print(f"  耗时: {duration:.2f} 秒")
            print(f"  吞吐量: {success_count/duration:.2f} req/s")
            
            await asyncio.sleep(2)
    
    # 测试阶段 3: 负载不均测试
    print("\n=== 阶段 3: 负载不均测试 ===")
    async with aiohttp.ClientSession() as session:
        # 发送不同长度的请求，测试策略是否能识别性能差异
        tasks = []
        
        # 短请求批次
        print("发送短请求...")
        for i in range(30):
            prompt = "Hi"
            tasks.append(send_request(session, router_url, prompt, 10))
        
        # 长请求批次
        print("发送长请求...")
        for i in range(10):
            prompt = "Write a detailed essay about artificial intelligence, covering its history, current applications, and future prospects. Include specific examples and technical details."
            tasks.append(send_request(session, router_url, prompt, 500))
        
        results = await asyncio.gather(*tasks)
        print(f"混合负载测试完成: {len([r for r in results if r])}/40 成功")

if __name__ == "__main__":
    asyncio.run(test_marginal_utility_routing())
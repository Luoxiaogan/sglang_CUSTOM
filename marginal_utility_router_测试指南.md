# MarginalUtilityRouter 测试指南

## 1. 概述

本文档提供了测试新实现的 MarginalUtilityRouter（边际效用路由策略）的完整指南。该策略通过分析历史性能数据的梯度来做出智能路由决策。

## 2. 环境准备

### 2.1 编译 Rust 代码

首先需要在有 Rust 环境的机器上编译 sgl-router：

```bash
cd /path/to/sglang/sgl-router
cargo build --release
maturin build --release
```

### 2.2 安装 Python 绑定

编译完成后，安装 Python 绑定：

```bash
cd /path/to/sglang
pip install -e sgl-router/
```

## 3. 功能测试

### 3.1 单元测试

运行 Rust 单元测试以验证核心功能：

```bash
cd /path/to/sglang/sgl-router
cargo test marginal_utility
```

预期结果：
- 所有测试应该通过
- 特别关注以下测试：
  - `test_gradient_calculation` - 验证梯度计算逻辑
  - `test_fallback_to_load_balancing` - 验证数据不足时的回退机制
  - `test_pd_mode_selection` - 验证 PD 模式兼容性

### 3.2 集成测试

#### 3.2.1 启动 SGLang 服务器

在多个 GPU 上启动 SGLang 服务器：

```bash
# GPU 0
python -m sglang.launch_server \
  --model-path /nas/models/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30001 \
  --base-gpu-id 0 \
  --enable-metrics

# GPU 1
python -m sglang.launch_server \
  --model-path /nas/models/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30002 \
  --base-gpu-id 1 \
  --enable-metrics

# GPU 2
python -m sglang.launch_server \
  --model-path /nas/models/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30003 \
  --base-gpu-id 2 \
  --enable-metrics
```

#### 3.2.2 启动路由器（使用新策略）

```bash
python start_router.py \
  --host 0.0.0.0 \
  --port 29000 \
  --policy marginal_utility \
  --prometheus-port 29001 \
  --enable-request-tracking \
  --log-level debug \
  localhost:30001 localhost:30002 localhost:30003
```

#### 3.2.3 发送测试请求

使用测试脚本发送请求并观察路由行为：

```python
# test_marginal_utility.py
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
```

## 4. 性能对比测试

### 4.1 准备对比测试脚本

创建一个脚本来对比不同路由策略的性能：

```python
# compare_policies.py
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
        "--port", "29000",
        "--policy", policy_name,
        "--prometheus-port", "29001",
        "--enable-request-tracking",
        "localhost:30001", "localhost:30002", "localhost:30003"
    ])
    
    # 等待路由器启动
    time.sleep(5)
    
    try:
        # 发送测试请求
        router_url = "http://localhost:29000"
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
```

### 4.2 监控和分析

#### 4.2.1 查看 Prometheus 指标

访问 `http://localhost:29001/metrics` 查看路由器指标：

重点关注：
- `router_requests_total` - 总请求数
- `router_request_duration_seconds` - 请求延迟分布
- `router_worker_load` - 各 worker 的负载

#### 4.2.2 查看请求追踪信息

```bash
# 获取请求追踪信息
curl http://localhost:29000/v1/traces | jq '.'
```

#### 4.2.3 分析日志

检查路由器日志中的梯度计算信息：

```bash
grep "gradient analysis" router.log
```

应该看到类似：
```
Worker http://localhost:30001 gradient analysis: throughput_grad=2.345, latency_grad=-0.123, score=1.456
```

## 5. 验证要点

### 5.1 功能验证

1. **冷启动行为**
   - 初始请求应该相对均匀分布
   - 日志中应显示 "using load-based scoring"

2. **梯度计算**
   - 有足够历史数据后，日志应显示 "gradient analysis"
   - 分数计算应该合理（吞吐量提升得高分，延迟降低得高分）

3. **性能趋势识别**
   - 如果某个 worker 性能持续改善，应该获得更多请求
   - 如果某个 worker 性能下降，应该获得更少请求

### 5.2 性能验证

1. **吞吐量**
   - MarginalUtility 应该达到或超过 RoundRobin 的吞吐量
   - 在负载不均的情况下，应该优于简单策略

2. **延迟**
   - P99 延迟应该保持稳定
   - 不应该出现严重的延迟峰值

3. **负载分布**
   - 长期运行后，负载分布应该反映各 worker 的实际处理能力

## 6. 故障排查

### 6.1 常见问题

1. **路由器启动失败**
   ```bash
   # 检查端口占用
   lsof -i :29000
   
   # 检查 Rust 编译
   cd sgl-router && cargo check
   ```

2. **没有梯度计算**
   - 检查 `min_history_for_trend` 设置（默认 10）
   - 确认请求完成并返回了 meta_info

3. **性能异常**
   - 检查 `window_size` 是否合适（默认 20）
   - 调整 `throughput_weight` 和 `latency_weight`

### 6.2 调试模式

启用详细日志：

```bash
RUST_LOG=debug python start_router.py \
  --policy marginal_utility \
  --log-level debug \
  ...
```

## 7. 参数调优

### 7.1 配置参数

可以通过修改代码中的默认值来调整策略行为：

```rust
// sgl-router/src/policies/marginal_utility.rs
pub struct MarginalUtilityConfig {
    pub window_size: usize,              // 历史窗口大小（默认 20）
    pub min_history_for_trend: usize,    // 最小历史记录数（默认 10）
    pub throughput_weight: f64,          // 吞吐量权重（默认 0.6）
    pub latency_weight: f64,             // 延迟权重（默认 0.4）
}
```

### 7.2 调优建议

1. **高吞吐量场景**
   - 增加 `throughput_weight` 到 0.8
   - 减少 `latency_weight` 到 0.2

2. **低延迟场景**
   - 减少 `throughput_weight` 到 0.3
   - 增加 `latency_weight` 到 0.7

3. **快速响应变化**
   - 减少 `window_size` 到 10-15
   - 减少 `min_history_for_trend` 到 5

## 8. 总结

MarginalUtility 路由策略通过分析历史性能趋势，能够智能地将请求路由到性能正在改善的 worker。测试时应重点验证：

1. 策略能够正确识别性能趋势
2. 在数据不足时能够优雅降级到负载均衡
3. 长期运行的稳定性和性能优势

通过本指南的测试，您应该能够全面验证新策略的功能和性能表现。
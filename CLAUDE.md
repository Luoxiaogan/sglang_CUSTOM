# SGLang Testing Framework Documentation

## 项目概述

SGLang Testing Framework 是一个专门为 SGLang 设计的综合测试框架，支持单节点（单 GPU）和多节点（多 GPU）测试场景。框架提供了灵活的路由策略以及详细的性能指标收集功能。SGLang 自动处理批处理，无需手动配置批处理策略。

## 核心功能

### 1. 测试模式

#### 1.1 节点级测试（Node Level Testing）
- 在单个 GPU 上启动一个 SGLang 服务器
- 可配置服务器参数，如 `--max-running-requests`、`--tp-size`、`--mem-fraction-static` 等
- 收集详细的延迟和吞吐量指标
- 支持张量并行（Tensor Parallelism）配置

#### 1.2 路由级测试（Routing Level Testing）
- 在多个 GPU 上启动多个 SGLang 服务器
- 支持 SGLang Router 的路由策略：
  - Cache-Aware Routing（缓存感知路由）
  - Round Robin（轮询）
  - Random（随机）
  - Shortest Queue（最短队列）
- 支持自定义路由策略扩展
- 注意：动态参数更新功能待 SGLang 支持相应 API

### 2. 指标收集

框架收集以下关键指标：

#### 2.1 延迟指标
- **Server Latency（服务器延迟）**：请求被发送到服务器到完成推理的时间
- **Total Latency（总延迟）**：请求生成（到达）到完成的总时间，包括排队时间
- **TTFT（Time to First Token）**：首个 token 生成时间
- **ITL（Inter-Token Latency）**：token 间延迟

#### 2.2 吞吐量指标
- **Request Throughput**：每秒处理的请求数
- **Token Throughput**：每秒处理的 token 数（输入/输出）
- **Concurrency**：并发处理能力

#### 2.3 系统指标
- **Queue Depth**：队列深度
- **Resource Utilization**：资源利用率
- **Cache Hit Rate**：缓存命中率（路由模式）

### 3. 请求生成

- 基于泊松分布生成请求到达时间
- 支持多种数据集：ShareGPT、Random、Custom
- 可配置输入/输出长度分布
- 精确记录三个关键时刻（遵循 bench_serving_new.py 模式）：
  1. 请求到达时刻（睡眠后记录，即实际到达时刻）
  2. 请求发送到服务器时刻
  3. 请求完成时刻

## 安装与配置

### 环境要求

```bash
# Python 版本
Python >= 3.8

# 必需依赖
sglang >= 0.3.0
sglang-router >= 0.1.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
aiohttp >= 3.8.0
tqdm >= 4.62.0
```

### 安装步骤

```bash
# 1. 克隆 SGLang 仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 2. 安装 SGLang
pip install -e python/

# 3. 安装 SGLang Router
pip install sglang-router

# 4. 安装测试框架依赖
cd sglang_test_framework
pip install -r requirements.txt
```

## 使用指南

### 1. 单节点测试

```python
from sglang_test_framework.tests.node_test import NodeTest
from sglang_test_framework.config.node_config import NodeConfig

# 配置单节点测试
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    gpu_id=0,
    max_running_requests=256,
    tp_size=1,  # 张量并行大小
    request_rate=10.0,  # 每秒 10 个请求
    num_prompts=1000,
    dataset_name="sharegpt"
)

# 运行测试
test = NodeTest(config)
results = test.run()

# 分析结果
test.analyze_results(results)
```

### 2. 多节点路由测试

```python
from sglang_test_framework.tests.routing_test import RoutingTest
from sglang_test_framework.config.routing_config import RoutingConfig

# 配置路由测试
config = RoutingConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    num_gpus=4,
    routing_policy="cache_aware",  # 可选: "round_robin", "random", "shortest_queue"
    request_rate=40.0,
    num_prompts=5000,
    max_running_requests=256  # 每个服务器的最大并发请求数
)

# 运行测试
test = RoutingTest(config)
results = test.run()

# 可视化结果
test.visualize_results(results)
```

### 3. 自定义路由策略

```python
from sglang_test_framework.strategies.routing.base import BaseRoutingPolicy

class CustomRoutingPolicy(BaseRoutingPolicy):
    def __init__(self, workers):
        super().__init__(workers)
        
    def route_request(self, request, server_metrics):
        """
        根据服务器指标路由请求
        
        Args:
            request: 输入请求
            server_metrics: 各服务器的性能指标
                - throughput: 吞吐量
                - latency: 平均延迟
                - queue_depth: 队列深度
                - memory_usage: 内存使用率
        
        Returns:
            worker_id: 选中的服务器 ID
        """
        # 实现自定义路由逻辑
        # 例如：选择综合得分最高的服务器
        scores = []
        for worker_id, metrics in server_metrics.items():
            score = (metrics['throughput'] / 
                    (metrics['latency'] * (1 + metrics['queue_depth'])))
            scores.append((worker_id, score))
        
        # 返回得分最高的服务器
        return max(scores, key=lambda x: x[1])[0]
```

### 4. API 类型支持

```python
# 使用原生 SGLang API
async with RequestSender() as sender:
    result = await sender.send_request(
        request, 
        api_url="http://localhost:30000/generate",
        api_type="sglang"
    )

# 使用 OpenAI 兼容 API
async with RequestSender() as sender:
    result = await sender.send_request(
        request, 
        api_url="http://localhost:30000/v1/chat/completions",
        api_type="openai"
    )
```

### 5. 动态参数更新（待支持）

```python
# 注意：SGLang 目前不支持通过 API 动态更新服务器参数
# 框架保留了相关接口，以便将来 SGLang 支持此功能时可以使用

# 当前可以通过重启服务器来更改参数
server_manager.stop_server("worker_0")
config.max_running_requests = 512
server_manager.launch_server(config)
```

## API 参考

### 配置类

#### NodeConfig

```python
class NodeConfig:
    model_path: str              # 模型路径
    gpu_id: int                  # GPU ID
    port: int = 30000           # 服务端口
    max_running_requests: int    # 最大并发请求数
    mem_fraction_static: float   # 静态内存分配比例
    tp_size: int = 1            # 张量并行大小
    request_rate: float          # 请求速率
    num_prompts: int            # 测试请求数
    dataset_name: str           # 数据集名称
    random_input_len: int       # 随机输入长度
    random_output_len: int      # 随机输出长度
```

#### RoutingConfig

```python
class RoutingConfig:
    model_path: str              # 模型路径
    num_gpus: int               # GPU 数量
    routing_policy: str         # 路由策略（cache_aware, round_robin, random, shortest_queue）
    request_rate: float         # 总请求速率
    num_prompts: int           # 测试请求数
    dataset_name: str          # 数据集名称
    max_running_requests: int   # 每个服务器的最大请求数
    router_config: RouterConfig # 路由器配置对象
```

### 核心组件

#### ServerManager

```python
class ServerManager:
    def launch_server(config: ServerConfig) -> SGLangServer
    def launch_multiple_servers(configs: List[ServerConfig]) -> List[SGLangServer]
    def launch_router(config: RouterConfig, worker_urls: List[str]) -> RouterManager
    def stop_server(server_id: str) -> None
    def update_server_param(server_id: str, param: str, value: Any) -> bool  # 目前返回 False，待 SGLang 支持
    def get_server_metrics(server_id: str) -> Dict[str, Any]
```

#### RequestGenerator

```python
class RequestGenerator:
    def generate_requests(config: Dict) -> List[Request]
    def generate_poisson_arrivals(rate: float, duration: float) -> List[float]
    def generate_length_distribution(config: Dict) -> Tuple[List[int], List[int]]
```

#### MetricsCollector

```python
class MetricsCollector:
    def start_collection() -> None
    def stop_collection() -> None
    def get_metrics() -> Dict[str, Any]
    def export_metrics(format: str, path: str) -> None
```

## 最佳实践

### 1. 测试前准备

- 确保 GPU 内存充足
- 关闭不必要的 GPU 进程
- 预热服务器（建议运行 100-200 个请求）
- 验证模型加载正确

### 2. 参数调优建议

#### 单节点测试
- 从较小的 `max_running_requests` 开始（如 64）
- 逐步增加直到观察到性能饱和
- 监控 GPU 内存使用，避免 OOM
- 调整 `mem_fraction_static` 以优化内存使用

#### 路由测试
- 确保各节点负载均衡
- 监控缓存命中率（cache-aware 策略）
- 根据实际场景选择合适的路由策略
- 考虑网络延迟对性能的影响

### 3. 结果分析

- 关注 P95/P99 延迟而非仅平均值
- 分析请求排队时间占比
- 观察吞吐量与延迟的权衡关系
- 记录系统资源使用情况

### 4. 故障排查

#### 常见问题

1. **服务器启动失败**
   - 检查端口占用
   - 验证模型路径
   - 确认 GPU 可用性

2. **OOM 错误**
   - 减小 `max_running_requests`
   - 降低 `mem_fraction_static`
   - 使用更小的批次大小

3. **路由不均衡**
   - 检查路由策略配置
   - 验证服务器健康状态
   - 调整负载均衡阈值

## 扩展开发

### 添加新的路由策略

1. 继承 `BaseRoutingPolicy`
2. 实现 `route_request` 方法
3. 注册到策略工厂

```python
from sglang_test_framework.strategies.routing.base import BaseRoutingPolicy

class MyRoutingPolicy(BaseRoutingPolicy):
    def route_request(self, request, server_metrics):
        # 实现路由逻辑
        # 返回选中的 worker_id
        pass

# 注册策略
RoutingFactory.register("my_policy", MyRoutingPolicy)
```

### 添加新的指标

1. 扩展 `MetricsCollector`
2. 添加指标收集逻辑
3. 更新结果分析器

```python
class CustomMetricsCollector(MetricsCollector):
    def collect_custom_metric(self):
        # 收集自定义指标
        pass
```

## 性能优化建议

### 1. 请求生成优化
- 预生成请求以减少运行时开销
- 使用异步 IO 提高并发性
- 批量发送请求减少网络开销

### 2. 指标收集优化
- 使用采样而非全量收集
- 异步写入结果文件
- 定期聚合指标减少内存使用

### 3. 路由优化
- 实现请求预测机制
- 使用缓存减少重复计算
- 考虑请求亲和性

## 更新日志

### v1.0.0 (2024-07-24)
- 初始版本发布
- 支持单节点和多节点测试
- 实现基础路由策略
- 完整的指标收集系统

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 编写测试
5. 提交 PR

### 代码规范
- 遵循 PEP 8
- 添加类型注解
- 编写文档字符串
- 保持测试覆盖率 > 80%

## 许可证

本项目遵循 Apache 2.0 许可证。

## 联系方式

- GitHub Issues: [sglang/issues](https://github.com/sgl-project/sglang/issues)
- 文档: [本文档]

## 附录

### 性能测试示例结果

```
==================== Node Test Results ====================
Model: meta-llama/Llama-2-7b-hf
GPU: NVIDIA A100-SXM4-80GB
Max Running Requests: 256
Request Rate: 10 req/s

Throughput Metrics:
- Request Throughput: 9.87 req/s
- Input Token Throughput: 5,032 tok/s
- Output Token Throughput: 1,258 tok/s

Latency Metrics:
- Mean Server Latency: 245.3 ms
- Mean Total Latency: 312.7 ms
- P95 Server Latency: 489.2 ms
- P99 Server Latency: 612.8 ms

Queue Metrics:
- Mean Queue Time: 67.4 ms
- Max Queue Depth: 42
==========================================================
```

### 配置模板

#### 高吞吐量配置
```yaml
node_config:
  max_running_requests: 512
  mem_fraction_static: 0.85
  chunked_prefill_size: 8192
  enable_torch_compile: true
  
routing_config:
  policy: "cache_aware"
  balance_abs_threshold: 32
  balance_rel_threshold: 1.0001
  cache_threshold: 0.5
```

#### 低延迟配置
```yaml
node_config:
  max_running_requests: 32
  mem_fraction_static: 0.9
  schedule_conservativeness: 0.5
  
routing_config:
  policy: "shortest_queue"
  # shortest_queue 策略会选择队列最短的服务器
```

### 常用脚本

#### 批量测试脚本
```bash
#!/bin/bash
# test_sweep.sh

for mrs in 32 64 128 256 512; do
    echo "Testing with max_running_requests=$mrs"
    python -m sglang_test_framework.tests.node_test \
        --max-running-requests $mrs \
        --output-file "results_mrs_$mrs.json"
done
```

#### 结果对比脚本
```python
# compare_results.py
import pandas as pd
import matplotlib.pyplot as plt

def compare_results(result_files):
    """对比多个测试结果"""
    data = []
    for file in result_files:
        df = pd.read_json(file, lines=True)
        data.append(df)
    
    # 生成对比图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 吞吐量对比
    # 延迟对比
    # ...
    
    plt.tight_layout()
    plt.savefig("comparison.png")
```

## 知识库引用

本框架的设计和实现基于以下 SGLang 文档：

1. **服务器参数** - 参见 `.DUCUMENT/Server_Arguments.md`
2. **路由实现** - 参见 `.DUCUMENT/SGLang_Router_详解.md`
3. **基准测试** - 参见 `.DUCUMENT/Benchmark_and_Profiling.md`
4. **采样参数** - 参见 `.DUCUMENT/Sampling_Parameters.md`
5. **超参数调优** - 参见 `.DUCUMENT/Hyperparameter_Tuning.md`

所有 API 使用必须查询 `.DUCUMENT` 目录下的相关文档以确保正确性。
# SGLang Router 详解：高性能分布式推理路由系统

## 目录

1. [概述](#概述)
2. [系统架构](#系统架构)
3. [核心方法论](#核心方法论)
4. [安装与部署](#安装与部署)
5. [使用指南](#使用指南)
6. [API 参考](#api-参考)
7. [配置参数详解](#配置参数详解)
8. [最佳实践](#最佳实践)
9. [故障排查](#故障排查)
10. [性能优化](#性能优化)

## 概述

SGLang Router 是一个高性能的分布式推理路由系统，专门为大语言模型（LLM）的数据并行推理而设计。它通过独特的缓存感知负载均衡算法，在多个 GPU 上运行的 SGLang Runtime 实例之间智能分配请求，显著提升整体推理性能。

### 核心特性

- **缓存感知路由**：通过近似基数树（Approximate Radix Tree）追踪每个工作节点的缓存内容，优化缓存命中率
- **智能负载均衡**：动态在缓存感知和最短队列策略之间切换，确保系统高效运行
- **高性能实现**：核心路由逻辑采用 Rust 实现，通过 PyO3 提供 Python 绑定
- **容错机制**：自动重试、健康检查和故障节点隔离
- **动态扩缩容**：支持运行时添加或移除工作节点
- **Kubernetes 原生**：内置服务发现，支持自动扩缩容
- **PD 解耦架构**：支持 Prefill-Decode 分离部署，优化推理性能

## 系统架构

### 技术栈

```
┌─────────────────────────────────────────────────────┐
│                   Python API Layer                   │
│              (launch_router.py, launch_server.py)    │
├─────────────────────────────────────────────────────┤
│                    PyO3 Bindings                     │
├─────────────────────────────────────────────────────┤
│                   Rust Core Engine                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Router    │  │ Load Balancer│  │   Cache    │ │
│  │   Logic     │  │   Policies   │  │  Manager   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Actix-web  │  │   Metrics    │  │ Service    │ │
│  │   Server    │  │ (Prometheus) │  │ Discovery  │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 组件说明

1. **Router Core**：处理请求路由逻辑，支持普通路由和 PD 解耦路由
2. **Load Balancer**：实现多种负载均衡策略
3. **Cache Manager**：管理缓存状态的近似基数树
4. **Health Monitor**：监控工作节点健康状态
5. **Metrics Collector**：收集和暴露 Prometheus 指标
6. **Service Discovery**：Kubernetes 服务发现集成

## 核心方法论

### 1. 缓存感知路由算法

缓存感知路由是 SGLang Router 的核心创新，它通过维护每个工作节点的近似缓存状态来优化请求分配。

#### 近似基数树（Approximate Radix Tree）

每个工作节点维护一个基数树，存储处理过的请求前缀：

```
Worker 1 Tree:
├── "What"
│   ├── " is"
│   │   ├── " the capital"
│   │   └── " machine learning"
│   └── " are"
└── "How"
    └── " does"

Worker 2 Tree:
├── "Explain"
│   └── " quantum"
└── "What"
    └── " is"
        └── " Python"
```

#### 路由决策流程

```python
def route_request(request_text, workers):
    # 1. 检查系统是否平衡
    if is_system_imbalanced(workers):
        # 使用最短队列策略
        return shortest_queue_routing(workers)
    
    # 2. 缓存感知路由
    best_match = None
    best_match_rate = 0
    
    for worker in workers:
        match_rate, matched_prefix = worker.tree.prefix_match(request_text)
        if match_rate > best_match_rate:
            best_match = worker
            best_match_rate = match_rate
    
    # 3. 根据匹配率决定路由策略
    if best_match_rate > cache_threshold:
        # 高匹配率：路由到最佳匹配的工作节点
        return best_match
    else:
        # 低匹配率：路由到缓存空间最大的节点
        return worker_with_smallest_tree()
```

#### 系统平衡检测

系统被认为不平衡的条件：

```python
def is_system_imbalanced(workers):
    loads = [w.pending_requests for w in workers]
    max_load = max(loads)
    min_load = min(loads)
    
    abs_imbalanced = (max_load - min_load) > balance_abs_threshold
    rel_imbalanced = max_load > balance_rel_threshold * min_load
    
    return abs_imbalanced and rel_imbalanced
```

### 2. 负载均衡策略

#### 最短队列（Shortest Queue）

在系统不平衡时使用，将请求路由到待处理请求最少的工作节点：

```python
def shortest_queue_routing(workers):
    return min(workers, key=lambda w: w.pending_requests)
```

#### 其他策略

- **随机路由**：随机选择工作节点
- **轮询路由**：按顺序循环选择
- **二选一（Power of Two）**：随机选择两个节点，取负载较小者

### 3. 多租户支持

缓存树支持多租户隔离，每个租户的缓存独立管理：

```rust
pub struct Tree {
    tenants: DashMap<String, Tenant>,  // 租户ID -> 租户数据
}

pub struct Tenant {
    root: Arc<RwLock<Node>>,
    size: AtomicUsize,
    last_access: AtomicU64,
}
```

### 4. LRU 缓存淘汰

后台线程定期执行 LRU 淘汰，防止缓存树无限增长：

```rust
// 每 eviction_interval 秒执行一次
fn evict_lru_nodes() {
    for (worker_id, tree) in trees.iter() {
        if tree.size() > max_tree_size {
            tree.evict_least_recently_used();
        }
    }
}
```

## 安装与部署

### 基础安装

```bash
# 安装 SGLang Router
pip install sglang-router
```

### 部署模式

#### 1. 联合启动模式（推荐用于单节点）

Router 和 Runtime 一起启动，自动管理进程：

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

#### 2. 分离启动模式（推荐用于多节点）

先启动工作节点：

```bash
# 节点1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30001

# 节点2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30002
```

再启动路由器：

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:30001 http://node2:30002 \
    --host 0.0.0.0 \
    --port 30000
```

#### 3. Kubernetes 部署

使用服务发现自动管理工作节点：

```yaml
# router-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sglang-router
  template:
    metadata:
      labels:
        app: sglang-router
    spec:
      containers:
      - name: router
        image: sglang/router:latest
        command:
        - python
        - -m
        - sglang_router.launch_router
        args:
        - --k8s-namespace=default
        - --k8s-pod-selector=app=sglang-worker
        - --host=0.0.0.0
        - --port=30000
```

#### 4. PD 解耦部署

支持 Prefill 和 Decode 服务器分离：

```bash
# 启动 Prefill 服务器
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30001 \
    --enable-pd-disagg \
    --disagg-prefill-role

# 启动 Decode 服务器
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30002 \
    --enable-pd-disagg \
    --disagg-decode-role

# 启动 PD Router
python -m sglang_router.launch_router \
    --mode pd \
    --prefill-urls http://localhost:30001 \
    --decode-urls http://localhost:30002
```

## 使用指南

### 基本使用

#### 1. 发送推理请求

```python
import requests

# SGLang 原生格式
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "什么是机器学习？",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 100
        }
    }
)
print(response.json())

# OpenAI 兼容格式
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "什么是机器学习？"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
print(response.json())
```

#### 2. 批量请求示例

```python
import asyncio
import aiohttp

async def send_request(session, prompt):
    async with session.post(
        "http://localhost:30000/generate",
        json={"text": prompt}
    ) as response:
        return await response.json()

async def batch_inference(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

# 使用示例
prompts = ["问题1", "问题2", "问题3", ...]
results = asyncio.run(batch_inference(prompts))
```

### 动态管理

#### 1. 添加工作节点

```bash
# 添加新的工作节点
curl -X POST http://localhost:30000/add_worker?url=http://new-worker:30003

# Python 示例
import requests
response = requests.post(
    "http://localhost:30000/add_worker",
    params={"url": "http://new-worker:30003"}
)
```

#### 2. 移除工作节点

```bash
# 移除工作节点
curl -X POST http://localhost:30000/remove_worker?url=http://worker:30002

# Python 示例
response = requests.post(
    "http://localhost:30000/remove_worker",
    params={"url": "http://worker:30002"}
)
```

#### 3. 查看系统状态

```python
# 列出所有工作节点
workers = requests.get("http://localhost:30000/list_workers").json()

# 获取负载信息
loads = requests.get("http://localhost:30000/get_loads").json()

# 健康检查
health = requests.get("http://localhost:30000/health").json()
```

## API 参考

### 推理端点

| 端点 | 方法 | 描述 | 示例 |
|------|------|------|------|
| `/generate` | POST | SGLang 原生推理 | `{"text": "...", "sampling_params": {...}}` |
| `/v1/chat/completions` | POST | OpenAI Chat API 兼容 | OpenAI 格式 |
| `/v1/completions` | POST | OpenAI Completion API 兼容 | OpenAI 格式 |

### 管理端点

| 端点 | 方法 | 描述 | 参数 |
|------|------|------|------|
| `/add_worker` | POST | 添加工作节点 | `url`: 工作节点地址 |
| `/remove_worker` | POST | 移除工作节点 | `url`: 工作节点地址 |
| `/list_workers` | GET | 列出所有工作节点 | 无 |
| `/flush_cache` | POST | 清空所有缓存 | 无 |

### 监控端点

| 端点 | 方法 | 描述 | 返回值 |
|------|------|------|--------|
| `/health` | GET | 健康状态 | Router 和工作节点状态 |
| `/health_generate` | GET | 测试推理能力 | 简单推理结果 |
| `/get_loads` | GET | 负载信息 | 各节点待处理请求数 |
| `/metrics` | GET | Prometheus 指标 | Prometheus 格式指标 |

### 信息端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/get_server_info` | GET | 服务器配置信息 |
| `/v1/models` | GET | 可用模型列表 |
| `/readiness` | GET | K8s 就绪探针 |
| `/liveness` | GET | K8s 存活探针 |

## 配置参数详解

### Router 配置

```python
# launch_router.py 参数
--host                    # 监听地址，默认 127.0.0.1
--port                    # 监听端口，默认 30000
--worker-urls             # 工作节点 URL 列表
--mode                    # 路由模式: regular(默认) 或 pd
--policy                  # 负载均衡策略，默认 cache_aware

# 缓存感知策略参数
--cache-threshold         # 缓存匹配阈值 (0.0-1.0)，默认 0.5
--balance-abs-threshold   # 绝对负载差阈值，默认 32
--balance-rel-threshold   # 相对负载比阈值，默认 1.0001
--eviction-interval-secs  # LRU 淘汰间隔（秒），默认 60
--max-tree-size          # 最大树节点数，默认 16777216

# 性能参数
--max-payload-size        # 最大请求大小，默认 10485760 (10MB)
--request-timeout-secs    # 请求超时时间，默认 90
--max-worker-retries      # 单个工作节点最大重试次数，默认 3
--max-total-retries       # 总最大重试次数，默认 6

# Kubernetes 服务发现
--k8s-namespace           # K8s 命名空间
--k8s-pod-selector        # Pod 标签选择器
--k8s-port               # 工作节点端口，默认 30000

# 指标
--metrics-enabled         # 启用 Prometheus 指标
--metrics-port           # 指标端口，默认 29000
```

### 联合启动配置

```python
# launch_server.py 额外参数
--model-path              # 模型路径或 Hugging Face ID
--dp-size                 # 数据并行大小（GPU 数量）
--base-gpu-id            # 起始 GPU ID，默认 0

# SGLang Runtime 参数会传递给每个工作节点
--max-running-requests    # 最大并发请求数
--mem-fraction-static     # 静态内存分配比例
--chunked-prefill-size   # 分块预填充大小
```

### 策略特定配置

#### 缓存感知策略

```python
{
    "policy": "cache_aware",
    "cache_aware": {
        "cache_threshold": 0.5,        # 使用缓存路由的最小匹配率
        "balance_abs_threshold": 32,   # 负载不平衡的绝对阈值
        "balance_rel_threshold": 1.0001, # 负载不平衡的相对阈值
        "eviction_interval_secs": 60,  # LRU 淘汰间隔
        "max_tree_size": 16777216      # 最大树大小
    }
}
```

#### 二选一策略

```python
{
    "policy": "pow_two",
    "pow_two": {
        "base_policy": "shortest_queue"  # 基础策略：random 或 shortest_queue
    }
}
```

## 最佳实践

### 1. 性能优化

#### 缓存阈值调优

```python
# 对于重复性高的工作负载，降低缓存阈值
--cache-threshold 0.3

# 对于多样性高的工作负载，提高缓存阈值
--cache-threshold 0.7
```

#### 负载均衡调优

```python
# 对于突发流量，使用更敏感的平衡检测
--balance-abs-threshold 16
--balance-rel-threshold 1.5

# 对于稳定流量，使用更宽松的平衡检测
--balance-abs-threshold 64
--balance-rel-threshold 2.0
```

### 2. 部署建议

#### 高可用部署

```yaml
# 使用多副本 Router
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-router
spec:
  replicas: 3  # 多副本
  template:
    spec:
      affinity:
        podAntiAffinity:  # 反亲和性
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - sglang-router
            topologyKey: kubernetes.io/hostname
```

#### 资源配置

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

### 3. 监控告警

#### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sglang-router'
    static_configs:
      - targets: ['router1:29000', 'router2:29000']
    metrics_path: '/metrics'
```

#### 关键指标监控

```promql
# 请求成功率
rate(sglang_router_requests_total{status="success"}[5m]) /
rate(sglang_router_requests_total[5m])

# 平均延迟
histogram_quantile(0.95, 
  rate(sglang_router_request_duration_seconds_bucket[5m])
)

# 工作节点负载分布
sglang_router_worker_pending_requests

# 缓存命中率
rate(sglang_router_cache_hits_total[5m]) /
rate(sglang_router_requests_total[5m])
```

### 4. 容量规划

#### 工作节点数量估算

```python
# 基于目标 QPS 和单节点容量
required_workers = ceil(target_qps / single_worker_qps * safety_factor)

# 示例：目标 1000 QPS，单节点 100 QPS，安全系数 1.2
required_workers = ceil(1000 / 100 * 1.2) = 12
```

#### 内存需求估算

```python
# Router 内存 = 基础内存 + 缓存树内存
cache_tree_memory = max_tree_size * avg_node_size * num_workers
router_memory = 1GB + cache_tree_memory

# 示例：16M 节点，128 字节/节点，4 个工作节点
cache_tree_memory = 16777216 * 128 * 4 / 1024 / 1024 / 1024 = 8GB
router_memory = 1GB + 8GB = 9GB
```

## 故障排查

### 常见问题

#### 1. 工作节点连接失败

```bash
# 检查工作节点状态
curl http://worker:30001/health

# 检查网络连通性
telnet worker 30001

# 查看 Router 日志
journalctl -u sglang-router -f
```

#### 2. 请求路由不均衡

```python
# 检查负载分布
loads = requests.get("http://router:30000/get_loads").json()
print(f"Load distribution: {loads}")

# 调整平衡阈值
--balance-abs-threshold 16
--balance-rel-threshold 1.2
```

#### 3. 缓存命中率低

```python
# 分析请求模式
# 如果请求过于随机，考虑调整策略
--policy shortest_queue

# 或增加缓存阈值
--cache-threshold 0.7
```

#### 4. 内存使用过高

```bash
# 减小树大小限制
--max-tree-size 8388608

# 缩短淘汰间隔
--eviction-interval-secs 30
```

### 调试技巧

#### 1. 启用详细日志

```bash
export RUST_LOG=debug
python -m sglang_router.launch_router --worker-urls ...
```

#### 2. 测试单个工作节点

```python
# 直接测试工作节点，绕过 Router
response = requests.post(
    "http://worker:30001/generate",
    json={"text": "test"}
)
```

#### 3. 监控缓存状态

```python
# 自定义监控脚本
import time
import requests

while True:
    metrics = requests.get("http://router:30000/metrics").text
    cache_hits = extract_metric(metrics, "cache_hits_total")
    total_requests = extract_metric(metrics, "requests_total")
    hit_rate = cache_hits / total_requests if total_requests > 0 else 0
    print(f"Cache hit rate: {hit_rate:.2%}")
    time.sleep(10)
```

## 性能优化

### 1. 网络优化

#### 使用 HTTP/2

```python
# 在 Router 配置中启用 HTTP/2
{
    "http2_enabled": true,
    "http2_max_concurrent_streams": 1000
}
```

#### 连接池配置

```python
# 调整连接池大小
{
    "connection_pool_size": 100,
    "max_idle_connections": 50,
    "idle_connection_timeout": 30
}
```

### 2. 批处理优化

```python
# 客户端批处理
async def batch_requests(prompts, batch_size=10):
    batches = [prompts[i:i+batch_size] 
               for i in range(0, len(prompts), batch_size)]
    
    results = []
    for batch in batches:
        batch_results = await asyncio.gather(*[
            send_request(prompt) for prompt in batch
        ])
        results.extend(batch_results)
    
    return results
```

### 3. 缓存预热

```python
# 预热常见请求
def warmup_cache(common_prompts):
    """预热缓存以提高命中率"""
    for prompt in common_prompts:
        requests.post(
            "http://router:30000/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1}
            }
        )
```

### 4. 负载测试

```python
# 使用 locust 进行负载测试
from locust import HttpUser, task, between

class SGLangUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={
                "text": "What is machine learning?",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 100
                }
            }
        )
```

## 高级特性

### 1. 自定义路由策略

```python
# 基于模型特性的路由
class ModelAwareRouter:
    def __init__(self, model_capabilities):
        self.capabilities = model_capabilities
    
    def route(self, request):
        # 根据请求类型选择最适合的模型
        if "code" in request.text.lower():
            return self.get_worker_with_model("codellama")
        elif "math" in request.text.lower():
            return self.get_worker_with_model("mathllama")
        else:
            return self.get_worker_with_model("general")
```

### 2. 请求优先级

```python
# 带优先级的请求
response = requests.post(
    "http://router:30000/generate",
    json={
        "text": "urgent query",
        "priority": "high",  # high, normal, low
        "sampling_params": {...}
    }
)
```

### 3. 流式响应

```python
# 流式接收生成结果
response = requests.post(
    "http://router:30000/generate",
    json={
        "text": "Write a long story",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8'))
        print(data['text'], end='', flush=True)
```

## 总结

SGLang Router 是一个功能强大的分布式推理路由系统，通过创新的缓存感知算法和灵活的负载均衡策略，能够显著提升大语言模型的推理性能。其主要优势包括：

1. **智能路由**：缓存感知算法最大化缓存利用率
2. **高性能**：Rust 实现保证低延迟和高吞吐
3. **易扩展**：支持动态扩缩容和多种部署模式
4. **生产就绪**：完善的监控、容错和运维支持

通过合理配置和优化，SGLang Router 可以帮助您构建高效、可靠的 LLM 推理服务，满足各种规模的生产需求。
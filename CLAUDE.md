# SGLang Testing Framework - 用户使用手册

## 快速开始（5分钟上手）

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang/sglang_test_framework

# 安装依赖
pip install -r requirements.txt

# 确保 SGLang 已安装
pip install sglang

# 对于路由测试，还需要安装 sglang-router
# 方法 1: 从 PyPI 安装
pip install sglang-router

# 方法 2: 从源码安装（推荐用于开发）
cd ../sgl-router
pip install -e .
cd ../sglang_test_framework
```

### 2. 运行第一个测试

```python
from sglang_test_framework import NodeTest, NodeConfig

# 配置测试
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    gpu_id=0,
    request_rate=10.0,  # 每秒10个请求
    num_prompts=100     # 总共100个请求
)

# 运行测试
test = NodeTest(config)
results = test.run()

# 查看结果
test.analyze_results(results)
```

就这么简单！测试会自动启动服务器、发送请求、收集指标并生成报告。

## 核心功能介绍

### 测试模式

#### 单节点测试 - 测试单个 GPU 的性能极限
```python
NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    gpu_id=0,
    max_running_requests=256,  # 最大并发请求数
    tp_size=1,                 # 张量并行大小
    mem_fraction_static=0.9    # 静态内存分配比例
)
```

#### 多节点路由测试 - 测试多 GPU 负载均衡
```python
# 注意：路由测试需要先安装 sglang-router（见安装说明）
RoutingConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    num_gpus=4,
    routing_policy="cache_aware",  # 路由策略
    request_rate=40.0
)
```

### 关键指标说明

- **Server Latency**: 请求在服务器上的处理时间
- **Total Latency**: 从请求到达到完成的总时间（包括排队）
- **TTFT**: 生成第一个 token 的时间
- **Queue Time**: 请求在队列中的等待时间

## 常见测试场景

### 场景 1: 寻找最佳并发数

```python
# 测试不同的 max_running_requests 值
for mrs in [32, 64, 128, 256, 512]:
    config = NodeConfig(
        model_path="meta-llama/Llama-2-7b-hf",
        max_running_requests=mrs,
        request_rate=20.0,
        num_prompts=1000
    )
    
    test = NodeTest(config)
    results = test.run()
    
    print(f"MRS={mrs}: "
          f"Throughput={results['metrics']['request_throughput']:.2f} req/s, "
          f"P99 Latency={results['metrics']['p99_server_latency']:.1f} ms")
```

### 场景 2: 压力测试

```python
# 逐步增加负载，找到系统极限
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    request_rate=float('inf'),  # 最大速率发送
    num_prompts=5000
)

test = NodeTest(config)
results = test.run()

# 查看系统在极限负载下的表现
print(f"最大吞吐量: {results['metrics']['request_throughput']:.2f} req/s")
print(f"错误率: {(1 - results['success_rate']) * 100:.1f}%")
```

### 场景 3: 路由策略对比

```python
# 测试不同路由策略的效果
for policy in ["cache_aware", "round_robin", "shortest_queue"]:
    config = RoutingConfig(
        model_path="meta-llama/Llama-2-7b-hf",
        num_gpus=4,
        routing_policy=policy,
        request_rate=50.0,
        num_prompts=2000
    )
    
    test = RoutingTest(config)
    results = test.run()
    test.visualize_results(results)
```

### 场景 4: 长短请求混合测试

```python
# 使用自定义数据集测试不同长度请求的处理
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    dataset_name="random",
    random_input_len=1024,      # 平均输入长度
    random_output_len=256,      # 平均输出长度
    random_range_ratio=0.8,     # 长度变化范围 ±80%
    request_rate=15.0
)
```

## 配置参数详解

### NodeConfig - 单节点测试配置

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| model_path | 必需 | 模型路径 |
| gpu_id | 0 | 使用的 GPU ID |
| max_running_requests | 256 | 最大并发请求数 |
| tp_size | 1 | 张量并行大小 |
| mem_fraction_static | 0.9 | 静态内存分配比例 |
| request_rate | inf | 请求发送速率（req/s） |
| num_prompts | 1000 | 测试请求总数 |
| dataset_name | "sharegpt" | 数据集类型 |

### RoutingConfig - 路由测试配置

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| num_gpus | 4 | GPU 数量 |
| routing_policy | "cache_aware" | 路由策略 |
| base_port | 30001 | 起始端口号 |
| collect_per_node_metrics | True | 是否收集每个节点的指标 |

### 路由策略说明

- **cache_aware**: 优先路由到有缓存的节点（SGLang Router 默认）
- **round_robin**: 轮询分配
- **random**: 随机分配
- **shortest_queue**: 选择队列最短的节点

## 结果分析指南

### 查看测试报告

测试完成后会自动生成：

1. **控制台摘要**：关键指标的文本报告
2. **CSV 文件**：每个请求的详细数据
3. **JSON 文件**：完整的测试结果和配置
4. **可视化图表**：性能分析图表（路由测试）

### 关键指标解读

```
📊 Performance Metrics:
  Success Rate: 99.8%              # 成功率应 > 99%
  Request Throughput: 45.2 req/s   # 实际处理能力
  Token Throughput:
    Input: 23,040 tok/s            # 输入 token 处理速度
    Output: 5,760 tok/s            # 输出 token 生成速度

⏱ Latency Metrics:
  Server Latency (ms):
    Mean: 245.1                    # 平均处理时间
    P95: 489.2                     # 95% 请求的延迟上限
    P99: 612.8                     # 99% 请求的延迟上限

  Queue Time (ms):
    Mean: 67.4                     # 平均排队时间
    P95: 145.2                     # 队列延迟应该较低
```

### 性能瓶颈识别

1. **高队列时间**：增加 `max_running_requests`
2. **高服务器延迟**：减少 `max_running_requests` 或使用更大的 GPU
3. **内存不足**：降低 `mem_fraction_static` 或减少并发数
4. **负载不均**：检查路由策略，考虑使用 `shortest_queue`

## 高级功能

### 使用自定义数据集

```python
# 方法 1: 使用 ShareGPT 数据集
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    dataset_name="sharegpt",
    dataset_path="/path/to/sharegpt.json"  # 可选，会自动下载
)

# 方法 2: 使用随机生成的数据
config = NodeConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    dataset_name="random",
    random_input_len=512,
    random_output_len=128
)

# 方法 3: 使用自定义数据集
# 创建符合格式的 JSON 文件：
# [{"prompt": "...", "completion": "...", "id": "..."}, ...]
```

### 导出和分析数据

```python
# 测试结果会自动导出，也可以手动处理
results = test.run()

# 获取 CSV 路径进行自定义分析
csv_path = results['exported_files']['csv']

# 使用 pandas 分析
import pandas as pd
df = pd.read_csv(csv_path)

# 计算自定义指标
print(f"中位数延迟: {df['server_latency'].median() * 1000:.1f} ms")
print(f"吞吐量标准差: {df.groupby(pd.cut(df['arrival_time'], bins=10))['req_id'].count().std():.2f}")
```

### 节点故障模拟（路由测试）

```python
config = RoutingConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    num_gpus=4,
    enable_node_failures=True,
    failure_schedule={
        300: [1, 2],  # 300秒时节点 1、2 故障
        600: []       # 600秒时恢复所有节点
    }
)
```

## 性能优化建议

### 高吞吐量配置

```yaml
适用场景: 批量推理、离线处理
配置建议:
  max_running_requests: 512
  mem_fraction_static: 0.85
  chunked_prefill_size: 8192
  enable_torch_compile: true
预期效果: 吞吐量提升 50-100%，延迟增加 20-30%
```

### 低延迟配置

```yaml
适用场景: 实时服务、交互式应用
配置建议:
  max_running_requests: 32
  mem_fraction_static: 0.9
  schedule_conservativeness: 1.2
预期效果: P99 延迟降低 30-50%，吞吐量降低 20-30%
```

### 内存优化配置

```yaml
适用场景: GPU 内存受限
配置建议:
  max_running_requests: 128
  mem_fraction_static: 0.7
  quantization: "fp8"
预期效果: 内存使用降低 30-40%，轻微性能损失
```

## 故障排查

### 常见问题

#### 1. 服务器启动失败
```
错误: "Address already in use"
解决: 修改 port 参数或检查端口占用
```

#### 2. GPU 内存不足
```
错误: "CUDA out of memory"
解决: 
- 减少 max_running_requests
- 降低 mem_fraction_static
- 使用量化（quantization="fp8"）
```

#### 3. 请求超时
```
错误: "Request timeout"
解决:
- 检查模型加载是否成功
- 增加 warmup_requests 数量
- 确认 GPU 没有被其他进程占用
```

#### 4. 结果不一致
```
问题: 多次测试结果差异很大
解决:
- 增加 num_prompts（建议 > 1000）
- 使用固定的 seed
- 确保系统负载稳定
```

## 最佳实践

### 测试前准备
1. 确保 GPU 空闲：`nvidia-smi`
2. 关闭其他 GPU 进程
3. 预留足够的测试时间（建议每次测试 > 5 分钟）

### 测试执行
1. 先用小规模测试验证配置
2. 逐步增加负载找到最佳工作点
3. 重复测试 3 次以上确保结果稳定

### 结果验证
1. 检查成功率是否 > 99%
2. 观察延迟分布是否有异常峰值
3. 确认吞吐量是否随时间稳定

## API 快速参考

### 基础用法
```python
from sglang_test_framework import NodeTest, NodeConfig

config = NodeConfig(model_path="...", gpu_id=0)
test = NodeTest(config)
results = test.run()
```

### 路由测试
```python
from sglang_test_framework import RoutingTest, RoutingConfig

config = RoutingConfig(model_path="...", num_gpus=4)
test = RoutingTest(config)
results = test.run()
test.visualize_results(results)
```

### 结果分析
```python
# 自动生成的文件
results['exported_files']['csv']      # 详细数据
results['exported_files']['json']     # 完整结果
results['metrics']                    # 聚合指标
```

## 更多资源

- **技术设计文档**：`sglang_test_framework/说明.md`
- **SGLang 官方文档**：`.DUCUMENT/` 目录
- **问题反馈**：https://github.com/sgl-project/sglang/issues

## 版本兼容性

- SGLang >= 0.3.0
- SGLang Router >= 0.1.0
- Python >= 3.8
- CUDA >= 11.0

## SGLang Router 测试指南

### 1. 启动 SGLang 服务器

```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 60005 \
--base-gpu-id 2 \
--enable-metrics \  # 添加这个参数,collect_metrics 只在 self.enable_metrics 为 True 时调用
--log-level debug # 这个可以在server的log里面检查相应的全部的log

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 60006 \
--base-gpu-id 3 \
--enable-metrics
```

### 2. 启动路由器（启用请求追踪）

```bash
bash /nas/ganluo/sglang/bash_start_router.sh
```
停止的方法是:
```bash
ctrl+Z
kill %1
clear
```

### 3. 测试router
```bash
bash /nas/ganluo/sglang/bash_send_req.sh
```

### 4. 故障排除
当遇到29000端口被占据的情况, 是因为之前的router端口没有释放, 需要这样做
```bash
lsof -i :29000
```
然后`kill -9`这个pid.

### 5. Router测试参数解读
目前记录在csv中的参数有（按照新的顺序）:

**基础信息：**
req_id,success,error,host,has_generated_text

**长度相关字段：**
input_length,expected_output_length,actual_prompt_tokens,actual_output_tokens,actual_total_tokens,output_tokens_from_trace,decode_length

**时间戳（按时间顺序）：**
arrival_time,router_send_time,server_created_time,queue_time_start,queue_time_end,server_first_token_time,finish_time

**延迟指标：**
server_latency,total_latency,ttft,queue_time_in_router,queue_time_in_server,tokenize_time,pure_queue_time

#### 参数含义详解：

**基础信息：**
- `req_id`: 请求的唯一标识符
- `success`: 请求是否成功（True/False）
- `error`: 错误信息（如果有）
- `host`: 实际处理请求的服务器地址
- `has_generated_text`: 是否有生成的文本（用于调试）

**长度相关字段说明（重要）：**
- `input_length`: 输入文本的**单词数**（由 `prompt.split()` 计算，以空格分割）
- `expected_output_length`: 请求时设置的期望输出 **token 数**（max_new_tokens 参数）
- `actual_prompt_tokens`: 输入文本经过 tokenizer 处理后的实际 **token 数**（从服务器返回）
- `actual_output_tokens`: 实际生成的 **token 数**（从服务器返回）
- `actual_total_tokens`: 实际总 **token 数**（输入+输出，从服务器返回）
- `output_tokens_from_trace`: 从路由器追踪信息获取的输出 **token 数**（通常为 0）
- `decode_length`: 综合字段，表示生成的 **token 数**（优先使用 actual_output_tokens）

**注意事项：**
1. **单词数 vs Token 数**：
   - `input_length` 是按空格分割的单词数，不是 token 数
   - 对于英文：一个单词可能对应 1-3 个 tokens
   - 对于中文：由于中文通常不使用空格分割，`input_length` 可能不准确
   - 因此 `actual_prompt_tokens` 才是真实的输入 token 数量

2. **准确的 Token 计数**：
   - **必须使用** `actual_prompt_tokens` 而不是 `input_length` 来计算输入 token 吞吐量
   - **必须使用** `actual_output_tokens` 而不是 `decode_length` 来计算输出 token 吞吐量
   
3. **示例对比**：
   - 英文文本 "Hello world": `input_length`=2（2个单词），`actual_prompt_tokens` 可能是 2-3
   - 中文文本 "你好世界": `input_length`=1（没有空格），`actual_prompt_tokens` 可能是 4-6

**时间戳（按时间顺序）：**
- `arrival_time`: 请求到达router的时刻（从测试开始计时）
- `router_send_time`: router将请求转发到server的时刻（原名 to_server_time）
- `server_created_time`: 请求在server端被tokenizer_manager接收的时刻
  - 根据`python/sglang/srt/entrypoints/http_server.py`, server_created_time 是请求被 tokenizer_manager 接收的时刻，而不是 HTTP 请求刚到达服务器的时刻
  - 但是请求到达服务器后基本上是立即转发到 tokenizer_manager 的，中间没有排队
- `queue_time_start`: 请求进入scheduler waiting_queue的时刻
- `queue_time_end`: 请求从waiting_queue取出准备处理的时刻
- `server_first_token_time`: server生成第一个token的时刻（prefill完成+第一次decode）
- `finish_time`: 请求完成的时刻

**延迟指标：**
- `server_latency`: 服务器中的总时间 = `finish_time - router_send_time`
- `total_latency`: 总延迟 = `finish_time - arrival_time`
- `ttft` (Time To First Token): 第一个token生成时间 = `server_first_token_time - arrival_time`（从客户端发送时刻算起）
- `queue_time_in_router`: router中的排队时间 = `router_send_time - arrival_time`（通常很小）
- `queue_time_in_server`: server端总排队时间 = `server_first_token_time - server_created_time`
- `tokenize_time`: server端tokenize的时间 = `queue_time_start - server_created_time`
- `pure_queue_time`: scheduler纯排队时间 = `queue_time_end - queue_time_start`（不含tokenize时间）

**各阶段耗时计算：**
1. **Router转发延迟**: `router_send_time - arrival_time` （通常<0.1ms）
2. **Tokenize时间**: `queue_time_start - server_created_time`
3. **Scheduler排队时间**: `queue_time_end - queue_time_start`（即 pure_queue_time）
4. **Prefill时间**: `server_first_token_time - queue_time_end`
5. **网络传输时间**: `server_created_time - router_send_time`

### 6. 关于 decode_length 和 token 计数说明（已修复）

#### 问题背景
之前测试中 `decode_length` 显示为 0 是因为依赖路由器追踪信息，但该信息可能不完整。

#### 修复方案（已实施）
现在测试代码已经更新，采用与 SGLang 官方 bench_serving.py 相同的方法获取实际生成的 token 数：

1. **从 meta_info 提取**（SGLang native API）：
   - `completion_tokens`: 实际生成的 token 数
   - `prompt_tokens`: 实际输入的 token 数
   - `total_tokens`: 总 token 数

2. **从 usage 字段提取**（OpenAI-compatible API）：
   - 作为 meta_info 的备选方案

3. **CSV 中记录的所有 token 相关字段**：
   - `expected_output_length`: 请求时设置的期望输出长度
   - `actual_output_tokens`: 从 meta_info/usage 获取的实际生成 token 数
   - `actual_prompt_tokens`: 从 meta_info/usage 获取的实际输入 token 数
   - `actual_total_tokens`: 从 meta_info/usage 获取的总 token 数
   - `output_tokens_from_trace`: 从路由器追踪信息获取的输出 token 数
   - `decode_length`: 综合字段（优先使用 actual_output_tokens，保持向后兼容）
   - `has_generated_text`: 是否有生成的文本（用于调试）

#### 使用说明
- 分析时优先使用 `actual_output_tokens` 字段，这是最准确的实际生成长度
- `decode_length` 保留用于向后兼容
- 如果 `actual_output_tokens` 为空，可能是服务器版本较旧或未启用相关功能

---

*本框架持续更新中，欢迎贡献代码和反馈问题！*
# demand_3_implement

**INSTRUCTION: 指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**

**INSTRUCTION: 对于可以直接复用的sglang的代码和模块的, 可以直接copy过来, 不需要自己实现. 但是需要在.py文件最前面注释来源.  指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**

## 1. 对于记录metric的再次说明和验证.

in `CLAUDE.md`, 你提到: 
```
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
```

这里需要说明的是, 对于一次实验, 我们需要记录的
+ level_1: per request 的 metrics
  + arrival time, to_server time, finish time
  + 由此计算出的, Server Latency, Total Latency, TTFT for each request
  + 保存的是csv, (req_id, input_length, decode_length, arrival time, to_server time, finish time, Server Latency, Total Latency, TTFT)
+ level_2: per experiment 的 metrics
  + 根据前面计算的evel_1: per request 的 metrics
  + 我们可以计算 avg_server_latency, avg_total_latency, avg_TTFT, 
    + throughput
    + concurrency
+ 重要的一点的, 你需要研究一下, 就是在`python/sglang/bench_serving_new.py`, `python/sglang/bench_serving.py`里面
  + 现在的API有没有可能记录，单次实验中随着时间推移变化的concurrency, running reqeusts?
  + 有没有现成的实现?
  + 以及是不是可以直接用 (req_id, input_length, decode_length, arrival time, to_server time, finish time, Server Latency, Total Latency, TTFT)近似算出来?

进行分析和修改之后, 需要更新`sglang_test_framework/说明.md` and `CLAUDE.md`

**INSTRUCTION: 指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**


## 2. implementation

根据
```
sglang_test_framework/
├── config/               # 配置模块
│   ├── base.py          # 基础配置类
│   ├── node_config.py   # 节点测试配置
│   └── routing_config.py # 路由测试配置
├── core/                # 核心功能模块
│   ├── server_manager.py    # 服务器管理
│   ├── request_generator.py # 请求生成
│   ├── metrics_collector.py # 指标收集
│   └── result_manager.py    # 结果管理
├── strategies/          # 策略实现（待开发）
│   ├── batching/       # 批处理策略
│   └── routing/        # 路由策略
├── tests/              # 测试运行器（待开发）
├── utils/              # 工具函数（待开发）
└── requirements.txt    # 依赖列表
```
config and core部分已经完成了开发

剩余的部分需要按照 `CLAUDE.md` and `sglang_test_framework/说明.md`进行implementation.

**INSTRUCTION: 对于可以直接复用的sglang的代码和模块的, 可以直接copy过来, 不需要自己实现. 但是需要在.py文件最前面注释来源.  指示中提到的.md文件和.py文件, 以及提到的文件夹下面的所有文件, 都必须阅读一遍**

更改代码之后需要更新文档
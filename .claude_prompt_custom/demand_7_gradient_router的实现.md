**INSTRUCTION: 用中文，思考,阅读文件和文档要完全阅读，不要使用compact和总结来代替阅读(除了log和csv这样的记录文件), 改动要求进行最小的改动,不要造成巨大的影响**
**INSTRUCTION: 用中文，思考,阅读文件和文档要完全阅读，不要使用compact和总结来代替阅读(除了log和csv这样的记录文件), 改动要求进行最小的改动,不要造成巨大的影响**
**INSTRUCTION: 用中文，思考,阅读文件和文档要完全阅读，不要使用compact和总结来代替阅读(除了log和csv这样的记录文件), 改动要求进行最小的改动,不要造成巨大的影响**

### 1. 总结已经有的router和具体结构:

#### 1.1 Base Class/Trait - LoadBalancingPolicy
   - **文件位置**: `sgl-router/src/policies/mod.rs`
   - **核心 trait**: `LoadBalancingPolicy`
   - **主要方法**:
     - `select_worker(&self, workers: &[Box<dyn Worker>], request_text: Option<&str>) -> Option<usize>` - 选择单个 worker
     - `select_worker_pair(&self, prefill_workers: &[Box<dyn Worker>], decode_workers: &[Box<dyn Worker>], request_text: Option<&str>) -> Option<(usize, usize)>` - 为 PD 模式选择 worker 对
     - `on_request_complete(&self, worker_url: &str, success: bool)` - 请求完成回调
     - `update_loads(&self, loads: &HashMap<String, isize>)` - 更新负载信息
     - `reset(&self)` - 重置内部状态
     - `name(&self) -> &'static str` - 获取策略名称
     - `as_any(&self) -> &dyn std::any::Any` - 用于向下转型

#### 1.2 Random Policy（随机路由）
   - **文件位置**: `sgl-router/src/policies/random.rs`
   - **结构体**: `RandomPolicy`
   - **特点**: 
     - 无状态实现
     - 从健康的 workers 中随机选择
     - 使用 `rand::thread_rng()` 生成随机数
   - **实现细节**: 先获取所有健康 worker 的索引，然后随机选择一个

#### 1.3 Round Robin Policy（轮询路由）
   - **文件位置**: `sgl-router/src/policies/round_robin.rs`
   - **结构体**: `RoundRobinPolicy`
   - **特点**:
     - 使用 `AtomicUsize` 计数器保证线程安全
     - 按顺序循环选择健康的 workers
     - 支持 `reset()` 方法重置计数器
   - **实现细节**: 使用 `fetch_add` 原子操作递增计数器，对健康 worker 数量取模

#### 1.4 Power of Two Policy（二选一路由）
   - **文件位置**: `sgl-router/src/policies/power_of_two.rs`
   - **结构体**: `PowerOfTwoPolicy`
   - **特点**:
     - 随机选择两个 workers，选择负载较低的
     - 支持缓存外部负载监控数据（通过 `cached_loads: RwLock<HashMap<String, isize>>`）
     - 需要 PD 模式支持
   - **实现细节**: 
     - 先尝试使用缓存的负载信息
     - 如果没有缓存，使用 worker 的本地负载计数器
     - 记录选择决策用于调试

#### 1.5 Cache Aware Policy（缓存感知路由）
   - **文件位置**: `sgl-router/src/policies/cache_aware.rs`
   - **结构体**: `CacheAwarePolicy`
   - **配置参数** (`CacheAwareConfig`):
     - `cache_threshold`: 缓存命中阈值（0.0-1.0）
     - `balance_abs_threshold`: 负载平衡绝对差值阈值
     - `balance_rel_threshold`: 负载平衡相对比率阈值
     - `eviction_interval_secs`: 缓存驱逐间隔
     - `max_tree_size`: 最大树节点数
   - **特点**:
     - 双模式运行：负载不均时使用最短队列，负载均衡时使用缓存感知
     - 使用近似 radix tree（`Arc<Mutex<Tree>>`）维护请求历史
     - 后台线程定期执行 LRU 缓存驱逐
   - **实现细节**:
     - 负载不均判断：`(max - min) > abs_threshold && max > min * rel_threshold`
     - 缓存命中时路由到匹配度最高的 worker
     - 缓存未命中时路由到树最小的 worker（可用缓存空间最大）

#### 1.6 策略工厂（PolicyFactory）
   - **文件位置**: `sgl-router/src/policies/factory.rs`
   - **主要方法**:
     - `create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy>` - 从配置创建
     - `create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>>` - 按名称创建
   - **支持的名称**: "random", "round_robin", "power_of_two", "cache_aware"

#### 1.7 配置类型定义
   - **文件位置**: `sgl-router/src/config/types.rs`
   - **枚举**: `PolicyConfig`
   - **变体**:
     - `Random` - 无参数
     - `RoundRobin` - 无参数
     - `PowerOfTwo { load_check_interval_secs: u64 }`
     - `CacheAware { cache_threshold, balance_abs_threshold, balance_rel_threshold, eviction_interval_secs, max_tree_size }`

#### 1.8 Router 主体实现
   - **文件位置**: `sgl-router/src/routers/router.rs`
   - **结构体**: `Router`
   - **主要功能**:
     - 使用注入的 `LoadBalancingPolicy` 进行路由决策
     - 健康检查和故障转移
     - 请求重试机制（最多 3 次单 worker 重试，总共 6 次重试）
     - 请求追踪支持（可选）
     - 为 PowerOfTwo 策略提供负载监控
   - **关键方法**:
     - `route_typed_request()` - 路由类型化请求
     - `add_worker()` / `remove_worker()` - 动态管理 workers
     - `send_request()` / `send_typed_request()` - 发送请求到选定的 worker

#### 1.9 Python 绑定层
   - **文件位置**: `start_router.py`
   - **支持的策略**: "cache_aware", "round_robin", "random", "power_of_two"
   - **策略映射** (line 110-115):
     ```python
     policy_map = {
         "cache_aware": PolicyType.CacheAware,
         "round_robin": PolicyType.RoundRobin,
         "random": PolicyType.Random,
         "power_of_two": PolicyType.PowerOfTwo,
     }
     ```

1. 现在我想要实现一个新的router, 具体来说，这个policy的核心思想是
   1. 根据`/Users/luogan/Code/sglang/router测试方法.md`, 我们知道我们可以跟踪历史上处理的request的相关参数
   2. 如果我们通过某种方式修改成流式的获取，那么router可以实时获取所有已经完成的request的相关参数
   3. 负载均衡的主要目标是为了最大化总的throughput, 最大化每个host上的throughput, 最小化每个host上的total_latency
   4. 因此我们想分析如果一个req被route到这个host, 这个host上处理的req的相关metric边际变化效应
      1. 我暂时想到了一个比较简单的近似方法, 例如得到了近似，那么可以选择是，例如route一个req到这个host的decode_throughput的边际增长率最大的host; 或者route一个req到这个host的total_latency的边际增长率最小的host？
2. 具体的算法实现的思想是:
   1. 边际效用路由策略 (Marginal Utility Routing Policy)
   2. 抛弃基于静态快照（如当前队列长度）的路由，采用基于性能趋势预测的动态路由。通过计算每个Worker（Host）性能指标的“加速度”（即边际效益的差商近似），选择那个改善趋势最明显的Worker，从而在宏观上最大化整个集群的吞吐量并控制延迟。
   3. 核心数据结构设计
      1. RequestMetric
         1. 就是`/Users/luogan/Code/sglang/router测试方法.md`可以得到的各种metrics. 
         2. 我们主要关注里面的
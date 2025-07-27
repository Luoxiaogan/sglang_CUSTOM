# SGLang Router请求追踪方案

## 背景
当前SGLang router使用Prometheus记录聚合指标，但无法追踪单个请求的路由决策。

## 方案设计

### 1. 添加请求追踪缓冲区

在router中维护一个固定大小的循环缓冲区：

```rust
// 在 src/routers/router.rs 中添加
use std::collections::VecDeque;
use std::sync::RwLock;

#[derive(Clone, Debug)]
struct RequestTrace {
    request_id: String,
    worker_url: String,
    policy: String,
    timestamp: SystemTime,
    // 可选：添加更多信息
    input_tokens: Option<usize>,
    cache_hit: Option<bool>,
}

pub struct Router {
    // ... 现有字段
    
    // 新增：请求追踪缓冲区
    request_traces: Arc<RwLock<VecDeque<RequestTrace>>>,
    max_traces: usize,  // 默认 10000
}

impl Router {
    fn record_request_trace(&self, request_id: String, worker_url: String, policy: String) {
        let trace = RequestTrace {
            request_id,
            worker_url,
            policy,
            timestamp: SystemTime::now(),
            input_tokens: None,
            cache_hit: None,
        };
        
        let mut traces = self.request_traces.write().unwrap();
        if traces.len() >= self.max_traces {
            traces.pop_front();  // 移除最旧的
        }
        traces.push_back(trace);
    }
}
```

### 2. 添加查询API

```rust
// 新增路由
.route("/traces", web::get().to(get_traces))
.route("/traces/{request_id}", web::get().to(get_trace_by_id))

async fn get_traces(router: web::Data<Arc<Router>>) -> impl Responder {
    let traces = router.request_traces.read().unwrap();
    let traces_vec: Vec<_> = traces.iter().cloned().collect();
    HttpResponse::Ok().json(&traces_vec)
}

async fn get_trace_by_id(
    router: web::Data<Arc<Router>>,
    path: web::Path<String>,
) -> impl Responder {
    let request_id = path.into_inner();
    let traces = router.request_traces.read().unwrap();
    
    if let Some(trace) = traces.iter().find(|t| t.request_id == request_id) {
        HttpResponse::Ok().json(trace)
    } else {
        HttpResponse::NotFound().body("Request not found")
    }
}
```

### 3. 在路由决策时记录

```rust
// 在 select_worker 方法中
async fn select_generate_worker_from_text(&self, text: &str) -> Option<String> {
    let worker = self.policy.select_worker(text).await?;
    
    // 记录追踪信息
    if let Some(request_id) = extract_request_id_from_context() {
        self.record_request_trace(
            request_id,
            worker.clone(),
            self.policy.name(),
        );
    }
    
    Some(worker)
}
```

### 4. 客户端集成

在Python测试框架中：

```python
async def get_request_trace(router_url: str, request_id: str) -> Optional[dict]:
    """获取特定请求的路由信息"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{router_url}/traces/{request_id}") as resp:
            if resp.status == 200:
                return await resp.json()
    return None

async def get_all_traces(router_url: str) -> List[dict]:
    """获取所有请求追踪信息"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{router_url}/traces") as resp:
            if resp.status == 200:
                return await resp.json()
    return []
```

## 优势

1. **精确追踪**：可以知道每个请求的确切路由
2. **低开销**：只保留最近N个请求，内存占用可控
3. **易于调试**：可以实时查看路由决策
4. **向后兼容**：不影响现有功能

## 配置选项

```rust
// 通过命令行参数配置
--enable-request-tracing  // 启用请求追踪
--max-trace-entries 10000 // 最大追踪条目数
--trace-ttl 3600         // 追踪信息保留时间（秒）
```

## 使用示例

```bash
# 查看最近的所有请求路由
curl http://localhost:30000/traces

# 查看特定请求的路由信息
curl http://localhost:30000/traces/req_12345

# 响应示例
{
  "request_id": "req_12345",
  "worker_url": "http://0.0.0.0:30001",
  "policy": "cache_aware",
  "timestamp": "2024-01-25T10:30:45Z",
  "cache_hit": true
}
```

## 实现步骤

1. 在`Router`结构体中添加追踪缓冲区
2. 实现`record_request_trace`方法
3. 在路由决策点调用记录方法
4. 添加HTTP API端点
5. 更新Python客户端支持
6. 添加文档和测试

这样就可以在不改变Prometheus聚合统计的基础上，添加单请求追踪功能。
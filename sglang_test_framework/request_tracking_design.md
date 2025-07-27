# SGLang Router 请求追踪原生支持设计

## 核心需求
支持查询每个请求被路由到哪个具体的node_id，包括：
- 实时查询：请求进行中就能查询
- 历史查询：请求完成后仍可查询
- 批量查询：一次查询多个请求的路由信息

## 设计方案

### 1. 数据结构

```rust
// 在 sgl-router/src/routers/router.rs 中添加

use std::collections::HashMap;
use std::sync::RwLock;
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTrace {
    pub request_id: String,
    pub node_id: String,      // worker标识，如 "worker_0" 或 "gpu_0"
    pub worker_url: String,    // 实际的worker URL
    pub timestamp: DateTime<Utc>,
    pub routing_policy: String,
    pub status: RequestStatus,
    // 可选的额外信息
    pub cache_hit: Option<bool>,
    pub queue_position: Option<usize>,
    pub input_tokens: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RequestStatus {
    Routed,      // 已路由
    Processing,  // 处理中
    Completed,   // 已完成
    Failed,      // 失败
}

pub struct Router {
    // ... 现有字段
    
    // 新增：请求追踪存储
    request_traces: Arc<RwLock<HashMap<String, RequestTrace>>>,
    max_trace_entries: usize,  // 最大存储条目数，防止内存溢出
    trace_ttl: Duration,       // 追踪信息保留时间
}
```

### 2. API设计

```rust
// 新增API端点

// 查询单个请求
// GET /v1/traces/{request_id}
async fn get_request_trace(
    path: web::Path<String>,
    router: web::Data<Arc<Router>>,
) -> impl Responder {
    let request_id = path.into_inner();
    let traces = router.request_traces.read().unwrap();
    
    match traces.get(&request_id) {
        Some(trace) => HttpResponse::Ok().json(trace),
        None => HttpResponse::NotFound().json(json!({
            "error": "Request not found",
            "request_id": request_id
        }))
    }
}

// 批量查询
// POST /v1/traces/batch
// Body: {"request_ids": ["req1", "req2", "req3"]}
async fn batch_get_traces(
    body: web::Json<BatchTraceRequest>,
    router: web::Data<Arc<Router>>,
) -> impl Responder {
    let traces = router.request_traces.read().unwrap();
    let mut results = HashMap::new();
    
    for request_id in &body.request_ids {
        if let Some(trace) = traces.get(request_id) {
            results.insert(request_id.clone(), trace.clone());
        }
    }
    
    HttpResponse::Ok().json(results)
}

// 查询最近N条
// GET /v1/traces?limit=100&node_id=gpu_0
async fn list_recent_traces(
    query: web::Query<ListTracesQuery>,
    router: web::Data<Arc<Router>>,
) -> impl Responder {
    let traces = router.request_traces.read().unwrap();
    
    let mut result: Vec<RequestTrace> = traces
        .values()
        .filter(|t| {
            query.node_id.as_ref().map_or(true, |nid| &t.node_id == nid)
        })
        .cloned()
        .collect();
    
    // 按时间排序
    result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    
    // 限制返回数量
    let limit = query.limit.unwrap_or(100).min(1000);
    result.truncate(limit);
    
    HttpResponse::Ok().json(result)
}

// 获取统计信息
// GET /v1/traces/stats
async fn get_trace_stats(
    router: web::Data<Arc<Router>>,
) -> impl Responder {
    let traces = router.request_traces.read().unwrap();
    
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    let mut status_counts: HashMap<String, usize> = HashMap::new();
    
    for trace in traces.values() {
        *node_counts.entry(trace.node_id.clone()).or_insert(0) += 1;
        *status_counts.entry(format!("{:?}", trace.status)).or_insert(0) += 1;
    }
    
    HttpResponse::Ok().json(json!({
        "total_traces": traces.len(),
        "node_distribution": node_counts,
        "status_distribution": status_counts,
        "oldest_trace": traces.values().min_by_key(|t| t.timestamp).map(|t| t.timestamp),
        "newest_trace": traces.values().max_by_key(|t| t.timestamp).map(|t| t.timestamp),
    }))
}
```

### 3. 集成到路由逻辑

```rust
impl Router {
    // 在路由决策时记录
    async fn route_request(&self, request: &GenerateRequest) -> Result<String, Error> {
        let request_id = request.request_id.clone()
            .unwrap_or_else(|| format!("req_{}", Uuid::new_v4()));
        
        // 选择worker
        let worker_url = self.policy.select_worker(&request.text).await?;
        let node_id = self.worker_url_to_node_id(&worker_url);
        
        // 记录追踪信息
        let trace = RequestTrace {
            request_id: request_id.clone(),
            node_id: node_id.clone(),
            worker_url: worker_url.clone(),
            timestamp: Utc::now(),
            routing_policy: self.policy.name(),
            status: RequestStatus::Routed,
            cache_hit: None,  // 可以从policy获取
            queue_position: None,
            input_tokens: Some(request.text.len() / 4), // 估算
        };
        
        // 存储追踪信息
        {
            let mut traces = self.request_traces.write().unwrap();
            
            // 清理过期条目
            if traces.len() >= self.max_trace_entries {
                self.cleanup_old_traces(&mut traces);
            }
            
            traces.insert(request_id.clone(), trace);
        }
        
        // 更新Prometheus指标
        RouterMetrics::record_processed_request(&worker_url);
        
        Ok(worker_url)
    }
    
    // 清理过期追踪
    fn cleanup_old_traces(&self, traces: &mut HashMap<String, RequestTrace>) {
        let cutoff = Utc::now() - self.trace_ttl;
        traces.retain(|_, trace| trace.timestamp > cutoff);
        
        // 如果还是太多，删除最旧的
        if traces.len() > self.max_trace_entries {
            let mut traces_vec: Vec<_> = traces.iter().collect();
            traces_vec.sort_by_key(|(_, t)| t.timestamp);
            
            let to_remove = traces.len() - self.max_trace_entries;
            for (id, _) in traces_vec.iter().take(to_remove) {
                traces.remove(*id);
            }
        }
    }
}
```

### 4. Python客户端支持

```python
# 在 sglang_test_framework/core/request_generator.py 中添加

class RouterTraceClient:
    """Router追踪信息客户端"""
    
    def __init__(self, router_url: str):
        self.router_url = router_url
        self.trace_url = f"{router_url}/v1/traces"
    
    async def get_request_trace(self, request_id: str) -> Optional[Dict]:
        """查询单个请求的路由信息"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.trace_url}/{request_id}") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    
    async def batch_get_traces(self, request_ids: List[str]) -> Dict[str, Dict]:
        """批量查询请求路由信息"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.trace_url}/batch",
                json={"request_ids": request_ids}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
    
    async def list_recent_traces(
        self, 
        limit: int = 100, 
        node_id: Optional[str] = None
    ) -> List[Dict]:
        """查询最近的追踪记录"""
        params = {"limit": limit}
        if node_id:
            params["node_id"] = node_id
            
        async with aiohttp.ClientSession() as session:
            async with session.get(self.trace_url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
    
    async def get_trace_stats(self) -> Dict:
        """获取追踪统计信息"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.trace_url}/stats") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
```

### 5. 使用示例

```python
# 在测试中使用
async def test_with_request_tracking():
    router_url = "http://localhost:30000"
    trace_client = RouterTraceClient(router_url)
    
    # 发送请求时记录ID
    request_ids = []
    async with RequestSender() as sender:
        for i in range(100):
            request_id = f"test_req_{i}"
            request_ids.append(request_id)
            
            # 发送请求（需要支持自定义request_id）
            result = await sender.send_request(
                api_url=f"{router_url}/generate",
                request=request,
                request_id=request_id  # 传递自定义ID
            )
    
    # 批量查询路由信息
    traces = await trace_client.batch_get_traces(request_ids)
    
    # 统计每个节点的请求数
    node_counts = {}
    for req_id, trace in traces.items():
        node_id = trace["node_id"]
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
    
    print("Node distribution:")
    for node_id, count in sorted(node_counts.items()):
        print(f"  {node_id}: {count} requests")
    
    # 查询统计信息
    stats = await trace_client.get_trace_stats()
    print(f"\nTotal traces: {stats['total_traces']}")
    print(f"Node distribution: {stats['node_distribution']}")
```

## 优势

1. **完整性**：记录每个请求的完整路由信息
2. **可查询**：支持实时和历史查询
3. **高效**：使用内存存储，查询快速
4. **可扩展**：易于添加更多追踪信息
5. **标准化**：RESTful API，易于集成

## 配置选项

```bash
# Router启动参数
--enable-request-tracing      # 启用请求追踪
--max-trace-entries 100000    # 最大追踪条目数
--trace-ttl 3600             # 追踪信息保留时间（秒）
--trace-cleanup-interval 60   # 清理间隔（秒）
```

## 实现步骤

1. **Phase 1**：基础追踪功能
   - 添加数据结构
   - 实现基本的记录和查询API
   - 集成到路由逻辑

2. **Phase 2**：性能优化
   - 实现高效的清理机制
   - 添加内存使用限制
   - 优化查询性能

3. **Phase 3**：高级功能
   - 支持持久化存储（可选）
   - 添加WebSocket实时推送
   - 集成到监控面板

这个方案提供了完整的请求追踪能力，是最适合生产环境的解决方案。
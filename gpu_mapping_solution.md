# SGLang Router 端口标识与映射配置

## 设计理念
SGLang Router 使用端口号作为 worker 的主要标识符，这是一个深思熟虑的设计决策：

1. **稳定性**：端口号在部署配置中是固定的，不会因硬件变化而改变
2. **唯一性**：每个 worker 进程都有唯一的端口号
3. **可扩展性**：在多节点、多GPU环境中，端口号提供了清晰的标识

## 端口到节点ID的映射规则

Router 根据端口号自动生成节点ID：
```
节点ID = "gpu_" + (端口号 - 30001)
```

例如：
- 端口 30001 → gpu_0
- 端口 30005 → gpu_4
- 端口 30006 → gpu_5

## 配置端口-GPU映射

为了方便在CPU端程序中管理端口与实际GPU的对应关系，可以在启动路由器时配置映射字典：

### 1. 在路由器启动脚本中配置

编辑 `start_test_router.py`：

```python
# ====== 端口到GPU映射配置 ======
# 请根据您的实际部署情况修改此映射
PORT_GPU_MAPPING = {
    30005: "cuda:2",  # 端口 30005 对应 GPU 2
    30006: "cuda:3",  # 端口 30006 对应 GPU 3
    # 添加更多映射...
}
```

这个映射会自动保存到 `/tmp/sglang_port_gpu_mapping.json`，供其他程序使用。

### 2. 测试脚本自动读取映射

`test_request_tracking_v2.py` 会自动读取这个映射文件，并在显示结果时展示实际的GPU信息：

```
请求 1:
  请求ID: req_xxx
  节点: gpu_4 -> cuda:2
  Worker URL: http://localhost:30005
```

## 实际部署示例

当您启动 SGLang 服务器时：
```bash
# Worker 1
python -m sglang.launch_server \
  --port 30005 \
  --base-gpu-id 2  # 使用 CUDA device 2

# Worker 2  
python -m sglang.launch_server \
  --port 30006 \
  --base-gpu-id 3  # 使用 CUDA device 3
```

Router 会显示：
- gpu_4 (端口 30005) - 实际运行在 CUDA:2
- gpu_5 (端口 30006) - 实际运行在 CUDA:3

## 为什么这样设计？

1. **部署灵活性**：端口配置是部署的一部分，而GPU分配可能会变化
2. **负载均衡**：Router 只需要知道端口就能正确路由请求
3. **监控简化**：通过端口号可以快速定位具体的服务进程

## 使用建议

1. **文档化端口映射**：在部署文档中记录端口与实际GPU的对应关系
2. **使用有意义的端口范围**：例如 30000-30099 用于生产环境
3. **监控和日志**：在日志中同时记录端口号和GPU ID便于调试

## 其他使用方式

### 在您自己的程序中使用映射

```python
import json

# 读取映射文件
with open("/tmp/sglang_port_gpu_mapping.json", "r") as f:
    port_gpu_mapping = json.load(f)

# 使用映射
port = 30005
gpu = port_gpu_mapping.get(str(port), f"未知GPU(端口{port})")
print(f"端口 {port} 对应 {gpu}")
```

### 环境变量方式（可选）

您也可以通过环境变量传递映射：

```bash
export SGLANG_PORT_GPU_MAPPING='{"30005": "cuda:2", "30006": "cuda:3"}'
```

## 查看映射关系

通过请求追踪API可以看到完整的映射：
```bash
curl http://localhost:30009/v1/traces?limit=10
```

响应中会显示：
- `node_id`: 路由器分配的节点标识（如 gpu_4）
- `worker_url`: 完整的 worker URL（包含端口号）

配合映射文件，您可以轻松追踪每个请求实际运行在哪个GPU上。

## 优势

这种设计让系统在保持简单的同时具有良好的可扩展性：
- Router 核心逻辑不需要关心GPU细节
- 端口号作为稳定的标识符
- 通过外部映射文件灵活配置GPU对应关系
- 方便在多节点、多GPU环境中部署和管理
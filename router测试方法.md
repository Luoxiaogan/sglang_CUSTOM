# SGLang Router 测试

### 1. 启动 SGLang 服务器

```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30005 \
--base-gpu-id 2 \
--enable-metrics   # 添加这个参数,collect_metrics 只在 self.enable_metrics 为 True 时调用

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30006 \
--base-gpu-id 3 \
--enable-metrics
```

### 2. 启动路由器（启用请求追踪）

```bash
bash /nas/ganluo/sglang_CUSTOM/bash_start_router.sh
```
停止的方法是:
```bash
ctrl+Z
kill %1
clear
```

### 3. 测试router
```bash
python /nas/ganluo/sglang_CUSTOM/bash_start_router.sh
```
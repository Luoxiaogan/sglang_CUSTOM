# Queue Timestamp Testing Guide

## 新增的时间戳功能

本次更新添加了精确的排队时间测量功能，新增了以下时间戳：

1. **queue_time_start**: 请求进入scheduler waiting_queue的时间
2. **queue_time_end**: 请求从waiting_queue取出的时间
3. **pure_queue_time**: 纯排队时间 = queue_time_end - queue_time_start

## 测试步骤

### 1. 启动服务器（必须启用metrics）

```bash
# 终端 1 - GPU 2
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30005 \
--base-gpu-id 2 \
--enable-metrics   # 重要：必须启用metrics

# 终端 2 - GPU 3
python -m sglang.launch_server \
--model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
--host "0.0.0.0" \
--port 30006 \
--base-gpu-id 3 \
--enable-metrics
```

### 2. 启动路由器

```bash
bash /nas/ganluo/sglang_CUSTOM/bash_start_router.sh
```

### 3. 运行测试

#### 方法1：使用测试脚本
```bash
python test_queue_timestamps.py
```

#### 方法2：使用send_req.py
```bash
# 低负载测试
python send_req.py --num-requests 10 --request-rate 5

# 高负载测试（观察排队行为）
python send_req.py --num-requests 100 --request-rate 50
```

### 4. 检查结果

查看生成的CSV文件，新增的列包括：
- `pure_queue_time`: 纯排队时间（秒）
- `queue_time_start`: 进入队列时间（相对于测试开始）
- `queue_time_end`: 离开队列时间（相对于测试开始）

## 时间戳含义说明

```
请求生命周期：
arrival_time (客户端)
    ↓
send_time (发送到router)
    ↓
server_created_time (到达tokenizer_manager)
    ↓ [tokenize阶段]
queue_time_start (进入scheduler队列)
    ↓ [纯排队时间 = pure_queue_time]
queue_time_end (从队列取出)
    ↓ [prefill计算]
server_first_token_time (第一个token生成)
    ↓ [decode阶段]
completion_time (请求完成)
```

## 性能分析

通过新的时间戳，可以精确分析：

1. **Tokenize耗时** = queue_time_start - server_created_time
2. **纯排队时间** = queue_time_end - queue_time_start
3. **Prefill耗时** = server_first_token_time - queue_time_end
4. **总server时间** = server_first_token_time - server_created_time

## 注意事项

1. **必须启用--enable-metrics**：queue_time_start和queue_time_end只在enable_metrics=True时记录
2. **时间戳可能为None**：在低负载情况下，某些时间戳可能为None
3. **使用perf_counter**：scheduler使用time.perf_counter()而不是time.time()，可能需要转换
# SGLang Router 测试

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

### 5. 参数解读
目前记录在csv中的参数有:
req_id,input_length,decode_length,arrival_time,to_server_time,finish_time,server_latency,total_latency,ttft,queue_time,queue_time_in_server,pure_queue_time,success,error,host,server_created_time,server_first_token_time,queue_time_start,queue_time_end

#### 参数含义详解：

**基础信息：**
- `req_id`: 请求的唯一标识符
- `input_length`: 输入token的长度
- `decode_length`: 生成的token长度（目前为0，因为测试使用的是tokenize模式）
- `success`: 请求是否成功（True/False）
- `error`: 错误信息（如果有）
- `host`: 实际处理请求的服务器地址

**时间戳（按时间顺序）：**
- `arrival_time`: 请求到达router的时刻（从测试开始计时）
- `to_server_time`: router将请求转发到server的时刻
- `server_created_time`: 请求在server端被tokenizer_manager接收的时刻
- `queue_time_start`: 请求进入scheduler waiting_queue的时刻
- `queue_time_end`: 请求从waiting_queue取出准备处理的时刻
- `server_first_token_time`: server生成第一个token的时刻（prefill完成）
- `finish_time`: 请求完成的时刻

**延迟指标：**
- `server_latency`: 服务器处理时间 = `finish_time - to_server_time`
- `total_latency`: 总延迟 = `finish_time - arrival_time`
- `ttft` (Time To First Token): 第一个token生成时间（从客户端发送时刻算起）
- `queue_time`: 客户端视角的排队时间 = `to_server_time - arrival_time`（router中几乎为0）
- `queue_time_in_server`: server端总排队时间 = `server_first_token_time - server_created_time`
- `pure_queue_time`: scheduler纯排队时间 = `queue_time_end - queue_time_start`（不含tokenize时间）

**各阶段耗时计算：**
1. **Router转发延迟**: `to_server_time - arrival_time` （通常<0.1ms）
2. **Tokenize时间**: `queue_time_start - server_created_time`
3. **Scheduler排队时间**: `queue_time_end - queue_time_start`
4. **Prefill时间**: `server_first_token_time - queue_time_end`
5. **网络传输时间**: `server_created_time - to_server_time`

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
# Queue时间戳最终修复方案

## 修改总结

### 1. 删除enable_metrics条件限制
文件：`python/sglang/srt/managers/scheduler.py`
- 第1759行：删除了queue_time_end设置的enable_metrics条件检查

### 2. 统一时间基准
文件：`python/sglang/srt/managers/scheduler.py`
- 第1247行：`time.perf_counter()` → `time.time()`
- 第1255行：`time.perf_counter()` → `time.time()`
- 第1768行：`time.perf_counter()` → `time.time()`

理由：保持与其他组件（tokenizer_manager等）的时间基准一致

### 3. 添加调试日志
增加了三个级别的调试日志：

#### scheduler.py
- 初始化时记录enable_metrics状态
- 设置queue_time_start时记录
- 设置queue_time_end时记录并计算duration

#### scheduler_output_processor_mixin.py
- 收集queue时间戳时记录
- 发送BatchTokenIDOut前记录时间戳值

### 4. 修复测试脚本
文件：`verify_queue_fix.py`
- 修复API格式问题（"prompt" → "text"）
- 默认端口改为路由器60009
- 添加错误处理和格式自动检测

### 5. 创建新测试工具
文件：`test_router_timestamps.py`
- 专门测试路由器的queue时间戳功能
- 自动检测API格式
- 提供详细的时间戳分析
- 包含单请求和并发请求测试

## 部署步骤

```bash
# 1. 同步修改后的文件到服务器
rsync -av /Users/luogan/Code/sglang/python/sglang/srt/managers/scheduler.py user@server:/path/to/sglang/python/sglang/srt/managers/
rsync -av /Users/luogan/Code/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py user@server:/path/to/sglang/python/sglang/srt/managers/

# 2. 重新安装（如果使用pip install -e）
cd /path/to/sglang
pip install -e "python[all]"

# 3. 重启所有服务（注意顺序）
# 停止服务
pkill -f "python -m sglang.launch_server"
pkill -f "start_router.py"

# 启动服务器（使用--log-level debug查看调试信息）
python -m sglang.launch_server \
    --model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
    --host "0.0.0.0" \
    --port 60005 \
    --base-gpu-id 2 \
    --enable-metrics \
    --log-level debug  # 可选，用于调试

python -m sglang.launch_server \
    --model-path "/nas/models/Meta-Llama-3-8B-Instruct" \
    --host "0.0.0.0" \
    --port 60006 \
    --base-gpu-id 3 \
    --enable-metrics \
    --log-level debug  # 可选，用于调试

# 启动路由器
python start_router.py \
    --port 60009 \
    --backend-urls http://localhost:60005 http://localhost:60006
```

## 测试验证

### 1. 使用新的测试脚本
```bash
# 测试路由器时间戳功能
python test_router_timestamps.py --base-url http://localhost:60009

# 测试单个服务器（可选）
python test_router_timestamps.py --base-url http://localhost:60005
```

### 2. 查看调试日志
在服务器日志中查找：
- `Scheduler initialized with enable_metrics=True`
- `[Queue] Set queue_time_start`
- `[Queue] Set queue_time_end`
- `[QueueTime] Collecting for req`
- `[QueueTime] Sending BatchTokenIDOut`

### 3. 运行完整路由测试
```bash
python send_req.py \
    --num-requests 100 \
    --request-rate 50 \
    --output-path router_test_final.csv
```

### 4. 验证CSV输出
检查以下列是否有值：
- `queue_time_start`：应该是Unix时间戳（如1753795137.123）
- `queue_time_end`：应该略大于queue_time_start
- `pure_queue_time`：应该是正数（秒）

## 预期结果

修复后的时间戳应该显示：
```
✅ server_created_time: 1753795137.123456
✅ queue_time_start: 1753795137.124567
✅ queue_time_end: 1753795137.125678
✅ server_first_token_time: 1753795138.234567
⏱️  纯排队时间: 1.11ms
⏱️  Tokenize时间: 1.11ms
⏱️  总服务器时间: 1111.11ms
```

## 故障排查

如果时间戳仍然为null：

1. **确认代码部署**
   - 在服务器上运行`grep "time.time()" scheduler.py`确认修改
   - 检查是否需要重新pip install -e

2. **检查日志输出**
   - 使用--log-level debug启动服务
   - 查找[Queue]和[QueueTime]相关日志

3. **验证请求流程**
   - 确认请求确实经过scheduler的waiting_queue
   - 某些特殊请求可能走不同的路径

4. **检查兼容性**
   - 确保所有服务器都使用相同版本的代码
   - 路由器可能需要重启以获取新的响应格式

## 技术说明

### 时间戳流程
1. 请求到达tokenizer_manager → created_time
2. 请求进入scheduler队列 → queue_time_start
3. 请求从队列取出准备处理 → queue_time_end
4. 第一个token生成完成 → first_token_time

### 关键代码路径
- scheduler.py: `_add_request_to_queue()` 和 `get_new_batch_prefill()`
- scheduler_output_processor_mixin.py: `stream_output_generation()`
- tokenizer_manager.py: `_handle_batch_output()`

### 性能影响
- 时间戳记录的开销极小（< 0.01%）
- 调试日志只在DEBUG级别启用
- 不影响正常的请求处理流程
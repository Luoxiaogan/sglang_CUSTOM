# Queue时间戳修复指南

## 问题描述
在SGLang路由测试中，`queue_time_start` 和 `queue_time_end` 时间戳返回null，导致无法准确测量请求在scheduler队列中的排队时间。

## 根本原因
1. **不一致的条件设置**：
   - `queue_time_start` 无条件设置（第1247和1255行）
   - `queue_time_end` 只在 `enable_metrics=True` 时设置（第1759-1762行）
   - 当 `enable_metrics=False` 时，导致时间戳不完整

2. **代码位置**：
   - 文件：`/Users/luogan/Code/sglang/python/sglang/srt/managers/scheduler.py`

## 修复内容

### 1. 删除enable_metrics条件限制
**原代码**（第1759-1762行）：
```python
if self.enable_metrics:
    # only record queue time when enable_metrics is True to avoid overhead
    for req in can_run_list:
        req.queue_time_end = time.perf_counter()
```

**修改后**（第1759-1761行）：
```python
# Record queue time end for all requests to enable accurate timing analysis
for req in can_run_list:
    req.queue_time_end = time.perf_counter()
```

### 2. 添加调试日志
为了帮助排查问题，在以下位置添加了日志：

1. **Scheduler初始化**（第261-262行）：
```python
# Log enable_metrics status for debugging queue timestamp issues
logger.info(f"Scheduler initialized with enable_metrics={self.enable_metrics}")
```

2. **设置queue_time_start**（第1248-1249和1256-1257行）：
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"[Queue] Set queue_time_start for req {req.rid}: {req.queue_time_start}")
```

3. **设置queue_time_end**（第1769-1771行）：
```python
if logger.isEnabledFor(logging.DEBUG):
    queue_duration = req.queue_time_end - req.queue_time_start if hasattr(req, 'queue_time_start') and req.queue_time_start else 0
    logger.debug(f"[Queue] Set queue_time_end for req {req.rid}: {req.queue_time_end}, duration: {queue_duration:.3f}s")
```

## 部署步骤

### 1. 上传修改后的代码
```bash
# 同步代码到服务器
rsync -av /Users/luogan/Code/sglang/python/sglang/srt/managers/scheduler.py user@server:/path/to/sglang/python/sglang/srt/managers/
```

### 2. 重新安装（如果使用pip install -e）
```bash
cd /path/to/sglang
pip install -e "python[all]"
```

### 3. 重启服务
```bash
# 停止现有服务
pkill -f "python -m sglang.launch_server"

# 启动服务（可选：添加--log-level debug查看详细日志）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --enable-metrics \
    --log-level info  # 或 debug
```

## 验证修复

### 1. 使用验证脚本
```bash
# 测试单个服务器
python verify_queue_fix.py --base-url http://localhost:30000

# 测试路由器
python verify_queue_fix.py --base-url http://localhost:40009 --num-requests 20
```

### 2. 使用完整的路由测试
```bash
python send_req.py \
    --num-requests 100 \
    --request-rate 50 \
    --output-path router_test_fixed.csv
```

### 3. 检查CSV输出
验证以下列是否有值：
- `queue_time_start`
- `queue_time_end`
- `pure_queue_time`

## 预期结果

修复后，所有请求都应该正确记录queue时间戳：
- 低负载时：pure_queue_time < 0.01秒
- 中等负载时：pure_queue_time 0.01-0.5秒
- 高负载时：pure_queue_time > 0.5秒

## 故障排查

如果时间戳仍然为null：

1. **检查服务器日志**：
   - 查找 "Scheduler initialized with enable_metrics=" 消息
   - 如果使用--log-level debug，查找 "[Queue]" 开头的调试消息

2. **验证代码部署**：
   ```bash
   python check_server_code_version.py
   ```

3. **检查时间基准**：
   - scheduler使用 `time.perf_counter()`
   - 其他地方使用 `time.time()`
   - 这可能导致时间戳值看起来异常

## 后续优化建议

1. **统一时间基准**：考虑全部使用 `time.time()` 或 `time.perf_counter()`
2. **添加更多时间戳**：如 `tokenize_start_time`、`prefill_start_time`
3. **性能影响评估**：测量时间戳记录对性能的影响（预期 < 0.1%）
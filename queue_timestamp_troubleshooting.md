# Queue Timestamp 问题排查指南

## 当前问题

queue_time_start 和 queue_time_end 时间戳返回 null，即使：
- 服务器已启用 --enable-metrics
- 代码修改已完成
- server_created_time 正常返回

## 排查步骤

### 1. 验证代码版本

在服务器上运行：
```bash
python check_server_code_version.py
```

这会检查：
- BatchTokenIDOut 是否有 queue_time_start/end 字段
- scheduler_output_processor_mixin.py 的修改是否生效
- tokenizer_manager.py 的修改是否生效

### 2. 直接测试服务器

绕过 router，直接测试 server：
```bash
python test_server_directly.py
```

这会：
- 直接向 server 发送请求
- 创建并发请求以产生排队
- 显示完整的 meta_info 响应

### 3. 应用调试补丁（如果需要）

运行：
```bash
python debug_scheduler_timestamps.py
```

这会创建调试补丁文件，然后：
1. 手动将补丁应用到对应文件
2. 重启服务器时加上 `--log-level info`
3. 查看服务器日志中的 [DEBUG] 消息

### 4. 检查可能的原因

#### 原因 1: 代码未同步
- 确认修改的文件已保存
- 如果使用 pip install -e，确保没有 .pyc 缓存
- 重启所有服务器进程

#### 原因 2: 参数不匹配
已修复的问题：
- spec_verify_ct 条件 append 导致列表长度不一致
- 需要确保所有列表长度相同

#### 原因 3: 时间基准问题
- scheduler 使用 time.perf_counter()
- 其他地方使用 time.time()
- 可能需要转换（但通常不是问题）

## 修复历史

### 第一次修复
- 添加 queue_time_start/end 到 BatchTokenIDOut
- 在 scheduler_output_processor_mixin 中收集时间戳
- 在 tokenizer_manager 中返回时间戳

### 第二次修复
- 添加 hasattr 检查避免属性错误
- 修复 spec_verify_ct 条件 append 问题

## 验证方法

成功的标志：
1. CSV 文件中 queue_time_start 和 queue_time_end 有值
2. pure_queue_time 被正确计算
3. 可以区分 tokenize、排队、prefill 各阶段耗时

## 备选方案

如果上述方法都无效：
1. 在 Req 类初始化时设置默认值
2. 使用不同的数据传递机制
3. 考虑在 server 端计算并返回 pure_queue_time
#!/usr/bin/env python3
"""
简单的补丁 - 直接显示需要修改的位置和内容
"""

print("""
========================================
SGLang tokenizer_manager.py 时间戳补丁
========================================

需要修改的位置：

1. 在文件开头添加（第30行附近，import 语句之后）：
----------------------------------------
# Timestamp tracking version
TOKENIZER_MANAGER_VERSION = "v2_timestamps_2025"
logger.info(f"TokenizerManager loaded with version: {TOKENIZER_MANAGER_VERSION}")
----------------------------------------

2. 找到 _handle_batch_output 方法中的这段代码（约1408-1417行）：

原代码：
            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

修改为：
            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                        # Add server-side timestamps
                        "server_created_time": state.created_time,
                        "server_first_token_time": state.first_token_time if state.first_token_time > 0 else None,
                        "_version": TOKENIZER_MANAGER_VERSION,  # 版本标记
                    }
                )
                # 添加调试日志
                logger.debug(f"Added timestamps to meta_info: created={state.created_time}, first_token={state.first_token_time}")

3. 在 e2e_latency 设置之前添加日志（约1455行）：

原代码：
                meta_info["e2e_latency"] = state.finished_time - state.created_time

修改为：
                meta_info["e2e_latency"] = state.finished_time - state.created_time
                logger.info(f"Request {rid} finished with e2e_latency={meta_info['e2e_latency']:.3f}s")

========================================
应用步骤：
1. SSH 到远程服务器
2. 编辑 /nas/ganluo/sglang/python/sglang/srt/managers/tokenizer_manager.py
3. 应用上述修改
4. 删除 .pyc 文件：find /nas/ganluo/sglang -name "*.pyc" -delete
5. 重启服务器
6. 运行测试验证
========================================
""")
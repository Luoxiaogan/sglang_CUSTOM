#!/bin/bash
# 应用调试补丁到远程服务器的脚本

echo "=========================================="
echo "应用调试补丁到 tokenizer_manager.py"
echo "=========================================="

# 配置
REMOTE_HOST="your_remote_host"  # 修改为你的远程服务器
REMOTE_PATH="/nas/ganluo/sglang/python/sglang/srt/managers/tokenizer_manager.py"
LOCAL_ORIGINAL="/Users/luogan/Code/sglang/python/sglang/srt/managers/tokenizer_manager.py"

# 1. 备份远程文件
echo "1. 备份远程文件..."
ssh $REMOTE_HOST "cp $REMOTE_PATH ${REMOTE_PATH}.backup_$(date +%Y%m%d_%H%M%S)"

# 2. 在本地创建调试版本
echo "2. 创建调试版本..."
cp $LOCAL_ORIGINAL tokenizer_manager_with_debug.py

# 3. 应用调试修改
echo "3. 应用调试修改..."
cat >> tokenizer_manager_with_debug.py << 'EOF'

# ===== 调试补丁开始 =====
# 在文件开头添加版本标记
import logging
logger = logging.getLogger(__name__)

TOKENIZER_MANAGER_VERSION = "v2_with_timestamps_debug"
print(f"=== TokenizerManager module loaded with version: {TOKENIZER_MANAGER_VERSION} ===")

# 注入到 _handle_batch_output 方法
# 需要手动修改该方法，添加日志
# 1. 在方法开始添加：
#    logger.info(f"=== DEBUG: _handle_batch_output called with {type(recv_obj).__name__} ===")
# 2. 在 meta_info.update 之前添加：
#    logger.info(f"=== DEBUG: Adding timestamps - created_time={state.created_time}, first_token_time={state.first_token_time} ===")
# 3. 在 meta_info.update 中添加：
#    "_debug_version": TOKENIZER_MANAGER_VERSION,
# ===== 调试补丁结束 =====
EOF

# 4. 复制到远程
echo "4. 复制到远程服务器..."
scp tokenizer_manager_with_debug.py $REMOTE_HOST:$REMOTE_PATH

# 5. 清理远程 pyc 文件
echo "5. 清理远程 .pyc 文件..."
ssh $REMOTE_HOST "find /nas/ganluo/sglang -name '*.pyc' -delete"
ssh $REMOTE_HOST "find /nas/ganluo/sglang -name '__pycache__' -type d -exec rm -rf {} +"

# 6. 重启提示
echo ""
echo "=========================================="
echo "补丁应用完成！"
echo "=========================================="
echo "请执行以下步骤："
echo "1. SSH 到远程服务器"
echo "2. 手动编辑 $REMOTE_PATH"
echo "3. 在 _handle_batch_output 方法中添加调试日志"
echo "4. 在 meta_info.update 中添加 '_debug_version' 字段"
echo "5. 重启所有 SGLang 服务"
echo "6. 运行 check_server_version.py 验证"
echo "=========================================="

# 清理本地临时文件
rm -f tokenizer_manager_with_debug.py
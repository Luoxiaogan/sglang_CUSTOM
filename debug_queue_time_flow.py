#!/usr/bin/env python3
"""
调试补丁：在关键位置添加日志以追踪queue_time的传输
"""

import os

def add_debug_logs():
    """在多个文件中添加调试日志"""
    
    # 1. 在scheduler_output_processor_mixin.py中添加发送前的日志
    scheduler_file = "/Users/luogan/Code/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py"
    
    print("添加调试日志到scheduler_output_processor_mixin.py...")
    
    # 读取文件
    with open(scheduler_file, 'r') as f:
        content = f.read()
    
    # 在BatchTokenIDOut构造之前添加详细日志
    debug_log = '''
            # DEBUG: Log queue time values before sending
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[QueueTimeDebug] About to create BatchTokenIDOut")
                logger.debug(f"[QueueTimeDebug] Number of requests: {len(rids)}")
                logger.debug(f"[QueueTimeDebug] queue_time_start type: {type(queue_time_start)}")
                logger.debug(f"[QueueTimeDebug] queue_time_start length: {len(queue_time_start) if queue_time_start else 0}")
                logger.debug(f"[QueueTimeDebug] queue_time_start values: {queue_time_start[:3] if queue_time_start else 'None'}...")
                logger.debug(f"[QueueTimeDebug] queue_time_end type: {type(queue_time_end)}")
                logger.debug(f"[QueueTimeDebug] queue_time_end length: {len(queue_time_end) if queue_time_end else 0}")
                logger.debug(f"[QueueTimeDebug] queue_time_end values: {queue_time_end[:3] if queue_time_end else 'None'}...")
'''
    
    # 在send_pyobj之后添加日志
    send_log = '''
            )
            
            # DEBUG: Log after creating BatchTokenIDOut
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[QueueTimeDebug] BatchTokenIDOut created and sent")
                logger.debug(f"[QueueTimeDebug] Sent object has queue_time_start: {hasattr(batch_out, 'queue_time_start')}")
                if hasattr(batch_out, 'queue_time_start'):
                    logger.debug(f"[QueueTimeDebug] Sent queue_time_start: {batch_out.queue_time_start[:3] if batch_out.queue_time_start else 'None'}...")
            
            self.send_to_detokenizer.send_pyobj(batch_out'''
    
    # 2. 在tokenizer_manager.py中添加接收后的日志
    tokenizer_file = "/Users/luogan/Code/sglang/python/sglang/srt/managers/tokenizer_manager.py"
    
    print("添加调试日志到tokenizer_manager.py...")
    
    # 在recv_pyobj之后添加日志
    recv_log = '''
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            
            # DEBUG: Log received object
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[QueueTimeDebug] Received object type: {type(recv_obj).__name__}")
                logger.debug(f"[QueueTimeDebug] Has queue_time_start: {hasattr(recv_obj, 'queue_time_start')}")
                logger.debug(f"[QueueTimeDebug] Has queue_time_end: {hasattr(recv_obj, 'queue_time_end')}")
                if hasattr(recv_obj, 'queue_time_start') and recv_obj.queue_time_start:
                    logger.debug(f"[QueueTimeDebug] Received queue_time_start[0]: {recv_obj.queue_time_start[0] if len(recv_obj.queue_time_start) > 0 else 'empty list'}")
'''
    
    print("\n调试日志位置说明:")
    print("1. scheduler_output_processor_mixin.py - 在发送BatchTokenIDOut之前和之后")
    print("2. tokenizer_manager.py - 在接收到对象之后")
    print("\n请手动添加这些日志，然后重启服务器并运行测试")

if __name__ == "__main__":
    print("Queue Time 传输调试补丁生成器")
    print("=" * 60)
    add_debug_logs()
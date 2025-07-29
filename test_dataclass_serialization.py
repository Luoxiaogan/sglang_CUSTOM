#!/usr/bin/env python3
"""
测试BatchTokenIDOut的序列化
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from sglang.srt.managers.io_struct import BatchTokenIDOut, BaseFinishReason
import pickle
import json

def test_batch_token_id_out():
    """测试BatchTokenIDOut的创建和属性访问"""
    
    # 创建一个简单的测试实例
    test_obj = BatchTokenIDOut(
        rids=["test_rid_1"],
        finished_reasons=[BaseFinishReason(type="stop")],
        decoded_texts=["test text"],
        decode_ids=[1],
        read_offsets=[0],
        output_ids=None,
        skip_special_tokens=[True],
        spaces_between_special_tokens=[False],
        no_stop_trim=[False],
        prompt_tokens=[10],
        completion_tokens=[5],
        cached_tokens=[2],
        spec_verify_ct=[0],
        input_token_logprobs_val=[0.1],
        input_token_logprobs_idx=[1],
        output_token_logprobs_val=[0.2],
        output_token_logprobs_idx=[2],
        input_top_logprobs_val=[[]],
        input_top_logprobs_idx=[[]],
        output_top_logprobs_val=[[]],
        output_top_logprobs_idx=[[]],
        input_token_ids_logprobs_val=[[]],
        input_token_ids_logprobs_idx=[[]],
        output_token_ids_logprobs_val=[[]],
        output_token_ids_logprobs_idx=[[]],
        output_hidden_states=[[]],
        queue_time_start=[1234567890.123],
        queue_time_end=[1234567890.456]
    )
    
    print("✅ BatchTokenIDOut 创建成功")
    
    # 检查属性
    print("\n属性检查:")
    print(f"  rids: {test_obj.rids}")
    print(f"  queue_time_start exists: {hasattr(test_obj, 'queue_time_start')}")
    print(f"  queue_time_start: {test_obj.queue_time_start}")
    print(f"  queue_time_end exists: {hasattr(test_obj, 'queue_time_end')}")
    print(f"  queue_time_end: {test_obj.queue_time_end}")
    
    # 测试pickle序列化
    print("\n测试pickle序列化:")
    try:
        serialized = pickle.dumps(test_obj)
        deserialized = pickle.loads(serialized)
        print("  ✅ Pickle序列化/反序列化成功")
        print(f"  反序列化后queue_time_start: {deserialized.queue_time_start}")
        print(f"  反序列化后queue_time_end: {deserialized.queue_time_end}")
    except Exception as e:
        print(f"  ❌ Pickle序列化失败: {e}")
    
    # 列出所有属性
    print("\n所有属性:")
    attrs = [attr for attr in dir(test_obj) if not attr.startswith('_')]
    for attr in sorted(attrs):
        print(f"  {attr}")

if __name__ == "__main__":
    print("测试BatchTokenIDOut dataclass")
    print("=" * 60)
    test_batch_token_id_out()
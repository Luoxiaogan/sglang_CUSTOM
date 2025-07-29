#!/usr/bin/env python3
"""
测试CSV导出功能，验证queue时间戳的相对时间导出
"""

import pandas as pd
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

from core.request_generator import Request, RequestGenerator, RequestSender
from core.metrics_collector import MetricsCollector


async def test_csv_export():
    """测试CSV导出功能"""
    
    print("CSV导出功能测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化组件
    generator = RequestGenerator()
    sender = RequestSender()
    metrics_collector = MetricsCollector()
    
    # 创建测试请求
    requests = [
        Request(
            request_id=f"test_{i}",
            prompt=f"Test prompt {i}: What is {i} + {i}?",
            prompt_len=20,
            output_len=10,
            arrival_time=time.time() + i * 0.5  # 模拟每0.5秒一个请求
        )
        for i in range(3)
    ]
    
    print(f"\n创建了 {len(requests)} 个测试请求")
    
    # 服务器URL
    server_url = "http://localhost:60005/generate"
    
    # 初始化会话
    await sender.initialize()
    
    try:
        # 发送请求并收集结果
        print("\n发送请求到服务器...")
        for i, request in enumerate(requests):
            print(f"\n请求 {i+1}/{len(requests)}: {request.request_id}")
            
            # 发送请求
            result = await sender.send_request(request, server_url, stream=True)
            
            # 打印结果信息
            if result.success:
                print(f"  ✅ 成功")
                print(f"  生成文本: {result.generated_text[:50]}...")
                print(f"  Server latency: {result.server_latency_ms:.1f} ms")
                print(f"  Queue time: {result.queue_time_ms:.1f} ms")
                
                # 打印queue时间戳信息
                if result.queue_time_start and result.queue_time_end:
                    print(f"  Queue时间戳:")
                    print(f"    - queue_time_start: {result.queue_time_start}")
                    print(f"    - queue_time_end: {result.queue_time_end}")
                    print(f"    - pure_queue_time: {result.pure_queue_time_ms:.3f} ms")
                else:
                    print(f"  ⚠️ 未收到queue时间戳")
            else:
                print(f"  ❌ 失败: {result.error}")
            
            # 添加到metrics collector
            metrics_collector.add_result(result)
            
            # 短暂延迟
            await asyncio.sleep(0.2)
        
        # 导出CSV
        print("\n\n导出测试结果到CSV...")
        csv_path = metrics_collector.export_metrics("csv", "test_csv_export.csv")
        print(f"CSV文件已导出到: {csv_path}")
        
        # 读取并验证CSV内容
        print("\n验证CSV内容...")
        df = pd.read_csv(csv_path)
        
        # 打印列名
        print(f"\nCSV列: {list(df.columns)}")
        
        # 检查新增的列
        new_columns = ['queue_time_start_relative', 'queue_time_end_relative', 
                      'pure_queue_time', 'pure_queue_time_ms']
        
        for col in new_columns:
            if col in df.columns:
                print(f"\n✅ 找到列 '{col}':")
                print(f"   值: {df[col].tolist()}")
            else:
                print(f"\n❌ 缺少列 '{col}'")
        
        # 打印CSV前几行
        print("\n\nCSV内容预览:")
        print(df.to_string(max_rows=10, max_cols=None))
        
        # 验证相对时间计算
        if 'arrival_time' in df.columns and 'queue_time_start_relative' in df.columns:
            print("\n\n验证相对时间计算:")
            for idx, row in df.iterrows():
                if pd.notna(row.get('queue_time_start_relative')):
                    print(f"\n请求 {row['req_id']}:")
                    print(f"  arrival_time: {row['arrival_time']:.3f}s")
                    print(f"  queue_time_start_relative: {row['queue_time_start_relative']:.3f}s")
                    print(f"  queue_time_end_relative: {row['queue_time_end_relative']:.3f}s")
                    print(f"  pure_queue_time_ms: {row.get('pure_queue_time_ms', 'N/A')} ms")
        
        # 导出JSON用于对比
        json_path = metrics_collector.export_metrics("json", "test_csv_export.json")
        print(f"\n\nJSON文件已导出到: {json_path}")
        
    finally:
        await sender.close()
        print("\n\n测试完成！")


if __name__ == "__main__":
    asyncio.run(test_csv_export())
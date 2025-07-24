import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from sglang_test_framework import (
    NodeConfig, ServerManager, RequestGenerator, 
    RequestSender, MetricsCollector, ResultManager
)

async def run_node_test():
    # 1. 配置测试
    config = NodeConfig(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",  # Use Hugging Face model ID
        gpu_id=1,
        max_running_requests=256,
        request_rate=10.0,
        num_prompts=100,  # Reduced for testing
    )
    
    # 2. 初始化组件
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    metrics_collector = MetricsCollector()
    result_manager = ResultManager(config.output_dir)
    
    # 3. 启动服务器
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    
    # 4. 生成请求
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    
    # 分配泊松到达时间
    requests = request_generator.generate_poisson_arrivals(
        requests, config.request_rate
    )
    
    # 5. 开始指标收集
    metrics_collector.start_collection()
    
    # 6. 发送请求并收集结果
    api_url = f"http://localhost:{server_config.port}/generate"
    
    async with RequestSender() as sender:
        tasks = []
        for request in requests:
            # 等待到达时间
            wait_time = request.arrival_time - asyncio.get_event_loop().time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # 记录请求开始
            metrics_collector.record_request_start(request.request_id)
            
            # 发送请求
            task = asyncio.create_task(sender.send_request(request, api_url))
            tasks.append(task)
        
        # 等待所有请求完成
        results = await asyncio.gather(*tasks)
        
        # 记录结果
        for result in results:
            metrics_collector.record_request_complete(result)
    
    # 7. 停止收集并生成报告
    metrics_collector.stop_collection()
    
    # 8. 保存结果和生成可视化
    test_dir = result_manager.save_results(
        metrics_collector, 
        config.to_dict(),
        "node_test_example"
    )
    
    plots = result_manager.create_visualizations(
        metrics_collector,
        test_dir
    )
    
    # 9. 打印摘要
    metrics_collector.print_summary()
    
    # 10. 清理
    server_manager.stop_all()

# 运行测试
asyncio.run(run_node_test())
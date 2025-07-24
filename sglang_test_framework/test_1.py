import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable server logs by setting environment variable before importing
# You can also set SGLANG_TEST_LOG_LEVEL=DEBUG for more detailed logs
os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'

import asyncio
import logging
from sglang_test_framework import (
    NodeConfig, ServerManager, RequestGenerator, 
    RequestSender, MetricsCollector, ResultManager
)

# Get logger for this module
logger = logging.getLogger(__name__)

async def run_node_test():
    logger.info("Starting SGLang node test...")
    
    # 1. 配置测试
    logger.info("Setting up test configuration...")
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",  # Use Hugging Face model ID
        gpu_id=1,
        max_running_requests=256,
        request_rate=10.0,
        num_prompts=100,  # Reduced for testing
        dataset_name="random",
    )
    logger.info(f"Test configuration: {config.num_prompts} prompts at {config.request_rate} req/s")
    
    # 2. 初始化组件
    logger.info("Initializing test components...")
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    metrics_collector = MetricsCollector()
    result_manager = ResultManager(config.output_dir)
    
    # 3. 启动服务器
    logger.info("Launching SGLang server...")
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    logger.info(f"Server started successfully on port {server_config.port}")
    
    # 4. 生成请求
    logger.info(f"Generating {config.num_prompts} test requests...")
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    logger.info(f"Generated {len(requests)} requests")
    
    # 分配泊松到达时间
    requests = request_generator.generate_poisson_arrivals(
        requests, config.request_rate
    )
    
    # 5. 开始指标收集
    logger.info("Starting metrics collection...")
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
        logger.info(f"Waiting for {len(tasks)} requests to complete...")
        results = await asyncio.gather(*tasks)
        
        # 记录结果
        successful = sum(1 for r in results if r.success)
        logger.info(f"Completed {len(results)} requests, {successful} successful")
        for result in results:
            metrics_collector.record_request_complete(result)
    
    # 7. 停止收集并生成报告
    metrics_collector.stop_collection()
    
    # 8. 保存结果和生成可视化
    logger.info("Saving test results...")
    test_dir = result_manager.save_results(
        metrics_collector, 
        config.to_dict(),
        "node_test_example"
    )
    logger.info(f"Results saved to: {test_dir}")
    
    plots = result_manager.create_visualizations(
        metrics_collector,
        test_dir
    )
    
    # 9. 打印摘要
    metrics_collector.print_summary()
    
    # 10. 清理
    logger.info("Cleaning up...")
    server_manager.stop_all()
    logger.info("Test completed successfully!")

# 运行测试
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("SGLang Testing Framework - Node Test")
    logger.info("="*50)
    
    try:
        asyncio.run(run_node_test())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
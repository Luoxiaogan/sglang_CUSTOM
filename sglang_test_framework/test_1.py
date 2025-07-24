import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable server logs by setting environment variable before importing
# You can also set SGLANG_TEST_LOG_LEVEL=DEBUG for more detailed logs
os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'

import asyncio
import logging
import time
from sglang_test_framework import (
    NodeConfig, ServerManager, RequestGenerator, 
    RequestSender, MetricsCollector, ResultManager,
    RequestResult
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
    
    # Enable incremental saving
    test_name = "node_test_example"
    incremental_path = str(result_manager.output_dir / test_name / "results")
    result_manager.output_dir.joinpath(test_name).mkdir(parents=True, exist_ok=True)
    metrics_collector.enable_incremental_save(incremental_path, interval=50)
    
    # 6. 发送请求并收集结果
    api_url = f"http://localhost:{server_config.port}/generate"
    
    # Use a semaphore to limit concurrent requests
    max_concurrent_requests = 20  # Limit concurrent requests to avoid overwhelming the server
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def send_request_with_semaphore(request, sender, api_url):
        """Send a request with semaphore to limit concurrency and retry logic."""
        async with semaphore:
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    result = await sender.send_request(request, api_url, stream=True)
                    if result.success:
                        return result
                    else:
                        logger.warning(f"Request {request.request_id} failed (attempt {attempt + 1}/{max_retries}): {result.error}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            return result
                except Exception as e:
                    logger.error(f"Request {request.request_id} exception (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        return RequestResult(
                            request=request,
                            success=False,
                            error=f"Failed after {max_retries} attempts: {str(e)}"
                        )
    
    async with RequestSender() as sender:
        tasks = []
        test_start_time = asyncio.get_event_loop().time()
        
        for i, request in enumerate(requests):
            # Calculate absolute arrival time from relative time
            absolute_arrival_time = test_start_time + request.arrival_time
            current_time = asyncio.get_event_loop().time()
            wait_time = absolute_arrival_time - current_time
            
            logger.debug(f"Request {i+1}: relative_arrival={request.arrival_time:.3f}, absolute_arrival={absolute_arrival_time:.3f}, current_time={current_time:.3f}, wait_time={wait_time:.3f}")
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Update arrival_time to absolute time after sleep
            request.arrival_time = time.time()
            
            # 记录请求开始
            logger.info(f"Sending request {i+1}/{len(requests)} (ID: {request.request_id})")
            metrics_collector.record_request_start(request.request_id)
            
            # 发送请求 with semaphore
            task = asyncio.create_task(send_request_with_semaphore(request, sender, api_url))
            tasks.append(task)
            
            # Log progress every 10 requests
            if (i + 1) % 10 == 0:
                elapsed = asyncio.get_event_loop().time() - test_start_time
                logger.info(f"Progress: {i+1}/{len(requests)} requests sent in {elapsed:.1f}s")
        
        # 等待所有请求完成
        logger.info(f"All requests sent. Waiting for {len(tasks)} requests to complete...")
        
        # Use asyncio.wait with timeout to avoid hanging
        done, pending = await asyncio.wait(tasks, timeout=300)  # 5 minute timeout
        
        if pending:
            logger.warning(f"{len(pending)} requests timed out!")
            for task in pending:
                task.cancel()
        
        # Collect results
        results = []
        for task in done:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to get result: {e}")
        
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
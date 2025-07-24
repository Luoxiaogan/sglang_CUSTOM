import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable server logs and debug logging
os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'
# os.environ['SGLANG_TEST_LOG_LEVEL'] = 'DEBUG'  # Uncomment for debug logs

import asyncio
import logging
import time
from sglang_test_framework import (
    NodeConfig, ServerManager, RequestGenerator, 
    RequestSender, MetricsCollector, ResultManager,
    RequestResult
)

logger = logging.getLogger(__name__)

async def run_fixed_test():
    logger.info("Starting fixed SGLang test with improved error handling...")
    
    # Configuration
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=10.0,  # 10 requests per second
        num_prompts=50,     # Test with 50 requests
        dataset_name="random",
        random_input_len=512,   # Moderate input size
        random_output_len=128,
    )
    logger.info(f"Test configuration: {config.num_prompts} prompts at {config.request_rate} req/s")
    
    # Initialize components
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    metrics_collector = MetricsCollector()
    result_manager = ResultManager(config.output_dir)
    
    # Launch server
    logger.info("Launching SGLang server...")
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    logger.info(f"Server started successfully on port {server_config.port}")
    
    # Generate requests
    logger.info(f"Generating {config.num_prompts} test requests...")
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    
    # Generate Poisson arrivals
    requests = request_generator.generate_poisson_arrivals(
        requests, config.request_rate
    )
    logger.info(f"Generated {len(requests)} requests")
    
    # Start metrics collection
    logger.info("Starting metrics collection...")
    metrics_collector.start_collection()
    
    # Send requests with improved error handling
    api_url = f"http://localhost:{server_config.port}/generate"
    
    # Limit concurrent requests to avoid overwhelming the server
    max_concurrent_requests = 20
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def send_request_with_retry(request, sender, api_url):
        """Send request with retry logic and semaphore."""
        async with semaphore:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await sender.send_request(request, api_url, stream=True)
                    if result.success:
                        return result
                    else:
                        logger.warning(f"Request {request.request_id} failed (attempt {attempt + 1}): {result.error}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0 * (attempt + 1))
                except Exception as e:
                    logger.error(f"Request {request.request_id} exception (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        return RequestResult(
                            request=request,
                            success=False,
                            error=f"Failed after {max_retries} attempts: {str(e)}"
                        )
    
    async with RequestSender() as sender:
        tasks = []
        test_start_time = asyncio.get_event_loop().time()
        
        # Send requests according to their arrival times
        for i, request in enumerate(requests):
            # Calculate wait time
            absolute_arrival_time = test_start_time + request.arrival_time
            current_time = asyncio.get_event_loop().time()
            wait_time = absolute_arrival_time - current_time
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Update arrival_time to absolute time after sleep
            request.arrival_time = time.time()
            
            # Record request arrival (enters queue)
            metrics_collector.record_request_arrival(request.request_id)
            
            # Record and send request
            logger.info(f"Sending request {i+1}/{len(requests)} (ID: {request.request_id})")
            metrics_collector.record_request_start(request.request_id)
            
            task = asyncio.create_task(send_request_with_retry(request, sender, api_url))
            tasks.append(task)
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = asyncio.get_event_loop().time() - test_start_time
                logger.info(f"Progress: {i+1}/{len(requests)} requests sent in {elapsed:.1f}s")
        
        # Wait for all requests to complete
        logger.info(f"All requests sent. Waiting for {len(tasks)} requests to complete...")
        done, pending = await asyncio.wait(tasks, timeout=300)
        
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
        
        # Record results
        successful = sum(1 for r in results if r.success)
        logger.info(f"Completed {len(results)} requests, {successful} successful")
        
        for result in results:
            metrics_collector.record_request_complete(result)
    
    # Stop collection and generate report
    metrics_collector.stop_collection()
    
    # Save results
    logger.info("Saving test results...")
    test_dir = result_manager.save_results(
        metrics_collector, 
        config.to_dict(),
        "fixed_test"
    )
    logger.info(f"Results saved to: {test_dir}")
    
    # Generate visualizations
    plots = result_manager.create_visualizations(
        metrics_collector,
        test_dir
    )
    
    # Print summary
    metrics_collector.print_summary()
    
    # Cleanup
    logger.info("Cleaning up...")
    server_manager.stop_all()
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SGLang Testing Framework - Fixed Test")
    logger.info("="*60)
    
    try:
        asyncio.run(run_fixed_test())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
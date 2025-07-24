"""Quick validation test to verify the fixes work correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'
os.environ['SGLANG_TEST_LOG_LEVEL'] = 'INFO'

import asyncio
import logging
from sglang_test_framework import NodeConfig, ServerManager, RequestGenerator, RequestSender

logger = logging.getLogger(__name__)

async def validate_fixes():
    """Run a quick test to validate the fixes work."""
    logger.info("Running validation test...")
    
    # Minimal config
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=float('inf'),  # Send all at once
        num_prompts=10,
        dataset_name="random",
        random_input_len=100,
        random_output_len=50,
    )
    
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    
    # Start server
    logger.info("Starting server...")
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    
    # Generate requests
    logger.info("Generating requests...")
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    
    # Set all requests to arrive immediately
    for req in requests:
        req.arrival_time = 0
    
    api_url = f"http://localhost:{server_config.port}/generate"
    
    # Test concurrent requests with semaphore
    logger.info("Testing concurrent requests with semaphore...")
    semaphore = asyncio.Semaphore(5)  # Only 5 concurrent
    
    async def send_with_semaphore(request, sender):
        async with semaphore:
            logger.info(f"Sending request {request.request_id}")
            result = await sender.send_request(request, api_url, stream=True)
            logger.info(f"Request {request.request_id} completed: success={result.success}")
            return result
    
    async with RequestSender() as sender:
        tasks = [send_with_semaphore(req, sender) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = 0
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed with exception: {result}")
                failed += 1
            elif result.success:
                successful += 1
            else:
                logger.error(f"Request {i} failed: {result.error}")
                failed += 1
        
        logger.info(f"Results: {successful} successful, {failed} failed out of {len(requests)}")
        
        if successful == len(requests):
            logger.info("✅ All requests completed successfully!")
        else:
            logger.warning(f"⚠️  {failed} requests failed")
    
    # Cleanup
    logger.info("Cleaning up...")
    server_manager.stop_all()
    
    return successful == len(requests)

if __name__ == "__main__":
    try:
        success = asyncio.run(validate_fixes())
        if success:
            logger.info("✅ Validation test PASSED!")
            sys.exit(0)
        else:
            logger.error("❌ Validation test FAILED!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Validation test error: {e}", exc_info=True)
        sys.exit(1)
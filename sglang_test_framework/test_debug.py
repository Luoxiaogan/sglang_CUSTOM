import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable server logs and debug logging
os.environ['SGLANG_TEST_SHOW_SERVER_LOGS'] = 'true'
os.environ['SGLANG_TEST_LOG_LEVEL'] = 'DEBUG'

import asyncio
import logging
from sglang_test_framework import (
    NodeConfig, ServerManager, RequestGenerator, 
    RequestSender, MetricsCollector, ResultManager
)

# Get logger for this module
logger = logging.getLogger(__name__)

async def run_debug_test():
    logger.info("Starting debug test...")
    
    # Simple config with just 5 requests
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=float('inf'),  # Send all at once
        num_prompts=5,
        dataset_name="random",
        random_input_len=100,  # Small inputs for debugging
        random_output_len=50,
    )
    
    logger.info("Initializing components...")
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    
    logger.info("Launching server...")
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    logger.info(f"Server started on port {server_config.port}")
    
    logger.info("Generating requests...")
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    
    # Don't use Poisson arrivals for debugging
    for i, req in enumerate(requests):
        req.arrival_time = 0
    
    logger.info(f"Generated {len(requests)} requests")
    
    # Send requests one by one for debugging
    api_url = f"http://localhost:{server_config.port}/generate"
    
    async with RequestSender() as sender:
        for i, request in enumerate(requests):
            logger.info(f"Sending request {i+1}/{len(requests)} (ID: {request.request_id})")
            logger.info(f"Request prompt length: {len(request.prompt)}, expected output: {request.output_len}")
            
            try:
                result = await sender.send_request(request, api_url, stream=False)
                logger.info(f"Request {i+1} completed: success={result.success}")
                
                if result.success:
                    logger.info(f"Generated text length: {len(result.generated_text)}")
                    logger.info(f"Latencies - Server: {result.server_latency:.1f}ms, Total: {result.total_latency:.1f}ms")
                else:
                    logger.error(f"Request failed: {result.error}")
                    
            except Exception as e:
                logger.error(f"Exception sending request {i+1}: {e}", exc_info=True)
    
    logger.info("Cleaning up...")
    server_manager.stop_all()
    logger.info("Debug test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(run_debug_test())
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
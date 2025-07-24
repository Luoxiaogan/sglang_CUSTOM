"""Test to verify timing calculations are fixed."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import logging
from sglang_test_framework import NodeConfig, ServerManager, RequestGenerator, RequestSender, MetricsCollector, ResultManager

logger = logging.getLogger(__name__)

async def test_timing_fixes():
    """Test that timing calculations are correct."""
    
    # Simple config for quick test
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=5.0,  # 5 requests per second
        num_prompts=10,    # Just 10 requests for quick test
        dataset_name="random",
        random_input_len=100,
        random_output_len=50,
    )
    
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    metrics_collector = MetricsCollector()
    result_manager = ResultManager(config.output_dir)
    
    # Launch server
    logger.info("Launching server...")
    server_config = config.get_server_config()
    server = await server_manager.launch_server(server_config)
    
    # Generate requests
    requests = request_generator.generate_requests(
        num_prompts=config.num_prompts,
        dataset_name=config.dataset_name,
        random_input_len=config.random_input_len,
        random_output_len=config.random_output_len
    )
    
    # Generate Poisson arrivals
    requests = request_generator.generate_poisson_arrivals(requests, config.request_rate)
    
    # Start collection with incremental saving
    metrics_collector.start_collection()
    test_dir = result_manager.output_dir / "timing_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    metrics_collector.enable_incremental_save(str(test_dir / "results"), interval=5)
    
    api_url = f"http://localhost:{server_config.port}/generate"
    
    # Send requests with proper timing
    async with RequestSender() as sender:
        test_start_time = time.time()
        
        for i, request in enumerate(requests):
            # Calculate absolute arrival time
            absolute_arrival_time = test_start_time + request.arrival_time
            current_time = time.time()
            wait_time = absolute_arrival_time - current_time
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Update arrival_time to absolute time
            request.arrival_time = time.time()
            
            # Record request arrival (enters queue)
            metrics_collector.record_request_arrival(request.request_id)
            
            logger.info(f"Sending request {i+1}/{len(requests)}")
            metrics_collector.record_request_start(request.request_id)
            
            result = await sender.send_request(request, api_url, stream=True)
            metrics_collector.record_request_complete(result)
            
            # Check timing calculations
            if result.success:
                logger.info(f"Request {request.request_id}:")
                logger.info(f"  Queue time: {result.queue_time_ms:.1f} ms")
                logger.info(f"  Server latency: {result.server_latency_ms:.1f} ms")
                logger.info(f"  Total latency: {result.total_latency_ms:.1f} ms")
                
                # Verify timing calculations are reasonable
                assert 0 <= result.queue_time_ms < 10000, f"Queue time out of range: {result.queue_time_ms}"
                assert 0 < result.server_latency_ms < 60000, f"Server latency out of range: {result.server_latency_ms}"
                assert result.total_latency_ms >= result.server_latency_ms, "Total latency should be >= server latency"
    
    # Stop collection
    metrics_collector.stop_collection()
    
    # Save results (should save both JSON and CSV)
    test_name = result_manager.save_results(metrics_collector, config.to_dict(), "timing_test")
    logger.info(f"Results saved to {test_name}")
    
    # Verify CSV was created
    csv_path = Path(test_name) / "results.csv"
    assert csv_path.exists(), "CSV file was not created"
    logger.info("✓ CSV export verified")
    
    # Verify incremental saves
    incremental_csv = test_dir / "results_incremental.csv"
    incremental_json = test_dir / "results_incremental_summary.json"
    assert incremental_csv.exists(), "Incremental CSV was not created"
    assert incremental_json.exists(), "Incremental summary was not created"
    logger.info("✓ Incremental saving verified")
    
    # Print summary
    metrics_collector.print_summary()
    
    # Cleanup
    server_manager.stop_all()
    logger.info("✅ All timing fixes verified successfully!")

if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(test_timing_fixes())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
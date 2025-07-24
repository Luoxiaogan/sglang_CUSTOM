"""Quick test to verify timing fixes are working correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['SGLANG_TEST_LOG_LEVEL'] = 'INFO'

import asyncio
import time
import logging
from sglang_test_framework import NodeConfig, ServerManager, RequestGenerator, RequestSender, MetricsCollector, ResultManager

logger = logging.getLogger(__name__)

async def verify_timing_fixes():
    """Verify that all timing calculations are now correct."""
    logger.info("Starting timing verification test...")
    
    # Simple config
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=10.0,  # 10 requests per second
        num_prompts=20,     # Small test
        dataset_name="random",
        random_input_len=200,
        random_output_len=100,
    )
    
    server_manager = ServerManager()
    request_generator = RequestGenerator(config.tokenizer_path)
    metrics_collector = MetricsCollector()
    result_manager = ResultManager(config.output_dir)
    
    try:
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
        
        # Start collection
        metrics_collector.start_collection()
        
        api_url = f"http://localhost:{server_config.port}/generate"
        
        # Send requests with proper timing
        async with RequestSender() as sender:
            test_start_time = time.time()
            results = []
            
            for i, request in enumerate(requests):
                # Calculate absolute arrival time
                absolute_arrival_time = test_start_time + request.arrival_time
                current_time = time.time()
                wait_time = absolute_arrival_time - current_time
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Update arrival_time to absolute time
                request.arrival_time = time.time()
                
                logger.info(f"Sending request {i+1}/{len(requests)} (ID: {request.request_id})")
                metrics_collector.record_request_start(request.request_id)
                
                result = await sender.send_request(request, api_url, stream=True)
                metrics_collector.record_request_complete(result)
                results.append(result)
                
                # Log timing details for first few requests
                if i < 5 and result.success:
                    logger.info(f"Request {request.request_id} timings:")
                    logger.info(f"  Arrival time: {request.arrival_time:.3f}")
                    logger.info(f"  Send time: {request.send_time:.3f}")
                    logger.info(f"  Completion time: {request.completion_time:.3f}")
                    logger.info(f"  Queue time: {result.queue_time:.3f}s ({result.queue_time_ms:.1f}ms)")
                    logger.info(f"  Server latency: {result.server_latency:.3f}s ({result.server_latency_ms:.1f}ms)")
                    logger.info(f"  Total latency: {result.total_latency:.3f}s ({result.total_latency_ms:.1f}ms)")
                    logger.info(f"  TTFT: {result.ttft:.3f}s ({result.ttft * 1000:.1f}ms)")
        
        # Stop collection
        metrics_collector.stop_collection()
        
        # Save results
        test_name = result_manager.save_results(metrics_collector, config.to_dict(), "timing_verification")
        logger.info(f"Results saved to {test_name}")
        
        # Verify results
        successful_results = [r for r in results if r.success]
        logger.info(f"\n✅ Verification Summary:")
        logger.info(f"  Total requests: {len(results)}")
        logger.info(f"  Successful: {len(successful_results)}")
        
        if successful_results:
            # Check that latencies are reasonable
            server_latencies = [r.server_latency_ms for r in successful_results]
            total_latencies = [r.total_latency_ms for r in successful_results]
            queue_times = [r.queue_time_ms for r in successful_results]
            
            logger.info(f"\n  Server Latency (ms):")
            logger.info(f"    Min: {min(server_latencies):.1f}")
            logger.info(f"    Max: {max(server_latencies):.1f}")
            logger.info(f"    Mean: {sum(server_latencies)/len(server_latencies):.1f}")
            
            logger.info(f"\n  Total Latency (ms):")
            logger.info(f"    Min: {min(total_latencies):.1f}")
            logger.info(f"    Max: {max(total_latencies):.1f}")
            logger.info(f"    Mean: {sum(total_latencies)/len(total_latencies):.1f}")
            
            logger.info(f"\n  Queue Time (ms):")
            logger.info(f"    Min: {min(queue_times):.1f}")
            logger.info(f"    Max: {max(queue_times):.1f}")
            logger.info(f"    Mean: {sum(queue_times)/len(queue_times):.1f}")
            
            # Verify CSV format
            import pandas as pd
            from pathlib import Path
            csv_path = Path(test_name) / "results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"\n  CSV Export Check:")
                logger.info(f"    Rows: {len(df)}")
                logger.info(f"    First arrival time: {df['arrival_time'].iloc[0]:.3f}s (relative)")
                logger.info(f"    Last finish time: {df['finish_time'].iloc[-1]:.3f}s (relative)")
                logger.info(f"    All times are relative: {'✓' if df['arrival_time'].max() < 1000 else '✗'}")
                
                # Show sample row
                logger.info(f"\n  Sample CSV row:")
                sample = df.iloc[0]
                for col in ['req_id', 'arrival_time', 'to_server_time', 'finish_time', 'server_latency', 'total_latency']:
                    logger.info(f"    {col}: {sample[col]}")
            
            # Final verification
            all_valid = all(r.server_latency > 0 for r in successful_results)
            logger.info(f"\n✅ All timing calculations valid: {'YES' if all_valid else 'NO'}")
            
        # Print summary
        metrics_collector.print_summary()
        
    finally:
        # Cleanup
        server_manager.stop_all()
        logger.info("Test completed!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        asyncio.run(verify_timing_fixes())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
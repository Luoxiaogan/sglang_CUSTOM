"""Demo test to show queue length tracking."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import logging
from sglang_test_framework import NodeConfig, ServerManager, RequestGenerator, RequestSender, MetricsCollector, ResultManager

logger = logging.getLogger(__name__)

async def demo_queue_tracking():
    """Demonstrate queue length tracking with high request rate."""
    logger.info("Starting queue tracking demo...")
    
    # Config with high request rate to build up queue
    config = NodeConfig(
        model_path="/data/pretrained_models/Llama-2-7b-hf",
        gpu_id=1,
        max_running_requests=256,
        request_rate=50.0,  # High rate to build queue
        num_prompts=50,
        dataset_name="random",
        random_input_len=500,
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
        
        # Use a small semaphore to demonstrate queue buildup
        max_concurrent_requests = 5  # Small to show queue
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def send_with_semaphore(request, sender):
            """Send request with semaphore control."""
            async with semaphore:
                # Now we're sending - remove from queue
                metrics_collector.record_request_start(request.request_id)
                result = await sender.send_request(request, api_url, stream=True)
                return result
        
        # Create all tasks at their arrival times
        async with RequestSender() as sender:
            test_start_time = time.time()
            tasks = []
            
            # Schedule all requests
            for i, request in enumerate(requests):
                # Calculate absolute arrival time
                absolute_arrival_time = test_start_time + request.arrival_time
                
                async def process_request(req, idx, arrival_time):
                    # Wait until arrival time
                    wait_time = arrival_time - time.time()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Update arrival_time to absolute
                    req.arrival_time = time.time()
                    
                    # Record arrival - enters queue
                    logger.info(f"Request {idx+1}/{len(requests)} arrived (ID: {req.request_id})")
                    metrics_collector.record_request_arrival(req.request_id)
                    
                    # Wait for semaphore and send
                    result = await send_with_semaphore(req, sender)
                    metrics_collector.record_request_complete(result)
                    return result
                
                task = asyncio.create_task(process_request(request, i, absolute_arrival_time))
                tasks.append(task)
            
            # Wait for all to complete
            logger.info(f"All {len(tasks)} requests scheduled. Waiting for completion...")
            results = await asyncio.gather(*tasks)
            
            # Log final stats
            successful = sum(1 for r in results if r.success)
            logger.info(f"Completed: {successful}/{len(results)} successful")
        
        # Stop collection
        metrics_collector.stop_collection()
        
        # Save results
        test_name = result_manager.save_results(metrics_collector, config.to_dict(), "queue_demo")
        logger.info(f"Results saved to {test_name}")
        
        # Create visualizations
        plots = result_manager.create_visualizations(metrics_collector, test_name)
        
        # Print summary
        metrics_collector.print_summary()
        
        # Show queue stats
        metrics = metrics_collector.get_aggregated_metrics()
        logger.info(f"\n✅ Queue Tracking Demo Complete:")
        logger.info(f"  Max Queue Length: {metrics.max_queue_length} requests")
        logger.info(f"  Mean Queue Length: {metrics.mean_queue_length:.1f} requests")
        logger.info(f"  Max Active Requests: {metrics.max_queue_depth}")
        logger.info(f"  Mean Active Requests: {metrics.mean_queue_depth:.1f}")
        
    finally:
        # Cleanup
        server_manager.stop_all()
        logger.info("Demo completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(demo_queue_tracking())
    except KeyboardInterrupt:
        logger.info("Demo interrupted")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
"""Server management for SGLang instances.

Adapted from SGLang's server launch and management patterns.
Source references:
- python/sglang/launch_server.py
- python/sglang/bench_serving_new.py
"""

import asyncio
import subprocess
import time
from typing import List, Dict, Any, Optional
import aiohttp
import logging
import psutil
import signal
import os

from ..config import ServerConfig, RouterConfig

logger = logging.getLogger(__name__)


class SGLangServer:
    """Manages a single SGLang server instance."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.start_time: Optional[float] = None
        
    async def start(self, timeout: int = 300) -> bool:
        """Start the SGLang server."""
        if self.is_running:
            logger.warning(f"Server {self.config.server_id} is already running")
            return True
            
        # Build launch command
        cmd = ["python", "-m", "sglang.launch_server"] + self.config.get_launch_args()[1:]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        
        logger.info(f"Starting server {self.config.server_id} with command: {' '.join(cmd)}")
        
        # Start the process
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.start_time = time.time()
        
        # Wait for server to be ready
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if await self.health_check():
                self.is_running = True
                logger.info(f"Server {self.config.server_id} started successfully in {time.time() - start_wait:.1f}s")
                return True
            
            # Check if process has failed
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Server {self.config.server_id} failed to start")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False
                
            await asyncio.sleep(2)
        
        logger.error(f"Server {self.config.server_id} failed to start within {timeout}s")
        self.stop()
        return False
    
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        url = f"http://{self.config.host}:{self.config.port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def stop(self):
        """Stop the server."""
        if self.process:
            logger.info(f"Stopping server {self.config.server_id}")
            
            # Try graceful shutdown first
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning(f"Force killing server {self.config.server_id}")
                self.process.kill()
                self.process.wait()
                
            self.process = None
            self.is_running = False
    
    async def update_param(self, param: str, value: Any) -> bool:
        """Update a server parameter dynamically.
        
        Note: SGLang does not currently support dynamic parameter updates via API.
        This method is kept for future compatibility when such endpoints are added.
        For now, it will log a warning and return False.
        """
        # TODO: When SGLang adds support for dynamic parameter updates,
        # implement the actual API call here
        logger.warning(f"Dynamic parameter update not yet supported by SGLang for {param}. "
                      f"Would have updated to {value} for server {self.config.server_id}")
        return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        url = f"http://{self.config.host}:{self.config.port}/get_server_info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get metrics: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}


class RouterManager:
    """Manages the SGLang router instance."""
    
    def __init__(self, config: RouterConfig, worker_urls: List[str]):
        self.config = config
        self.worker_urls = worker_urls
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
    async def start(self, timeout: int = 60) -> bool:
        """Start the router."""
        if self.is_running:
            logger.warning("Router is already running")
            return True
            
        # Build launch command
        cmd = ["python", "-m", "sglang_router.launch_router"]
        cmd.extend(["--worker-urls"] + self.worker_urls)
        cmd.extend(self.config.get_launch_args())
        
        logger.info(f"Starting router with command: {' '.join(cmd)}")
        
        # Start the process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for router to be ready
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if await self.health_check():
                self.is_running = True
                logger.info(f"Router started successfully in {time.time() - start_wait:.1f}s")
                return True
                
            # Check if process has failed
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error("Router failed to start")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False
                
            await asyncio.sleep(1)
        
        logger.error(f"Router failed to start within {timeout}s")
        self.stop()
        return False
    
    async def health_check(self) -> bool:
        """Check if the router is healthy."""
        url = f"http://{self.config.host}:{self.config.port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def stop(self):
        """Stop the router."""
        if self.process:
            logger.info("Stopping router")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing router")
                self.process.kill()
                self.process.wait()
                
            self.process = None
            self.is_running = False
    
    async def add_worker(self, url: str) -> bool:
        """Add a worker to the router."""
        api_url = f"http://{self.config.host}:{self.config.port}/add_worker"
        params = {"url": url}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, params=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully added worker: {url}")
                        return True
                    else:
                        logger.error(f"Failed to add worker: {await response.text()}")
                        return False
        except Exception as e:
            logger.error(f"Error adding worker: {e}")
            return False
    
    async def remove_worker(self, url: str) -> bool:
        """Remove a worker from the router."""
        api_url = f"http://{self.config.host}:{self.config.port}/remove_worker"
        params = {"url": url}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, params=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully removed worker: {url}")
                        return True
                    else:
                        logger.error(f"Failed to remove worker: {await response.text()}")
                        return False
        except Exception as e:
            logger.error(f"Error removing worker: {e}")
            return False


class ServerManager:
    """Manages multiple SGLang servers and router."""
    
    def __init__(self):
        self.servers: Dict[str, SGLangServer] = {}
        self.router: Optional[RouterManager] = None
        
    async def launch_server(self, config: ServerConfig) -> SGLangServer:
        """Launch a single server."""
        server = SGLangServer(config)
        if await server.start():
            self.servers[config.server_id] = server
            return server
        else:
            raise RuntimeError(f"Failed to start server {config.server_id}")
    
    async def launch_multiple_servers(self, configs: List[ServerConfig]) -> List[SGLangServer]:
        """Launch multiple servers concurrently."""
        tasks = [self.launch_server(config) for config in configs]
        servers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        failed = []
        successful = []
        for i, result in enumerate(servers):
            if isinstance(result, Exception):
                failed.append((configs[i].server_id, result))
            else:
                successful.append(result)
                
        if failed:
            logger.error(f"Failed to start {len(failed)} servers:")
            for server_id, error in failed:
                logger.error(f"  {server_id}: {error}")
                
        return successful
    
    async def launch_router(self, config: RouterConfig, worker_urls: List[str]) -> RouterManager:
        """Launch the router."""
        router = RouterManager(config, worker_urls)
        if await router.start():
            self.router = router
            return router
        else:
            raise RuntimeError("Failed to start router")
    
    def stop_server(self, server_id: str):
        """Stop a specific server."""
        if server_id in self.servers:
            self.servers[server_id].stop()
            del self.servers[server_id]
        else:
            logger.warning(f"Server {server_id} not found")
    
    def stop_all_servers(self):
        """Stop all servers."""
        for server_id in list(self.servers.keys()):
            self.stop_server(server_id)
            
    def stop_router(self):
        """Stop the router."""
        if self.router:
            self.router.stop()
            self.router = None
            
    def stop_all(self):
        """Stop all servers and router."""
        self.stop_router()
        self.stop_all_servers()
        
    async def update_server_param(self, server_id: str, param: str, value: Any) -> bool:
        """Update a parameter on a specific server."""
        if server_id in self.servers:
            return await self.servers[server_id].update_param(param, value)
        else:
            logger.warning(f"Server {server_id} not found")
            return False
    
    async def update_all_servers_param(self, param: str, value: Any) -> Dict[str, bool]:
        """Update a parameter on all servers."""
        tasks = []
        server_ids = []
        
        for server_id, server in self.servers.items():
            tasks.append(server.update_param(param, value))
            server_ids.append(server_id)
            
        results = await asyncio.gather(*tasks)
        return dict(zip(server_ids, results))
    
    async def get_server_metrics(self, server_id: str) -> Dict[str, Any]:
        """Get metrics from a specific server."""
        if server_id in self.servers:
            return await self.servers[server_id].get_metrics()
        else:
            logger.warning(f"Server {server_id} not found")
            return {}
    
    async def get_all_server_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all servers."""
        tasks = []
        server_ids = []
        
        for server_id, server in self.servers.items():
            tasks.append(server.get_metrics())
            server_ids.append(server_id)
            
        results = await asyncio.gather(*tasks)
        return dict(zip(server_ids, results))
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_all()
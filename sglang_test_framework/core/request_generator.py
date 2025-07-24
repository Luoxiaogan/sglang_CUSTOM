"""Request generation for testing."""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, AsyncGenerator
import numpy as np
from pathlib import Path
import aiohttp
import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Represents a single request."""
    request_id: str
    prompt: str
    prompt_len: int
    output_len: int
    arrival_time: float  # Time when request "arrives" (generated)
    send_time: Optional[float] = None  # Time when request is sent to server
    completion_time: Optional[float] = None  # Time when request completes
    image_data: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class RequestResult:
    """Result of a request execution."""
    request: Request
    success: bool
    generated_text: str = ""
    error: str = ""
    ttft: float = 0.0  # Time to first token
    itl: List[float] = None  # Inter-token latencies
    server_latency: float = 0.0  # From send to completion
    total_latency: float = 0.0  # From arrival to completion
    queue_time: float = 0.0  # From arrival to send
    
    def __post_init__(self):
        if self.itl is None:
            self.itl = []
        
        # Calculate derived metrics
        if self.request.completion_time and self.request.send_time:
            self.server_latency = self.request.completion_time - self.request.send_time
        
        if self.request.completion_time and self.request.arrival_time:
            self.total_latency = self.request.completion_time - self.request.arrival_time
            
        if self.request.send_time and self.request.arrival_time:
            self.queue_time = self.request.send_time - self.request.arrival_time


class RequestGenerator:
    """Generates requests for testing."""
    
    def __init__(self, tokenizer_path: str = None):
        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    def generate_requests(
        self,
        num_prompts: int,
        dataset_name: str = "random",
        dataset_path: str = "",
        random_input_len: int = 1024,
        random_output_len: int = 128,
        random_range_ratio: float = 0.5,
        seed: int = 42
    ) -> List[Request]:
        """Generate a list of requests based on configuration."""
        random.seed(seed)
        np.random.seed(seed)
        
        if dataset_name == "sharegpt":
            return self._generate_sharegpt_requests(num_prompts, dataset_path)
        elif dataset_name == "random":
            return self._generate_random_requests(
                num_prompts, random_input_len, random_output_len, random_range_ratio
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def generate_poisson_arrivals(
        self,
        requests: List[Request],
        request_rate: float,
        start_time: float = None
    ) -> List[Request]:
        """Assign Poisson arrival times to requests."""
        if start_time is None:
            start_time = time.time()
            
        if request_rate == float('inf'):
            # All requests arrive at the same time
            for req in requests:
                req.arrival_time = start_time
        else:
            # Generate Poisson process intervals
            current_time = start_time
            for req in requests:
                interval = np.random.exponential(1.0 / request_rate)
                current_time += interval
                req.arrival_time = current_time
                
        return requests
    
    def generate_length_distribution(
        self,
        num_samples: int,
        distribution_type: str = "normal",
        mean_input: int = 1024,
        mean_output: int = 128,
        variance: float = 10.0,
        range_ratio: float = 0.5
    ) -> Tuple[List[int], List[int]]:
        """Generate input and output length distributions."""
        if distribution_type == "normal":
            # Normal distribution
            input_lengths = np.random.normal(mean_input, np.sqrt(variance), num_samples)
            output_lengths = np.random.normal(mean_output, np.sqrt(variance), num_samples)
        elif distribution_type == "uniform":
            # Uniform distribution within range
            input_min = int(mean_input * (1 - range_ratio))
            input_max = int(mean_input * (1 + range_ratio))
            output_min = int(mean_output * (1 - range_ratio))
            output_max = int(mean_output * (1 + range_ratio))
            
            input_lengths = np.random.uniform(input_min, input_max, num_samples)
            output_lengths = np.random.uniform(output_min, output_max, num_samples)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        # Ensure positive integers
        input_lengths = np.maximum(1, np.round(input_lengths)).astype(int)
        output_lengths = np.maximum(1, np.round(output_lengths)).astype(int)
        
        return input_lengths.tolist(), output_lengths.tolist()
    
    def _generate_random_requests(
        self,
        num_prompts: int,
        input_len: int,
        output_len: int,
        range_ratio: float
    ) -> List[Request]:
        """Generate random requests."""
        input_lens, output_lens = self.generate_length_distribution(
            num_prompts, "uniform", input_len, output_len, 
            range_ratio=range_ratio
        )
        
        requests = []
        for i in range(num_prompts):
            # Generate random text or use placeholder
            prompt = f"Random prompt {i} " + "x" * (input_lens[i] - 20)
            
            request = Request(
                request_id=f"req_{i}",
                prompt=prompt,
                prompt_len=input_lens[i],
                output_len=output_lens[i],
                arrival_time=0.0  # Will be set by generate_poisson_arrivals
            )
            requests.append(request)
            
        return requests
    
    def _generate_sharegpt_requests(
        self,
        num_prompts: int,
        dataset_path: str
    ) -> List[Request]:
        """Generate requests from ShareGPT dataset."""
        # Download ShareGPT if needed
        if not dataset_path:
            dataset_path = self._download_sharegpt()
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Filter conversations with at least 2 turns
        conversations = []
        for data in dataset:
            conv = data.get("conversations", data.get("conversation", []))
            if len(conv) >= 2:
                conversations.append((conv[0]["value"], conv[1]["value"]))
                
        # Sample conversations
        if len(conversations) > num_prompts:
            conversations = random.sample(conversations, num_prompts)
            
        requests = []
        for i, (prompt, completion) in enumerate(conversations):
            # Calculate token lengths if tokenizer available
            if self.tokenizer:
                prompt_len = len(self.tokenizer.encode(prompt))
                output_len = len(self.tokenizer.encode(completion))
            else:
                # Estimate based on character count
                prompt_len = len(prompt) // 4
                output_len = len(completion) // 4
                
            request = Request(
                request_id=f"req_{i}",
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                arrival_time=0.0
            )
            requests.append(request)
            
        return requests
    
    def _download_sharegpt(self) -> str:
        """Download ShareGPT dataset if not available."""
        cache_dir = Path.home() / ".cache" / "sglang" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = cache_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"
        if filepath.exists():
            return str(filepath)
            
        # Download the dataset
        url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        
        import requests
        from tqdm import tqdm
        
        logger.info(f"Downloading ShareGPT dataset to {filepath}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        return str(filepath)


class RequestSender:
    """Sends requests to SGLang servers."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),
            connector=aiohttp.TCPConnector(limit=1000)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_request(
        self,
        request: Request,
        api_url: str,
        stream: bool = True
    ) -> RequestResult:
        """Send a single request to the server."""
        request.send_time = time.time()
        
        # Prepare payload
        payload = {
            "text": request.prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": request.output_len,
                "ignore_eos": True
            },
            "stream": stream
        }
        
        if request.image_data:
            payload["image_data"] = request.image_data
            
        if request.extra_params:
            payload.update(request.extra_params)
            
        try:
            async with self.session.post(api_url, json=payload) as response:
                if response.status == 200:
                    if stream:
                        return await self._handle_stream_response(request, response)
                    else:
                        return await self._handle_non_stream_response(request, response)
                else:
                    error = await response.text()
                    logger.error(f"Request {request.request_id} failed: {error}")
                    return RequestResult(
                        request=request,
                        success=False,
                        error=f"HTTP {response.status}: {error}"
                    )
        except Exception as e:
            logger.error(f"Request {request.request_id} failed with exception: {e}")
            return RequestResult(
                request=request,
                success=False,
                error=str(e)
            )
    
    async def _handle_stream_response(
        self,
        request: Request,
        response: aiohttp.ClientResponse
    ) -> RequestResult:
        """Handle streaming response."""
        result = RequestResult(request=request, success=True)
        
        generated_text = ""
        ttft = 0.0
        last_timestamp = request.send_time
        
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
                
            chunk = chunk_bytes.decode('utf-8')
            if chunk.startswith("data: "):
                chunk = chunk[6:]
                
            if chunk == "[DONE]":
                break
                
            try:
                data = json.loads(chunk)
                
                if "text" in data:
                    current_time = time.time()
                    generated_text = data["text"]
                    
                    # First token
                    if ttft == 0.0:
                        ttft = current_time - request.send_time
                        result.ttft = ttft
                    else:
                        # Inter-token latency
                        result.itl.append(current_time - last_timestamp)
                        
                    last_timestamp = current_time
                    
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse chunk: {chunk}")
                
        request.completion_time = time.time()
        result.generated_text = generated_text
        return result
    
    async def _handle_non_stream_response(
        self,
        request: Request,
        response: aiohttp.ClientResponse
    ) -> RequestResult:
        """Handle non-streaming response."""
        data = await response.json()
        request.completion_time = time.time()
        
        result = RequestResult(
            request=request,
            success=True,
            generated_text=data.get("text", ""),
            ttft=request.completion_time - request.send_time  # No streaming, so TTFT = total time
        )
        
        return result


async def generate_and_send_requests(
    generator: RequestGenerator,
    sender: RequestSender,
    api_url: str,
    config: Dict[str, Any],
    max_concurrency: Optional[int] = None
) -> AsyncGenerator[RequestResult, None]:
    """Generate requests and send them according to Poisson arrival times."""
    # Generate requests
    requests = generator.generate_requests(
        num_prompts=config["num_prompts"],
        dataset_name=config["dataset_name"],
        dataset_path=config.get("dataset_path", ""),
        random_input_len=config.get("random_input_len", 1024),
        random_output_len=config.get("random_output_len", 128),
        random_range_ratio=config.get("random_range_ratio", 0.5),
        seed=config.get("seed", 42)
    )
    
    # Assign arrival times
    requests = generator.generate_poisson_arrivals(
        requests, 
        config["request_rate"]
    )
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def send_with_delay(request: Request):
        """Send request after waiting for its arrival time."""
        # Wait until arrival time
        wait_time = request.arrival_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            
        # Acquire semaphore if needed
        if semaphore:
            async with semaphore:
                return await sender.send_request(request, api_url)
        else:
            return await sender.send_request(request, api_url)
    
    # Create tasks for all requests
    tasks = [asyncio.create_task(send_with_delay(req)) for req in requests]
    
    # Yield results as they complete
    for task in asyncio.as_completed(tasks):
        result = await task
        yield result
#!/usr/bin/env python3
"""
Router startup fix for async context issues.

This script provides a workaround for the tokio runtime error when starting
the sgl-router in an async context.
"""

import asyncio
import subprocess
import sys
import threading
import time
from typing import Optional


def start_router_in_thread(cmd: list, ready_event: threading.Event, error_event: threading.Event, error_msg: list):
    """Start router in a separate thread to avoid async context issues."""
    try:
        # Run router in subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor output
        startup_timeout = time.time() + 60  # 60 second timeout
        
        while time.time() < startup_timeout:
            # Check if process has failed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                error_msg.append(f"Router exited with code {process.returncode}\nstdout: {stdout}\nstderr: {stderr}")
                error_event.set()
                return
            
            # Check for successful startup (you may need to adjust this based on router output)
            time.sleep(0.5)
            
        # If we get here, assume router started successfully
        ready_event.set()
        
        # Keep thread alive while router runs
        process.wait()
        
    except Exception as e:
        error_msg.append(str(e))
        error_event.set()


async def start_router_async(cmd: list) -> Optional[subprocess.Popen]:
    """Start router in a way that avoids async context issues."""
    # Use asyncio.to_thread (Python 3.9+) or run_in_executor
    loop = asyncio.get_event_loop()
    
    try:
        # Start router in a separate process, not affected by current async context
        process = await loop.run_in_executor(
            None,
            subprocess.Popen,
            cmd,
            subprocess.PIPE,
            subprocess.PIPE,
            subprocess.STDOUT,
            None,  # bufsize
            None,  # executable
            None,  # stdin
            None,  # stdout
            None,  # stderr
            None,  # preexec_fn
            False,  # close_fds
            None,  # shell
            None,  # cwd
            None,  # env
            True,  # text mode
        )
        
        # Give router time to start
        await asyncio.sleep(2)
        
        # Check if still running
        if process.poll() is None:
            return process
        else:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Router exited immediately: {stdout}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to start router: {e}")


if __name__ == "__main__":
    # Test the async startup
    async def test():
        cmd = [sys.executable, "-m", "sglang_router.launch_router", "--help"]
        try:
            process = await start_router_async(cmd)
            print("Router started successfully")
            if process:
                process.terminate()
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(test())
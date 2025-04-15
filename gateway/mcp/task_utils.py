"""Task utilities for running async functions"""

import asyncio
import concurrent.futures
import threading

from contextlib import redirect_stdout
from typing import Any

from gateway.mcp.log import MCP_LOG_FILE, mcp_log


def run_task_in_background(async_func, *args, **kwargs):
    """
    Runs an async function in a background thread with its own event loop.
    This function does NOT block the calling thread as it immediately returns
    after starting the background thread.

    Args:
        async_func: The async function to run
        *args: Positional arguments to pass to the async function
        **kwargs: Keyword arguments to pass to the async function
    """

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            mcp_log(
                f"[ERROR] Error in async thread while running run_task_in_background: {e}"
            )
        finally:
            loop.close()

    # Create and start a daemon thread
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()


def run_task_sync(async_func, *args, **kwargs) -> Any:
    """
    Runs an asynchronous function synchronously in a separate
    thread with its own event loop. This function blocks the calling
    thread until completion or timeout (10 seconds).

    Args:
        async_func: The async function to run
        *args: Positional arguments to pass to the async function
        **kwargs: Keyword arguments to pass to the async function

    Returns:
        Any: The return value of the async function
    """

    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                async_func(
                    *args,
                    **kwargs,
                )
            )
        finally:
            loop.close()

    with redirect_stdout(MCP_LOG_FILE):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result(timeout=10.0)

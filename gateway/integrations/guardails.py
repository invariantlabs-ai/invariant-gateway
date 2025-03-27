"""Utility functions for Guardrails execution."""

import asyncio
import os
import time
from typing import Any, Dict, List
from functools import wraps

import httpx

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


# Timestamps of last API calls per guardrails string
_guardrails_cache = {}
# Locks per guardrails string
_guardrails_locks = {}


def rate_limit(expiration_time: int = 3600):
    """
    Decorator to limit API calls to once per expiration_time seconds
    per unique guardrails string.

    Args:
        expiration_time (int): Time in seconds to cache the guardrails.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(guardrails: str, *args, **kwargs):
            now = time.time()

            # Get or create a per-guardrail lock
            if guardrails not in _guardrails_locks:
                _guardrails_locks[guardrails] = asyncio.Lock()
            guardrail_lock = _guardrails_locks[guardrails]

            async with guardrail_lock:
                last_called = _guardrails_cache.get(guardrails)

                if last_called and (now - last_called < expiration_time):
                    # Skipping API call: Guardrails '{guardrails}' already
                    # preloaded within expiration_time
                    return

                # Update cache timestamp
                _guardrails_cache[guardrails] = now

            try:
                await func(guardrails, *args, **kwargs)
            finally:
                _guardrails_locks.pop(guardrails, None)

        return wrapper

    return decorator


@rate_limit(3600)  # Don't preload the same guardrails string more than once per hour
async def _preload(guardrails: str, invariant_authorization: str) -> None:
    """
    Calls the Guardrails API to preload the provided policy for faster checking later.

    Args:
        guardrails (str): The guardrails to preload.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.
    """
    async with httpx.AsyncClient() as client:
        url = os.getenv("GUADRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        result = await client.post(
            f"{url}/api/v1/policy/load",
            json={"policy": guardrails},
            headers={
                "Authorization": invariant_authorization,
                "Accept": "application/json",
            },
        )
        result.raise_for_status()


async def preload_guardrails(context: "RequestContextData") -> None:
    """
    Preloads the guardrails for faster checking later.

    Args:
        context: RequestContextData object.
    """
    if not context.config or not context.config.guardrails:
        return

    try:
        task = asyncio.create_task(
            _preload(context.config.guardrails, context.invariant_authorization)
        )
        asyncio.shield(task)
    except Exception as e:
        print(f"Error scheduling preload_guardrails task: {e}")


class YieldException(Exception):
    """
    Raise this exception in stream instrumentor listeners to
    end the stream early, or to emit additional items in a stream.
    """

    def __init__(self, value, end_of_stream=False):
        super().__init__(value)
        self.value = value
        self.end_of_stream = end_of_stream

    def __str__(self):
        return f"YieldException: {self.value}"


class StreamInstrumentor:
    """
    A class to instrument async iterables with hooks for processing
    chunks, before processing, and on completion.

    Use `@on('chunk')`, `@on('start')`, and `@on('end')` decorators
    to register listeners for different events.

    Listeners can simply process data, or alternatively raise a designated
    YieldException to yield additional values or stop the stream.

    Example usage:

    ```
    instrumentor = StreamInstrumentor()

    @instrumentor.on('chunk')
    async def process_chunk(chunk):
        # Process the chunk
        print(f"Processing chunk: {chunk}")

        if some_condition:
            # Yield an additional value that will be interleaved in the stream
            # Pass `end_of_stream=True` to stop the stream after yielding
            # Pass `end_of_stream=False` to continue the stream after the interleaved value
            raise YieldException("Extra value", end_of_stream=True)
    ```
    """

    def __init__(self):
        # called on every chunk (async)
        self.on_chunk_listeners = []
        # called once before the first chunk is processed, or even earlier (async)
        self.before_listeners = []
        # called once on stream completion (async)
        self.on_complete_listeners = []

        self.stat_token_times = []
        self.stat_before_time = None
        self.stat_after_time = None

        self.stat_first_item_time = None

    # decorator
    def on(self, event: str):
        """
        Decorator to register listeners for different events.

        Args:
            event (str): The event to listen for. Can be 'on_chunk',
                         'before', or 'on_complete'.

        Returns:
            Callable: A decorator to register the listener.
        """

        def decorator(func):
            if event == "chunk":
                if self.on_chunk_listeners is None:
                    self.on_chunk_listeners = []
                self.on_chunk_listeners.append(func)
            elif event == "start":
                if self.before_listeners is None:
                    self.before_listeners = []
                self.before_listeners.append(func)
            elif event == "end":
                if self.on_complete_listeners is None:
                    self.on_complete_listeners = []
                self.on_complete_listeners.append(func)
            else:
                raise ValueError("Invalid event type. Use 'chunk', 'before', or 'end'.")

            return func

        return decorator

    async def stream(self, async_iterable):
        """
        Streams the async iterable and invokes all instrumented hooks.

        Args:
            async_iterable: An async iterable to stream.

        Yields:
            The streamed data.
        """
        try:
            start = time.time()

            # schedule all before listeners which can be run concurrently
            before_tasks = [
                asyncio.create_task(listener()) for listener in self.before_listeners
            ]

            # create async iterator from async_iterable
            aiterable = aiter(async_iterable)

            # [STAT] capture start time of first item
            start_first_item_request = time.time()

            # waits for first item of the iterable
            async def wait_for_first_item():
                nonlocal start_first_item_request, aiterable

                r = await aiterable.__anext__()
                self.stat_first_item_time = time.time() - start_first_item_request
                return r

            next_item_task = asyncio.create_task(wait_for_first_item())

            # wait for all before listeners to finish
            for before_task in before_tasks:
                try:
                    await before_task
                except YieldException as e:
                    # yield extra value before any real items
                    yield e.value
                    # stop the stream if end_of_stream is True
                    if e.end_of_stream:
                        # if first item is already available
                        if not next_item_task.done():
                            # cancel the task
                            next_item_task.cancel()
                            # [STAT] capture time to first item to be now +0.01
                            if self.stat_first_item_time is None:
                                self.stat_first_item_time = (
                                    time.time() - start_first_item_request + 0.01
                                )
                        else:
                            print(
                                "before yields, but next item already ready", flush=True
                            )

            # [STAT] capture before time stamp
            self.stat_before_time = time.time() - start

            while True:
                # wait for first item
                try:
                    item = await next_item_task
                except StopAsyncIteration:
                    break

                # schedule next item
                next_item_task = asyncio.create_task(aiterable.__anext__())

                # [STAT] capture token time stamp
                if len(self.stat_token_times) == 0:
                    self.stat_token_times.append(time.time() - start)
                else:
                    self.stat_token_times.append(
                        time.time() - start - sum(self.stat_token_times)
                    )

                # invoke on_chunk listeners
                for listener in self.on_chunk_listeners:
                    any_end_of_stream = False
                    try:
                        await listener(item)
                    except YieldException as e:
                        yield e.value
                        # if end_of_stream is True, stop the stream
                        if e.end_of_stream:
                            any_end_of_stream = True

                # if end_of_stream is True, stop the stream
                if any_end_of_stream:
                    break

                # yield item
                yield item
            # execute on complete listeners
            on_complete_tasks = [
                asyncio.create_task(listener())
                for listener in self.on_complete_listeners
            ]
            for result in asyncio.as_completed(on_complete_tasks):
                try:
                    await result
                except YieldException as e:
                    # yield extra value before any real items
                    yield e.value
                    # we ignore end_of_stream here, because we are already at the end

            # [STAT] capture after time stamp
            self.stat_after_time = time.time() - start

        finally:
            # [STAT] end all open intervals if not already closed
            if self.stat_after_time is None:
                self.stat_before_time = time.time() - start
            if self.stat_after_time is None:
                self.stat_after_time = 0
            if self.stat_first_item_time is None:
                self.stat_first_item_time = 0

            token_times_5_decimale = str([f"{x:.5f}" for x in self.stat_token_times])
            print(
                f"[STATS]\n [token times: {token_times_5_decimale} ({len(self.stat_token_times)})]"
            )
            print(f" [before:             {self.stat_before_time:.2f}s] ")
            print(f" [time-to-first-item: {self.stat_first_item_time:.2f}s]")
            print(
                f" [zero-latency:       {' TRUE' if self.stat_before_time < self.stat_first_item_time else 'FALSE'}]"
            )
            print(
                f" [extra-latency:      {self.stat_before_time - self.stat_first_item_time:.2f}s]"
            )
            print(f" [after:              {self.stat_after_time:.2f}s]")
            if len(self.stat_token_times) > 0:
                print(
                    f" [average token time: {sum(self.stat_token_times) / len(self.stat_token_times):.2f}s]"
                )
            print(f" [total: {time.time() - start:.2f}s]")


async def check_guardrails(
    messages: List[Dict[str, Any]], guardrails: str, invariant_authorization: str
) -> Dict[str, Any]:
    """
    Checks guardrails on the list of messages.

    Args:
        messages (List[Dict[str, Any]]): List of messages to verify the guardrails against.
        guardrails (str): The guardrails to check against.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        Dict: Response containing guardrail check results.
    """
    async with httpx.AsyncClient() as client:
        url = os.getenv("GUADRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        try:
            result = await client.post(
                f"{url}/api/v1/policy/check",
                json={"messages": messages, "policy": guardrails},
                headers={
                    "Authorization": invariant_authorization,
                    "Accept": "application/json",
                },
            )
            print(f"Guardrail check response: {result.json()}")
            return result.json()
        except Exception as e:
            print(f"Failed to verify guardrails: {e}")
            return {"error": str(e)}

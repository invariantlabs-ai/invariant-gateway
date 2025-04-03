"""Utility functions for Guardrails execution."""

import asyncio
import os
import time
from typing import Any, Dict, List
from functools import wraps

import httpx
from common.guardrails import Guardrail
from common.request_context import RequestContext

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


async def preload_guardrails(context: "RequestContext") -> None:
    """
    Preloads the guardrails for faster checking later.

    Args:
        context: RequestContext object.
    """
    if not context.guardrails:
        return

    try:
        # Move these calls to a batch preload/validate API.
        for blocking_guardrail in context.guardrails.blocking_guardrails:
            task = asyncio.create_task(
                _preload(blocking_guardrail.content, context.invariant_authorization)
            )
            asyncio.shield(task)
        for logging_guadrail in context.guardrails.logging_guardrails:
            task = asyncio.create_task(
                _preload(logging_guadrail.content, context.invariant_authorization)
            )
            asyncio.shield(task)
    except Exception as e:
        print(f"Error scheduling preload_guardrails task: {e}")


class ExtraItem:
    """
    Return this class in a instrumented stream callback, to yield an extra item in the resulting stream.
    """

    def __init__(self, value, end_of_stream=False):
        self.value = value
        self.end_of_stream = end_of_stream

    def __str__(self):
        return f"<ExtraItem value={self.value} end_of_stream={self.end_of_stream}>"


class Replacement(ExtraItem):
    """
    Like ExtraItem, but used to replace the full request result in case of 'InstrumentedResponse'.
    """

    def __init__(self, value):
        super().__init__(value, end_of_stream=True)

    def __str__(self):
        return f"<Replacement value={self.value}>"


class InstrumentedStreamingResponse:
    def __init__(self):
        # request statistics
        self.stat_token_times = []
        self.stat_before_time = None
        self.stat_after_time = None

        self.stat_first_item_time = None

    async def on_chunk(self, chunk: Any) -> ExtraItem | None:
        """
        This called will be called on every chunk (async).
        """
        pass

    async def on_start(self) -> ExtraItem | None:
        """
        Decorator to register a listener for start events.
        """
        pass

    async def on_end(self) -> ExtraItem | None:
        """
        Decorator to register a listener for end events.
        """
        pass

    async def event_generator(self):
        """
        Streams the async iterable and invokes all instrumented hooks.

        Args:
            async_iterable: An async iterable to stream.

        Yields:
            The streamed data.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    async def instrumented_event_generator(self):
        """
        Streams the async iterable and invokes all instrumented hooks.

        Args:
            async_iterable: An async iterable to stream.

        Yields:
            The streamed data.
        """
        try:
            start = time.time()

            # schedule on_start which can be run concurrently
            start_task = asyncio.create_task(self.on_start(), name="instrumentor:start")

            # create async iterator from async_iterable
            aiterable = aiter(self.event_generator())

            # [STAT] capture start time of first item
            start_first_item_request = time.time()

            # waits for first item of the iterable
            async def wait_for_first_item():
                nonlocal start_first_item_request, aiterable

                r = await aiterable.__anext__()
                if self.stat_first_item_time is None:
                    # [STAT] capture time to first item
                    self.stat_first_item_time = time.time() - start_first_item_request
                return r

            next_item_task = asyncio.create_task(
                wait_for_first_item(), name="instrumentor:next:first"
            )

            # check if 'start_task' yields an extra item
            if extra_item := await start_task:
                # yield extra value before any real items
                yield extra_item.value
                # stop the stream if end_of_stream is True
                if extra_item.end_of_stream:
                    # if first item is already available
                    if not next_item_task.done():
                        # cancel the task
                        next_item_task.cancel()
                        # [STAT] capture time to first item to be now +0.01
                        if self.stat_first_item_time is None:
                            self.stat_first_item_time = (
                                time.time() - start_first_item_request
                            ) + 0.01
                    # don't wait for the first item if end_of stream is True
                    return

            # [STAT] capture before time stamp
            self.stat_before_time = time.time() - start

            while True:
                # wait for first item
                try:
                    item = await next_item_task
                except StopAsyncIteration:
                    break

                # schedule next item
                next_item_task = asyncio.create_task(
                    aiterable.__anext__(), name="instrumentor:next"
                )

                # [STAT] capture token time stamp
                if len(self.stat_token_times) == 0:
                    self.stat_token_times.append(time.time() - start)
                else:
                    self.stat_token_times.append(
                        time.time() - start - sum(self.stat_token_times)
                    )

                if extra_item := await self.on_chunk(item):
                    yield extra_item.value
                    # if end_of_stream is True, stop the stream
                    if extra_item.end_of_stream:
                        # cancel next task
                        next_item_task.cancel()
                        return

                # yield item
                yield item

            # run on_end, before closing the stream (may yield an extra value)
            if extra_item := await self.on_end():
                # yield extra value before any real items
                yield extra_item.value
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

            # print statistics
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


class InstrumentedResponse(InstrumentedStreamingResponse):
    """
    A class to instrument an async request with hooks for concurrent
    pre-processing and post-processing (input and output guardrailing).
    """

    async def event_generator(self):
        """
        We implement the 'event_generator' as a single item stream,
        where the item is the full result of the request.
        """
        yield await self.request()

    async def request(self):
        """
        This method should be implemented in a subclass to perform the actual request.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    async def instrumented_request(self):
        """
        Returns the 'Response' object of the request, after applying all instrumented hooks.
        """
        results = [r async for r in self.instrumented_event_generator()]
        assert len(results) >= 1, "InstrumentedResponse must yield at least one item"

        # we return the last item, in case the end callback yields an extra item. Then,
        # don't return the actual result but the 'end' result, e.g. for output guardrailing.
        return results[-1]


async def check_guardrails(
    messages: List[Dict[str, Any]],
    guardrails: List[Guardrail],
    invariant_authorization: str,
) -> Dict[str, Any]:
    """
    Checks guardrails on the list of messages.
    This calls the batch check API of the Guardrails service.

    Args:
        messages (List[Dict[str, Any]]): List of messages to verify the guardrails against.
        guardrails (List[Guardrail]): The guardrails to check against.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        Dict: Response containing guardrail check results.
    """
    async with httpx.AsyncClient() as client:
        url = os.getenv("GUADRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        try:
            result = await client.post(
                f"{url}/api/v1/policy/check/batch",
                json={
                    "messages": messages,
                    "policies": [g.content for g in guardrails],
                },
                headers={
                    "Authorization": invariant_authorization,
                    "Accept": "application/json",
                },
            )
            if not result.is_success:
                raise Exception(
                    f"Guardrails check failed: {result.status_code} - {result.text}"
                )
            guardrails_result = result.json()

            aggregated_errors = {"errors": []}
            for res, guardrail in zip(guardrails_result.get("result", []), guardrails):
                for error in res.get("errors", []):
                    # add each error to the aggregated errors but keep track
                    # of which guardrail it belongs to
                    aggregated_errors["errors"].append(
                        {
                            **error,
                            "guardrail": {
                                "id": guardrail.id,
                                "name": guardrail.name,
                                "content": guardrail.content,
                                "action": guardrail.action,
                            },
                        }
                    )

                # check for any error_message
                if error_message := res.get("error_message"):
                    return {
                        "errors": [
                            {"args": [error_message], "kwargs": {}, "ranges": []}
                        ]
                    }
            return aggregated_errors
        except Exception as e:
            print(f"Failed to verify guardrails: {e}")
            # make sure runtime errors are also visible in e.g. Explorer
            return {
                "errors": [
                    {
                        "args": ["Gateway: " + str(e)],
                        "kwargs": {},
                        "ranges": ["messages[0].content:L0"],
                    }
                ]
            }

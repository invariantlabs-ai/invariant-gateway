"""Instrumentation module for LLM provider routes."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from fastapi import Response

from gateway.routes.base_provider import BaseProvider, ExtraItem
from gateway.common.constants import CONTENT_TYPE_JSON
from gateway.common.guardrails import GuardrailAction
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import (
    push_trace,
    create_annotations_from_guardrails_errors,
)
from gateway.integrations.guardrails import check_guardrails, preload_guardrails


class BaseInstrumentedResponse(ABC):
    """
    Base class for instrumented responses that provides common functionality
    for both streaming and non-streaming responses.
    """

    def __init__(
        self,
        context: RequestContext,
        client: Any,
        provider_request: Any,
        provider: BaseProvider,
        is_streaming: bool,
    ):
        """Configure the instrumented response for a specific provider"""
        self.context = context
        self.client = client
        self.provider_request = provider_request
        self.provider = provider
        self.is_streaming = is_streaming

        # Response tracking
        self.response = None
        self.response_json = None
        self.guardrails_execution_result = {}

        # For streaming: initialize provider-specific response and state
        if is_streaming:
            self.merged_response = provider.initialize_streaming_response()
            self.streaming_state = provider.initialize_streaming_state()

        # request statistics
        self.stat_token_times = []
        self.stat_before_time = None
        self.stat_after_time = None
        self.stat_first_item_time = None

    @abstractmethod
    async def event_generator(self):
        """
        An async iterable that yields events (e.g., chunks of data).
        This method should be implemented by subclasses to provide the actual data source.
        """

    @abstractmethod
    async def on_start(self):
        """
        Pre-processing hook.
        This can be used for input guardrails or other pre-processing tasks.
        """

    @abstractmethod
    async def on_chunk(self, chunk: Any):
        """
        Process a chunk of data.
        This can be used for streaming responses to handle each chunk as it arrives.
        """

    @abstractmethod
    async def on_end(self):
        """
        Post-processing hook.
        This can be used for output guardrails or other post-processing tasks.
        """

    async def check_guardrails_common(
        self, messages: list[dict[str, Any]], action: GuardrailAction
    ) -> dict[str, Any]:
        """Common guardrails checking"""

        guardrails = (
            self.context.guardrails.logging_guardrails
            if action == GuardrailAction.LOG
            else self.context.guardrails.blocking_guardrails
        )

        if not guardrails:
            return {}

        return await check_guardrails(
            messages=messages, guardrails=guardrails, context=self.context
        )

    async def push_to_explorer(
        self, response_json: dict[str, Any], guardrails_result: dict[str, Any] = None
    ) -> None:
        """Common explorer integration"""
        guardrails_result = guardrails_result or {}

        # Create annotations from blocking guardrails errors
        blocking_annotations = create_annotations_from_guardrails_errors(
            guardrails_result.get("errors", [])
        )

        # Execute logging guardrails - provider handles message conversion
        messages = self.provider.combine_messages(
            self.context.request_json, response_json
        )
        logging_result = await self.check_guardrails_common(
            messages, GuardrailAction.LOG
        )
        logging_annotations = create_annotations_from_guardrails_errors(
            logging_result.get("errors", [])
        )

        # Combine all annotations
        all_annotations = blocking_annotations + logging_annotations

        # Create provider-specific metadata
        metadata = self.provider.create_metadata(
            self.context.request_json, response_json
        )

        # Push to explorer
        await push_trace(
            dataset_name=self.context.dataset_name,
            messages=[messages],
            invariant_authorization=self.context.invariant_authorization,
            metadata=[metadata],
            annotations=[all_annotations] if all_annotations else None,
        )

    async def handle_input_guardrails(self) -> Any:
        """Handle input guardrails"""
        if not self.context or not self.context.guardrails:
            return None

        asyncio.create_task(preload_guardrails(self.context))
        response_data = getattr(self, "merged_response", {})
        messages = self.provider.combine_messages(
            self.context.request_json, response_data
        )
        self.guardrails_execution_result = await self.check_guardrails_common(
            messages, GuardrailAction.BLOCK
        )

        if self.guardrails_execution_result.get("errors", []):
            if self.context.dataset_name:
                asyncio.create_task(
                    self.push_to_explorer(
                        response_data, self.guardrails_execution_result
                    )
                )

            if self.is_streaming:
                return self.provider.create_error_chunk(
                    self.guardrails_execution_result, location="request"
                )
            return self.provider.create_non_streaming_error_response(
                self.guardrails_execution_result, location="request"
            )

    async def handle_output_guardrails(self, response_data: dict[str, Any]) -> Any:
        """Handle output guardrails"""
        if not self.context or not self.context.guardrails:
            return None

        messages = self.provider.combine_messages(
            self.context.request_json, response_data
        )
        self.guardrails_execution_result = await self.check_guardrails_common(
            messages, GuardrailAction.BLOCK
        )

        if self.guardrails_execution_result.get("errors", []):
            # Push to explorer
            if self.context.dataset_name:
                print(
                    "Pushing to explorer from inside handle_output_guardrails",
                    self.guardrails_execution_result.get("errors", []),
                    flush=True,
                )
                asyncio.create_task(
                    self.push_to_explorer(
                        response_data, self.guardrails_execution_result
                    )
                )

            if self.is_streaming:
                return self.provider.create_error_chunk(
                    self.guardrails_execution_result,
                    location="response",
                )
            return self.provider.create_non_streaming_error_response(
                self.guardrails_execution_result,
                location="response",
            )

    async def push_trace_to_explorer(self, response_data: dict[str, Any]) -> None:
        """Push trace to explorer if dataset is configured"""
        if self.context.dataset_name:
            should_push = self.provider.should_push_trace(
                response_data,
                bool(self.guardrails_execution_result.get("errors", [])),
            )
            if not should_push:
                return

            asyncio.create_task(
                self.push_to_explorer(response_data, self.guardrails_execution_result)
            )

    async def instrumented_event_generator(self):
        """
        Streams the async iterable and invokes all instrumented hooks.
        Common functionality for both streaming and non-streaming responses.

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


class InstrumentedStreamingResponse(BaseInstrumentedResponse):
    """A class to instrument streaming for LLM provider responses with guardrailing."""

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        provider_request: httpx.Request,
        provider: BaseProvider,
    ):
        super().__init__(context, client, provider_request, provider, is_streaming=True)

    async def on_chunk(self, chunk: Any) -> ExtraItem | None:
        """Process a chunk of streaming data and handle guardrails."""
        # Use provider-specific chunk processing
        self.provider.process_streaming_chunk(
            chunk, self.merged_response, self.streaming_state
        )

        # Check if streaming is complete using provider-specific logic
        chunk_text = chunk.decode("utf-8", errors="replace")
        if (
            self.provider.is_streaming_complete(self.merged_response, chunk_text)
            and self.context.guardrails
        ):
            return await self.handle_output_guardrails(self.merged_response)

    async def on_start(self) -> ExtraItem | None:
        """Run pre-processing before starting the streaming response."""
        return await self.handle_input_guardrails()

    async def on_end(self) -> ExtraItem | None:
        """Run post-processing after the streaming response ends."""
        await self.push_trace_to_explorer(self.merged_response)

    async def event_generator(self):
        """Generic event generator using provider protocol"""
        response = await self.client.send(self.provider_request, stream=True)
        await self.provider.check_error_in_streaming_response(response)

        async for chunk in response.aiter_bytes():
            yield chunk


class InstrumentedResponse(BaseInstrumentedResponse):
    """
    A class to instrument an async request with hooks for concurrent
    pre-processing and post-processing (input and output guardrailing).
    """

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        provider_request: httpx.Request,
        provider: BaseProvider,
    ):
        super().__init__(
            context, client, provider_request, provider, is_streaming=False
        )
        self.response: httpx.Response | None = None
        self.response_json: dict | None = None

    async def on_start(self):
        """Input guardrails"""
        return await self.handle_input_guardrails()

    async def on_chunk(self, _: Any):
        """No-op for non-streaming responses"""
        return None

    async def on_end(self):
        """Output guardrails and explorer integration"""
        if self.response is not None and self.response_json is not None:
            # Check output guardrails
            result = await self.handle_output_guardrails(self.response_json)
            if result:  # If guardrails failed
                return result

            await self.push_trace_to_explorer(self.response_json)

    async def event_generator(self):
        """
        We implement the 'event_generator' as a single item stream,
        where the item is the full result of the request.
        """
        self.response = await self.client.send(self.provider_request)
        self.provider.check_error_in_non_streaming_response(self.response)
        self.response_json = self.response.json()

        response_string = json.dumps(self.response_json)
        updated_headers = dict(self.response.headers)
        updated_headers.pop("content-length", None)

        yield Response(
            content=response_string,
            status_code=self.response.status_code,
            media_type=CONTENT_TYPE_JSON,
            headers=updated_headers,
        )

    async def instrumented_request(self):
        """
        Returns the 'Response' object of the request, after applying all instrumented hooks.
        """
        results = [r async for r in self.instrumented_event_generator()]
        assert len(results) >= 1, "InstrumentedResponse must yield at least one item"

        # we return the last item, in case the end callback yields an extra item. Then,
        # don't return the actual result but the 'end' result, e.g. for output guardrailing.
        return results[-1]

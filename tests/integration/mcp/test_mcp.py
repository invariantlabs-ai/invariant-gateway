"""Test MCP gateway via SSE and stdio transports."""

import os
import uuid

from resources.mcp.sse.client.main import run as mcp_sse_client_run
from resources.mcp.stdio.client.main import run as mcp_stdio_client_run
from utils import create_dataset, add_guardrail_to_dataset

import pytest
import requests

from mcp.shared.exceptions import McpError

MCP_SSE_SERVER_HOST = "mcp-messenger-sse-server"
MCP_SSE_SERVER_PORT = 8123


@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "push_to_explorer, transport",
    [
        (False, "stdio"),
        (False, "sse"),
        (True, "stdio"),
        (True, "sse"),
    ],
)
async def test_mcp_with_gateway(
    explorer_api_url,
    invariant_gateway_package_whl_file,
    gateway_url,
    push_to_explorer,
    transport,
):
    """Test MCP gateway and verify trace is pushed to explorer"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    # Run the MCP client and make the tool call.
    if transport == "sse":
        result = await mcp_sse_client_run(
            gateway_url + "/api/v1/gateway/mcp/sse",
            f"http://{MCP_SSE_SERVER_HOST}:{MCP_SSE_SERVER_PORT}",
            project_name,
            push_to_explorer=push_to_explorer,
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
        )
    else:
        result = await mcp_stdio_client_run(
            invariant_gateway_package_whl_file,
            project_name,
            server_script_path="resources/mcp/stdio/messenger_server/main.py",
            push_to_explorer=push_to_explorer,
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
        )

    assert result.isError is False
    assert (
        result.content[0].type == "text"
        and result.content[0].text == "What is your favorite food?\n"
    )

    if push_to_explorer:
        # Fetch the trace ids for the dataset
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{project_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == 1
        trace_id = traces[0]["id"]

        # Fetch the trace
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}",
            timeout=5,
        )
        trace = trace_response.json()
        metadata = trace["extra_metadata"]
        assert (
            metadata["source"] == "mcp"
            and metadata["mcp_client"] == "mcp"
            and metadata["mcp_server"] == "messenger_server"
        )
        assert trace["messages"][0]["role"] == "assistant"
        assert trace["messages"][0]["tool_calls"][0]["function"] == {
            "name": "get_last_message_from_user",
            "arguments": {"username": "Alice"},
        }
        assert trace["messages"][1]["role"] == "tool"
        assert trace["messages"][1]["content"] == [
            {"type": "text", "text": "What is your favorite food?\n"}
        ]


@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_mcp_with_gateway_and_logging_guardrails(
    explorer_api_url, invariant_gateway_package_whl_file, gateway_url, transport
):
    """Test MCP gateway and verify that logging guardrails work"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    dataset_creation_response = await create_dataset(
        explorer_api_url,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        dataset_name=project_name,
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "food in ToolOutput" if:\n   (tool_output: ToolOutput)\n   (chunk: str) in text(tool_output.content)\n   "food" in chunk',
        action="log",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "get_last_message_from_user is called" if:\n   (tool_call: ToolCall)\n   tool_call is tool:get_last_message_from_user',
        action="log",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    # Run the MCP client and make the tool call.
    if transport == "sse":
        result = await mcp_sse_client_run(
            gateway_url + "/api/v1/gateway/mcp/sse",
            f"http://{MCP_SSE_SERVER_HOST}:{MCP_SSE_SERVER_PORT}",
            project_name,
            push_to_explorer=True,
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
        )
    else:
        result = await mcp_stdio_client_run(
            invariant_gateway_package_whl_file,
            project_name,
            server_script_path="resources/mcp/stdio/messenger_server/main.py",
            push_to_explorer=True,
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
        )

    assert result.isError is False
    assert (
        result.content[0].type == "text"
        and result.content[0].text == "What is your favorite food?\n"
    )

    # Fetch the trace ids for the dataset
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{project_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    # Fetch the trace
    trace_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}",
        timeout=5,
    )
    trace = trace_response.json()
    metadata = trace["extra_metadata"]
    assert (
        metadata["source"] == "mcp"
        and metadata["mcp_client"] == "mcp"
        and metadata["mcp_server"] == "messenger_server"
    )
    assert trace["messages"][0]["role"] == "assistant"
    assert trace["messages"][0]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }
    assert trace["messages"][1]["role"] == "tool"
    assert trace["messages"][1]["content"] == [
        {"type": "text", "text": "What is your favorite food?\n"}
    ]

    # Validate the annotations
    annotations = trace["annotations"]
    food_annotation = None
    tool_call_annotation = None

    assert len(annotations) == 2
    for annotation in annotations:
        if (
            annotation["content"] == "food in ToolOutput"
            and annotation["address"] == "messages.1.content.0.text:22-26"
        ):
            food_annotation = annotation
        elif (
            annotation["content"] == "get_last_message_from_user is called"
            and annotation["address"] == "messages.0.tool_calls.0"
        ):
            tool_call_annotation = annotation
    assert food_annotation is not None, "Missing 'food in ToolOutput' annotation"
    assert (
        tool_call_annotation is not None
    ), "Missing 'get_last_message_from_user is called' annotation"
    assert food_annotation["extra_metadata"]["source"] == "guardrails-error"
    assert tool_call_annotation["extra_metadata"]["source"] == "guardrails-error"


@pytest.mark.asyncio
@pytest.mark.timeout(15)
@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_mcp_with_gateway_and_blocking_guardrails(
    explorer_api_url, invariant_gateway_package_whl_file, gateway_url, transport
):
    """Test MCP gateway and verify that blocking guardrails work"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    dataset_creation_response = await create_dataset(
        explorer_api_url,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        dataset_name=project_name,
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "get_last_message_from_user is called" if:\n   (tool_call: ToolCall)\n   tool_call is tool:get_last_message_from_user',
        action="block",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    # Run the MCP client and make the tool call.
    try:
        if transport == "sse":
            _ = await mcp_sse_client_run(
                gateway_url + "/api/v1/gateway/mcp/sse",
                f"http://{MCP_SSE_SERVER_HOST}:{MCP_SSE_SERVER_PORT}",
                project_name,
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        else:
            _ = await mcp_stdio_client_run(
                invariant_gateway_package_whl_file,
                project_name,
                server_script_path="resources/mcp/stdio/messenger_server/main.py",
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        # If we get here, the tool call was not blocked
        pytest.fail("Expected McpError to be raised")
    # The tool call should be blocked by the guardrail
    # and an error should be raised.
    except McpError as e:
        assert (
            "[Invariant Guardrails] The MCP tool call was blocked for security reasons"
            in e.error.message
        )
        assert "get_last_message_from_user is called" in e.error.message
        assert e.error.code == -32600

    # Fetch the trace ids for the dataset
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{project_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    # Fetch the trace
    trace_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}",
        timeout=5,
    )
    trace = trace_response.json()
    metadata = trace["extra_metadata"]
    assert (
        metadata["source"] == "mcp"
        and metadata["mcp_client"] == "mcp"
        and metadata["mcp_server"] == "messenger_server"
    )
    assert trace["messages"][0]["role"] == "assistant"
    assert trace["messages"][0]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }

    # Validate the annotations
    annotations = trace["annotations"]
    assert len(annotations) == 1
    assert (
        annotations[0]["content"] == "get_last_message_from_user is called"
        and annotations[0]["address"] == "messages.0.tool_calls.0"
    )
    assert annotations[0]["extra_metadata"]["source"] == "guardrails-error"


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_mcp_sse_with_gateway_hybrid_guardrails(
    explorer_api_url, invariant_gateway_package_whl_file, gateway_url, transport
):
    """Test MCP gateway and verify that logging and blocking guardrails work together"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    dataset_creation_response = await create_dataset(
        explorer_api_url,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        dataset_name=project_name,
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "get_last_message_from_user is called" if:\n   (tool_call: ToolCall)\n   tool_call is tool:get_last_message_from_user',
        action="log",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "food in ToolOutput" if:\n   (tool_output: ToolOutput)\n   (chunk: str) in text(tool_output.content)\n   "food" in chunk',
        action="block",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    # Run the MCP client and make the tool call.
    try:
        if transport == "sse":
            _ = await mcp_sse_client_run(
                gateway_url + "/api/v1/gateway/mcp/sse",
                f"http://{MCP_SSE_SERVER_HOST}:{MCP_SSE_SERVER_PORT}",
                project_name,
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        else:
            _ = await mcp_stdio_client_run(
                invariant_gateway_package_whl_file,
                project_name,
                server_script_path="resources/mcp/stdio/messenger_server/main.py",
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        # If we get here, the tool call was not blocked
        pytest.fail("Expected McpError to be raised")
    # The tool call output should be blocked by the guardrail
    # and an error should be raised.
    except McpError as e:
        assert (
            "[Invariant Guardrails] The MCP tool call was blocked for security reasons"
            in e.error.message
        )
        assert "food in ToolOutput" in e.error.message
        assert e.error.code == -32600

    # Fetch the trace ids for the dataset
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{project_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    # Fetch the trace
    trace_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}",
        timeout=5,
    )
    trace = trace_response.json()
    metadata = trace["extra_metadata"]
    assert (
        metadata["source"] == "mcp"
        and metadata["mcp_client"] == "mcp"
        and metadata["mcp_server"] == "messenger_server"
    )
    assert trace["messages"][0]["role"] == "assistant"
    assert trace["messages"][0]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }
    assert trace["messages"][1]["role"] == "tool"
    assert trace["messages"][1]["content"] == [
        {"type": "text", "text": "What is your favorite food?\n"}
    ]

    # Validate the annotations
    annotations = trace["annotations"]
    assert len(annotations) == 2
    assert (
        annotations[0]["content"] == "get_last_message_from_user is called"
        and annotations[0]["address"] == "messages.0.tool_calls.0"
    )
    assert annotations[0]["extra_metadata"]["source"] == "guardrails-error"
    assert (
        annotations[1]["content"] == "food in ToolOutput"
        and annotations[1]["address"] == "messages.1.content.0.text:22-26"
    )
    assert annotations[1]["extra_metadata"]["source"] == "guardrails-error"

"""Test MCP gateway via stdio."""

import os
import uuid

import pytest
import requests

from mcp.shared.exceptions import McpError
from utils import create_dataset, add_guardrail_to_dataset

from resources.mcp.stdio.client.main import run as mcp_client_run


@pytest.mark.asyncio
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_mcp_stdio_with_gateway(
    explorer_api_url, invariant_gateway_package_whl_file, push_to_explorer
):
    """Test MCP gateway via stdio and verify trace is pushed to explorer"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    # Run the MCP client and make the tool call.
    result = await mcp_client_run(
        invariant_gateway_package_whl_file,
        project_name,
        server_script_path="resources/mcp/stdio/messenger_server/main.py",
        push_to_explorer=push_to_explorer,
        tool_name="get_last_message_from_user",
        tool_args={"username": "Alice"},
        metadata_keys={"my-custom-key": "value1", "my-custom-key-2": "value2"},
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
        # ensure custom keys are present
        assert metadata["my-custom-key"] == "value1"
        assert metadata["my-custom-key-2"] == "value2"

        assert trace["messages"][2]["role"] == "assistant"
        assert trace["messages"][2]["tool_calls"][0]["function"] == {
            "name": "get_last_message_from_user",
            "arguments": {"username": "Alice"},
        }
        assert trace["messages"][3]["role"] == "tool"
        assert trace["messages"][3]["content"] == [
            {"type": "text", "text": "What is your favorite food?\n"}
        ]


@pytest.mark.asyncio
async def test_mcp_stdio_with_gateway_and_logging_guardrails(
    explorer_api_url, invariant_gateway_package_whl_file
):
    """Test MCP gateway via stdio and verify that logging guardrails work"""
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
    result = await mcp_client_run(
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
    assert trace["messages"][2]["role"] == "assistant"
    assert trace["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }
    assert trace["messages"][3]["role"] == "tool"
    assert trace["messages"][3]["content"] == [
        {"type": "text", "text": "What is your favorite food?\n"}
    ]

    # Fetch annotations
    annotations_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
        timeout=5,
    )
    annotations = annotations_response.json()
    food_annotation = None
    tool_call_annotation = None

    assert len(annotations) == 2
    for annotation in annotations:
        if (
            annotation["content"] == "food in ToolOutput"
            and annotation["address"] == "messages.3.content.0.text:22-26"
        ):
            food_annotation = annotation
        elif (
            annotation["content"] == "get_last_message_from_user is called"
            and annotation["address"] == "messages.2.tool_calls.0"
        ):
            tool_call_annotation = annotation
    assert food_annotation is not None, "Missing 'food in ToolOutput' annotation"
    assert (
        tool_call_annotation is not None
    ), "Missing 'get_last_message_from_user is called' annotation"
    assert food_annotation["extra_metadata"]["source"] == "guardrails-error"
    assert tool_call_annotation["extra_metadata"]["source"] == "guardrails-error"


@pytest.mark.asyncio
async def test_mcp_stdio_with_gateway_and_blocking_guardrails(
    explorer_api_url, invariant_gateway_package_whl_file
):
    """Test MCP gateway via stdio and verify that blocking guardrails work"""
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
        _ = await mcp_client_run(
            invariant_gateway_package_whl_file,
            project_name,
            server_script_path="resources/mcp/stdio/messenger_server/main.py",
            push_to_explorer=True,
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
        )
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
    assert trace["messages"][2]["role"] == "assistant"
    assert trace["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }

    # Fetch annotations
    annotations_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
        timeout=5,
    )
    annotations = annotations_response.json()
    assert len(annotations) == 1
    assert (
        annotations[0]["content"] == "get_last_message_from_user is called"
        and annotations[0]["address"] == "messages.2.tool_calls.0"
    )
    assert annotations[0]["extra_metadata"]["source"] == "guardrails-error"

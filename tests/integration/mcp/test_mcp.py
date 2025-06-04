"""Test MCP gateway via SSE and stdio transports."""

import os
import uuid
from resources.mcp.sse.client.main import run as mcp_sse_client_run
from resources.mcp.stdio.client.main import run as mcp_stdio_client_run
from resources.mcp.streamable.client.main import run as mcp_streamable_client_run
from utils import create_dataset, add_guardrail_to_dataset

import httpx
import pytest
import requests

# Taken from docker-compose.test.yml
MCP_SSE_SERVER_HOST = "mcp-messenger-sse-server"
MCP_SSE_SERVER_PORT = 8123
MCP_STREAMABLE_HOSTS = {
    "streamable-json-stateless": {
        "host": "mcp-messenger-streamable-json-stateless-server",
        "port": 8124,
    },
    "streamable-json-stateful": {
        "host": "mcp-messenger-streamable-json-stateful-server",
        "port": 8125,
    },
    "streamable-sse-stateless": {
        "host": "mcp-messenger-streamable-sse-stateless-server",
        "port": 8126,
    },
    "streamable-sse-stateful": {
        "host": "mcp-messenger-streamable-sse-stateful-server",
        "port": 8127,
    },
}


def _get_mcp_sse_server_base_url() -> str:
    return f"http://{MCP_SSE_SERVER_HOST}:{MCP_SSE_SERVER_PORT}"


def _get_streamable_server_base_url(transport: str) -> str:
    if transport not in MCP_STREAMABLE_HOSTS:
        raise ValueError(f"Unknown transport: {transport}")
    host_info = MCP_STREAMABLE_HOSTS[transport]
    return f"http://{host_info['host']}:{host_info['port']}"


def _get_server_base_url(transport: str) -> str:
    if transport == "sse":
        return _get_mcp_sse_server_base_url()
    elif transport.startswith("streamable-"):
        return _get_streamable_server_base_url(transport)
    else:
        raise ValueError(f"Unknown transport: {transport}")


def _get_headers(
    server_base_url: str, project_name: str, push_to_explorer: bool = True
) -> dict[str, str]:
    return {
        "MCP-SERVER-BASE-URL": server_base_url,
        "INVARIANT-PROJECT-NAME": project_name,
        "PUSH-INVARIANT-EXPLORER": str(push_to_explorer),
    }


async def _invoke_mcp_tool(
    transport, gateway_url, project_name, tool_name, tool_args, whl=None, push=True
):
    if transport == "stdio":
        return await mcp_stdio_client_run(
            whl,
            project_name,
            "resources/mcp/stdio/messenger_server/main.py",
            push,
            tool_name,
            tool_args,
        )
    elif transport == "sse":
        return await mcp_sse_client_run(
            f"{gateway_url}/api/v1/gateway/mcp/sse",
            tool_name,
            tool_args,
            headers=_get_headers(_get_server_base_url(transport), project_name, push),
        )
    return await mcp_streamable_client_run(
        f"{gateway_url}/api/v1/gateway/mcp/streamable",
        tool_name,
        tool_args,
        headers=_get_headers(_get_server_base_url(transport), project_name, push),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "transport",
    [
        "stdio",
        "sse",
        "streamable-json-stateless",
        "streamable-json-stateful",
        "streamable-sse-stateless",
        "streamable-sse-stateful",
    ],
)
async def test_mcp_with_gateway(
    explorer_api_url,
    invariant_gateway_package_whl_file,
    gateway_url,
    transport,
):
    """Test MCP gateway and verify trace is pushed to explorer"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    # Run the MCP client and make the tool call.
    result = await _invoke_mcp_tool(
        transport,
        gateway_url,
        project_name,
        tool_name="get_last_message_from_user",
        tool_args={"username": "Alice"},
        whl=invariant_gateway_package_whl_file,
        push=True,
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
    if transport == "streamable-json-stateless":
        assert metadata["server_response_type"] == "json"
        assert metadata["is_stateless_http_server"] is True
    elif transport == "streamable-json-stateful":
        assert metadata["server_response_type"] == "json"
        assert metadata["is_stateless_http_server"] is False
    elif transport == "streamable-sse-stateless":
        assert metadata["server_response_type"] == "sse"
        assert metadata["is_stateless_http_server"] is True
    elif transport == "streamable-sse-stateful":
        assert metadata["server_response_type"] == "sse"
        assert metadata["is_stateless_http_server"] is False

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
@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "transport",
    [
        "stdio",
        "sse",
        "streamable-json-stateless",
        "streamable-json-stateful",
        "streamable-sse-stateless",
        "streamable-sse-stateful",
    ],
)
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
    result = await _invoke_mcp_tool(
        transport,
        gateway_url,
        project_name,
        tool_name="get_last_message_from_user",
        tool_args={"username": "Alice"},
        whl=invariant_gateway_package_whl_file,
        push=True,
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
    assert "session_id" in metadata
    assert "system_user" in metadata
    if transport == "streamable-json-stateless":
        assert metadata["server_response_type"] == "json"
        assert metadata["is_stateless_http_server"] is True
    elif transport == "streamable-json-stateful":
        assert metadata["server_response_type"] == "json"
        assert metadata["is_stateless_http_server"] is False
    elif transport == "streamable-sse-stateless":
        assert metadata["server_response_type"] == "sse"
        assert metadata["is_stateless_http_server"] is True
    elif transport == "streamable-sse-stateful":
        assert metadata["server_response_type"] == "sse"
        assert metadata["is_stateless_http_server"] is False

    assert trace["messages"][2]["role"] == "assistant"
    assert trace["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }
    assert trace["messages"][3]["role"] == "tool"
    assert trace["messages"][3]["content"] == [
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
    assert food_annotation["extra_metadata"]["guardrail"]["action"] == "log"
    assert tool_call_annotation["extra_metadata"]["source"] == "guardrails-error"
    assert tool_call_annotation["extra_metadata"]["guardrail"]["action"] == "log"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "transport",
    [
        "stdio",
        "sse",
        "streamable-json-stateless",
        "streamable-json-stateful",
        "streamable-sse-stateless",
        "streamable-sse-stateful",
    ],
)
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

    with pytest.raises(ExceptionGroup) as exc_group:
        if transport == "sse":
            _ = await mcp_sse_client_run(
                gateway_url + "/api/v1/gateway/mcp/sse",
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
                headers=_get_headers(
                    _get_mcp_sse_server_base_url(), project_name, True
                ),
            )
        elif transport == "stdio":
            _ = await mcp_stdio_client_run(
                invariant_gateway_package_whl_file,
                project_name,
                server_script_path="resources/mcp/stdio/messenger_server/main.py",
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        else:
            _ = await mcp_streamable_client_run(
                gateway_url + "/api/v1/gateway/mcp/streamable",
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
                headers=_get_headers(
                    _get_streamable_server_base_url(transport), project_name, True
                ),
            )
    if transport.startswith("streamable-"):
        # Extract the actual HTTPStatusError
        http_errors = [
            e
            for e in exc_group.value.exceptions
            if isinstance(e, httpx.HTTPStatusError)
        ]
        assert http_errors[0].response.status_code == 400
    else:
        mcp_error = [e for e in exc_group.value.exceptions][0].exceptions[0]
        assert (
            "[Invariant Guardrails] The MCP tool call was blocked for security reasons"
            in mcp_error.error.message
        )
        assert "get_last_message_from_user is called" in mcp_error.error.message
        assert -32600 == mcp_error.error.code

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
    assert "session_id" in metadata
    assert "system_user" in metadata
    assert trace["messages"][2]["role"] == "assistant"
    assert trace["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }

    # Validate the annotations
    annotations = trace["annotations"]
    assert len(annotations) == 1
    assert (
        annotations[0]["content"] == "get_last_message_from_user is called"
        and annotations[0]["address"] == "messages.2.tool_calls.0"
    )
    assert annotations[0]["extra_metadata"]["source"] == "guardrails-error"
    assert annotations[0]["extra_metadata"]["guardrail"]["action"] == "block"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "transport",
    [
        "stdio",
        "sse",
        "streamable-json-stateless",
        "streamable-json-stateful",
        "streamable-sse-stateless",
        "streamable-sse-stateful",
    ],
)
async def test_mcp_with_gateway_hybrid_guardrails(
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

    with pytest.raises(ExceptionGroup) as exc_group:
        if transport == "sse":
            _ = await mcp_sse_client_run(
                gateway_url + "/api/v1/gateway/mcp/sse",
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
                headers=_get_headers(
                    _get_mcp_sse_server_base_url(), project_name, True
                ),
            )
        elif transport == "stdio":
            _ = await mcp_stdio_client_run(
                invariant_gateway_package_whl_file,
                project_name,
                server_script_path="resources/mcp/stdio/messenger_server/main.py",
                push_to_explorer=True,
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
            )
        else:
            _ = await mcp_streamable_client_run(
                gateway_url + "/api/v1/gateway/mcp/streamable",
                tool_name="get_last_message_from_user",
                tool_args={"username": "Alice"},
                headers=_get_headers(
                    _get_streamable_server_base_url(transport), project_name, True
                ),
            )
    if transport.startswith("streamable-json"):
        # Extract the actual HTTPStatusError
        http_errors = [
            e
            for e in exc_group.value.exceptions
            if isinstance(e, httpx.HTTPStatusError)
        ]
        assert http_errors[0].response.status_code == 400
    else:
        mcp_error = [e for e in exc_group.value.exceptions][0].exceptions[0]
        assert (
            "[Invariant Guardrails] The MCP tool call was blocked for security reasons"
            in mcp_error.error.message
        )
        assert "food in ToolOutput" in mcp_error.error.message
        assert -32600 == mcp_error.error.code

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
    assert "session_id" in metadata
    assert "system_user" in metadata
    assert trace["messages"][2]["role"] == "assistant"
    assert trace["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_last_message_from_user",
        "arguments": {"username": "Alice"},
    }
    assert trace["messages"][3]["role"] == "tool"
    assert trace["messages"][3]["content"] == [
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
    assert food_annotation["extra_metadata"]["guardrail"]["action"] == "block"
    assert tool_call_annotation["extra_metadata"]["source"] == "guardrails-error"
    assert tool_call_annotation["extra_metadata"]["guardrail"]["action"] == "log"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "transport",
    [
        "stdio",
        "sse",
        "streamable-json-stateless",
        "streamable-json-stateful",
        "streamable-sse-stateless",
        "streamable-sse-stateful",
    ],
)
async def test_mcp_tool_list_blocking(
    explorer_api_url, invariant_gateway_package_whl_file, gateway_url, transport
):
    """
    Tests that blocking guardrails work for the tools/list call.

    For those, the expected behavior is that the returned tools are all renamed to blocked_... and include an informative block notice, instead of the original tool description.
    """
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
        policy='raise "get_last_message_from_user is called" if:\n   (tool_output: ToolOutput)\n   tool_call(tool_output).function.name == "tools/list"',
        action="block",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    if transport.startswith("streamable-json"):
        with pytest.raises(ExceptionGroup) as exc_group:
            _ = await mcp_streamable_client_run(
                gateway_url + "/api/v1/gateway/mcp/streamable",
                tool_name="tools/list",
                tool_args={},
                headers=_get_headers(
                    _get_streamable_server_base_url(transport), project_name, True
                ),
            )
        # Extract the actual HTTPStatusError
        http_errors = [
            e
            for e in exc_group.value.exceptions
            if isinstance(e, httpx.HTTPStatusError)
        ]
        assert http_errors[0].response.status_code == 400
        return

    # Run the MCP client and make the tools/list call.
    # Run the MCP client and make the tool call.
    tools_result = await _invoke_mcp_tool(
        transport,
        gateway_url,
        project_name,
        tool_name="tools/list",
        tool_args={},
        whl=invariant_gateway_package_whl_file,
        push=True,
    )
    assert "blocked_get_last_message_from_user" in str(tools_result), (
        "Expected the tool names to be renamed and blocked because of the blocking guardrail on the tools/list call. Instead got: "
        + str(tools_result)
    )


@pytest.mark.asyncio
async def test_mcp_sse_post_endpoint_exceptions(gateway_url):
    """
    Tests that the SSE POST endpoint returns the correct error messages for various exceptions.
    """
    # Test missing session_id query parameter
    response = requests.post(
        f"{gateway_url}/api/v1/gateway/mcp/sse/messages/",
        timeout=5,
    )
    assert response.status_code == 400
    assert "Missing 'session_id' query parameter" in response.text

    # Test unknown session_id in query parameter
    response = requests.post(
        f"{gateway_url}/api/v1/gateway/mcp/sse/messages/?session_id=session_id_1",
        timeout=5,
    )
    assert response.status_code == 400
    assert "Session does not exist" in response.text

    # Test missing mcp-server-base-url header
    with pytest.raises(ExceptionGroup) as exc_group:
        await mcp_sse_client_run(
            gateway_url + "/api/v1/gateway/mcp/sse",
            tool_name="get_last_message_from_user",
            tool_args={"username": "Alice"},
            headers={
                "INVARIANT-PROJECT-NAME": "something-123",
                "PUSH-INVARIANT-EXPLORER": "True",
            },
        )

    # Extract the actual HTTPStatusError
    http_errors = [
        e for e in exc_group.value.exceptions if isinstance(e, httpx.HTTPStatusError)
    ]
    assert http_errors[0].response.status_code == 400

"""Test MCP gateway via stdio."""

import uuid

import pytest
import requests


from resources.mcp.client.main import run as mcp_client_run


@pytest.mark.asyncio
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_mcp_stdio_with_gateway(
    explorer_api_url, invariant_gateway_package_whl_file, push_to_explorer
):
    """Test MCP gateway via stdio and verify trace is pushed to explorer"""
    project_name = "test-mcp-" + str(uuid.uuid4())

    # Run the MCP client and get the result
    result = await mcp_client_run(
        invariant_gateway_package_whl_file,
        project_name,
        server_script_path="resources/mcp/messenger_server/main.py",
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

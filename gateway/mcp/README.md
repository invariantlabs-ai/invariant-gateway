## MCP Integration with Invariant Gateway

This is a work-in-progress implementation of the Model Context Protocol (MCP) integrated with the Invariant Gateway.


### Original MCP Config (Baseline)

Given a standard MCP configuration like:

```
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "weather.py"
      ]
    }
  }
}
```

### Using the PyPI Package

To enable runtime guardrails and trace logging via the Invariant Gateway, modify your MCP config as follows:

```
{
  "mcpServers": {
    "weather": {
      "command": "uvx",
      "args": [
        "invariant-gateway@latest",
        "mcp",
        "--project-name",
        "<your-project-name>",
        "--push-explorer",
        "--exec",
        "uv",
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "weather.py"
      ],
      "env": {
        "INVARIANT_API_KEY": "<Add Invariant API key here>"
      }
    }
  }
}
```

Explanation

This configuration wraps the original MCP server invocation with the invariant-gateway CLI.

*	Arguments before --exec are handled by the Invariant Gateway.

*	Arguments after --exec are passed directly to your MCP server.

Key Parameters

*	--project-name: The name of the Invariant Explorer project (dataset). Guardrails will be fetched from this project.

*	--push-explorer: Enables pushing annotated traces of MCP message flows to the specified project in Invariant Explorer.

### Local Development Workflow

To test or develop the gateway locally:

1.	Clone the invariant-gateway repository.

2.	Build the package:

```bash
python -m build
```

This will generate a .whl file under the dist/ directory.

3.	Update your MCP config to reference the local build (replace <VERSION> with the version specified in pyproject.toml):

```
{
  "mcpServers": {
    "weather": {
      "command": "uvx",
      "args": [
        "--refresh",
        "--from",
        "/ABSOLUTE/PATH/TO/INVARIANT_GATEWAY_REPO/dist/invariant_gateway-<VERSION>-py3-none-any.whl",
        "invariant-gateway",
        "mcp",
        "--project-name",
        "<your-project-name>",
        "--push-explorer",
        "--exec",
        "uv",
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "weather.py"
      ],
      "env": {
        "INVARIANT_API_KEY": "<Add Invariant API key here>"
      }
    }
  }
}
```

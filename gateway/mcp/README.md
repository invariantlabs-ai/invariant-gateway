This is a work in progress implementation for MCP (Model Context Protocol) with the Gateway.

This repository will be pushed to PyPi and then using it with the MCP config file will be simpler.

For now if the original MCP config file looks like:

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

You need to:

1. Checkout the invariant-gatway repo.
2. Run `python -m build`. This will generate a .whl file in dist.
3. Modify the MCP config like this:

```
  {
    "mcpServers": {
      "weather": {
        "command": "uvx",
        "args": [
          "--refresh",
          "--from",
          "/ABSOLUTE/PATH/TO/INVARIANT_GATEWAY_REPO/dist/invariant_gateway-0.0.1-py3-none-any.whl",
          "run",
          "mcp",
          "--project-name",
          "weather-testing",
          "--push-explorer",
          "--exec",
          "uv",
          "--directory",
          "/Users/hemang/Sdk/mcp/weather",
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

This moves the original `command` and `args` to the `args` list after the `--exec` flag.

All args before the `--exec` flag are relevant to the Invariant MCP gateway. These include:

- `--project-name`: With this you can specify the name of the Invariant Explorer project (dataset). The guardrails are pulled from this.
- `--push-explorer`: With this you can specify if you want to push the annotated traces to the Invariant Explorer. The annotated traces are pushed to the project name provided above.

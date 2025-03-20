"""Converts the request and response formats from Anthropic to Invariant API format."""


def convert_anthropic_to_invariant_message_format(
    messages: list[dict], keep_empty_tool_response: bool = False
) -> list[dict]:
    """Converts a list of messages from the Anthropic API to the Invariant API format."""
    output = []
    role_mapping = {
        "system": lambda msg: {"role": "system", "content": msg["content"]},
        "user": lambda msg: handle_user_message(msg, keep_empty_tool_response),
        "assistant": handle_assistant_message,
    }

    for message in messages:
        handler = role_mapping.get(message["role"])
        if handler:
            result = handler(message)
            if isinstance(result, list):
                output.extend(result)
            else:
                output.append(result)

    return output


def handle_user_message(message, keep_empty_tool_response):
    """Handle the user message from the Anthropic API"""
    output = []
    content = message["content"]
    if isinstance(content, list):
        user_content = []
        for sub_message in content:
            if sub_message["type"] == "tool_result":
                if sub_message["content"]:
                    output.append(
                        {
                            "role": "tool",
                            "content": sub_message["content"],
                            "tool_call_id": sub_message["tool_use_id"],
                        }
                    )
                elif keep_empty_tool_response and any(sub_message.values()):
                    output.append(
                        {
                            "role": "tool",
                            "content": {"is_error": True}
                            if sub_message["is_error"]
                            else {},
                            "tool_call_id": sub_message["tool_use_id"],
                        }
                    )
            elif sub_message["type"] == "text":
                user_content.append({"type": "text", "text": sub_message["text"]})
            elif sub_message["type"] == "image":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:"
                            + sub_message["source"]["media_type"]
                            + ";base64,"
                            + sub_message["source"]["data"],
                        },
                    },
                )
        if user_content:
            output.append({"role": "user", "content": user_content})
    else:
        output.append({"role": "user", "content": content})
    return output


def handle_assistant_message(message):
    """Handle the assistant message from the Anthropic API"""
    output = []
    for sub_message in message["content"]:
        if sub_message["type"] == "text":
            output.append({"role": "assistant", "content": sub_message.get("text")})
        elif sub_message["type"] == "tool_use":
            output.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": sub_message.get("id"),
                            "type": "function",
                            "function": {
                                "name": sub_message.get("name"),
                                "arguments": sub_message.get("input"),
                            },
                        }
                    ],
                }
            )
    return output

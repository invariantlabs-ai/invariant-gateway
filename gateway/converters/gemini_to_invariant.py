"""Converts the request and response formats from Gemini to Invariant API format."""

def convert_request(request: dict) -> list[dict]:
    """Converts the request from Gemini API to Invariant API format."""
    openai_messages = []

    if "systemInstruction" in request:
        system_content = " ".join(
            part.get("text", "")
            for part in request["systemInstruction"].get("parts", [])
        )
        openai_messages.append({"role": "system", "content": system_content})

    if "contents" in request:
        for content in request["contents"]:
            role = content.get("role", "")

            if role == "user":
                message_content = []
                for part in content.get("parts", []):
                    if "text" in part:
                        message_content.append({"type": "text", "text": part["text"]})
                    elif "inlineData" in part:
                        message_content.append(
                            {
                                "type": "image",
                                "image_url": {
                                    "url": f"data:{part['inlineData']['mime_type']};base64,{part['inlineData']['data']}"
                                },
                            }
                        )
                    elif "functionResponse" in part:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_name": part["functionResponse"]["name"],
                                "content": part["functionResponse"]["response"].get(
                                    "result", {}
                                ),
                            }
                        )
                if message_content:
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": message_content
                            if len(message_content) > 1
                            else message_content[0],
                        }
                    )

            elif role == "model":
                for part in content.get("parts", []):
                    if "text" in part:
                        openai_messages.append(
                            {"role": "assistant", "content": part["text"]}
                        )
                    elif "functionCall" in part:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": part["functionCall"]["name"],
                                            "arguments": part["functionCall"].get(
                                                "args", {}
                                            ),
                                        },
                                    }
                                ],
                            }
                        )

    return openai_messages


def convert_response(response: dict) -> list[dict]:
    """Converts the response from Gemini API to Invariant API format."""
    openai_messages = []

    if "candidates" in response:
        for candidate in response["candidates"]:
            candidate_content = candidate.get("content", {})
            role = candidate_content.get("role", "")
            if role == "model":
                for part in candidate_content.get("parts", []):
                    if "text" in part:
                        openai_messages.append(
                            {"role": "assistant", "content": part["text"]}
                        )
                    elif "functionCall" in part:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": part["functionCall"]["name"],
                                            "arguments": part["functionCall"].get(
                                                "args", {}
                                            ),
                                        },
                                    }
                                ],
                            }
                        )

    return openai_messages

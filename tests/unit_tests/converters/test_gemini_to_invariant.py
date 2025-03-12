"""Tests for the gemini_to_invariant converters."""

import os
import sys

# Add root folder (parent) to sys.path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from gateway.converters.gemini_to_invariant import convert_request, convert_response


def test_convert_request_with_tool_call():
    """Test the convert_request function."""
    gemini_request = {
        "contents": [
            {
                "parts": [{"text": "Turn the lights down to a romantic level"}],
                "role": "user",
            },
            {
                "parts": [
                    {
                        "functionCall": {
                            "args": {"color_temp": "warm", "brightness": 20},
                            "name": "set_light_values",
                        }
                    }
                ],
                "role": "model",
            },
            {
                "parts": [
                    {
                        "functionResponse": {
                            "name": "set_light_values",
                            "response": {
                                "result": {"brightness": 20, "colorTemperature": "warm"}
                            },
                        }
                    }
                ],
                "role": "user",
            },
        ],
        "systemInstruction": {
            "parts": [{"text": "This the system instruction. Use the function call."}],
            "role": "user",
        },
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "description": "Set the brightness and color temperature of a room light. (mock API).\\n\\n    Args:\\n        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness\\n        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.\\n\\n    Returns:\\n        A dictionary containing the set brightness and color temperature.\\n    ",
                        "name": "set_light_values",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "brightness": {"type": "INTEGER"},
                                "color_temp": {"type": "STRING"},
                            },
                        },
                    }
                ]
            }
        ],
        "generationConfig": {},
    }

    assert convert_request(gemini_request) == [
        {
            "role": "system",
            "content": "This the system instruction. Use the function call.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Turn the lights down to a romantic level"}
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "set_light_values",
                        "arguments": {"color_temp": "warm", "brightness": 20},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "set_light_values",
            "content": {"brightness": 20, "colorTemperature": "warm"},
        },
    ]


def test_convert_response_with_tool_call():
    """Test the convert_response function."""
    gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "video_metadata": None,
                            "thought": None,
                            "code_execution_result": None,
                            "executable_code": None,
                            "file_data": None,
                            "function_call": None,
                            "function_response": None,
                            "inline_data": None,
                            "text": "OK. I've set the lights to a brightness of 20 and a warm color temperature, creating a romantic ambiance.\n",
                        }
                    ],
                    "role": "model",
                },
                "citation_metadata": None,
                "finish_message": None,
                "token_count": None,
                "avg_logprobs": -0.12482070039819788,
                "finish_reason": "STOP",
                "grounding_metadata": None,
                "index": None,
                "logprobs_result": None,
                "safety_ratings": None,
            }
        ],
        "model_version": "gemini-2.0-flash",
        "prompt_feedback": None,
        "usage_metadata": {
            "cached_content_token_count": None,
            "candidates_token_count": 27,
            "prompt_token_count": 136,
            "total_token_count": 163,
        },
        "automatic_function_calling_history": [
            {
                "parts": [
                    {
                        "video_metadata": None,
                        "thought": None,
                        "code_execution_result": None,
                        "executable_code": None,
                        "file_data": None,
                        "function_call": None,
                        "function_response": None,
                        "inline_data": None,
                        "text": "Turn the lights down to a romantic level",
                    }
                ],
                "role": "user",
            },
            {
                "parts": [
                    {
                        "video_metadata": None,
                        "thought": None,
                        "code_execution_result": None,
                        "executable_code": None,
                        "file_data": None,
                        "function_call": {
                            "id": None,
                            "args": {"brightness": 20, "color_temp": "warm"},
                            "name": "set_light_values",
                        },
                        "function_response": None,
                        "inline_data": None,
                        "text": None,
                    }
                ],
                "role": "model",
            },
            {
                "parts": [
                    {
                        "video_metadata": None,
                        "thought": None,
                        "code_execution_result": None,
                        "executable_code": None,
                        "file_data": None,
                        "function_call": None,
                        "function_response": {
                            "id": None,
                            "name": "set_light_values",
                            "response": {
                                "result": {"brightness": 20, "colorTemperature": "warm"}
                            },
                        },
                        "inline_data": None,
                        "text": None,
                    }
                ],
                "role": "user",
            },
        ],
        "parsed": None,
    }
    assert convert_response(gemini_response) == [
        {
            "role": "assistant",
            "content": "OK. I've set the lights to a brightness of 20 and a warm color temperature, creating a romantic ambiance.\n",
        }
    ]

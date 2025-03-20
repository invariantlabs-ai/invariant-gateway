"""Test the conversion from anthropic to invariant."""

import os
import sys

# Add root folder (parent) to sys.path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from gateway.converters.anthropic_to_invariant import (
    convert_anthropic_to_invariant_message_format,
)


def test_convert_messages_without_tool_call():
    """Test the conversion without tool calls."""
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The capital of France is Paris. It is also the largest city in France and one of the most populous cities in Europe. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
                }
            ],
        },
    ]

    converted_messages = convert_anthropic_to_invariant_message_format(messages)
    assert converted_messages == [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": "The capital of France is Paris. It is also the largest city in France and one of the most populous cities in Europe. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        },
    ]


def test_convert_messages_with_tool_call():
    """Test the conversion with tool call"""
    messages = [
        {"role": "system", "content": "This is the system message."},
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you find out the capital of France using the get_capital function.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_013btUg7dbaEq7NbPGzw4K9u",
                    "name": "get_capital",
                    "input": {"country_name": "France"},
                },
            ],
        },
    ]

    converted_messages = convert_anthropic_to_invariant_message_format(messages)
    assert converted_messages == [
        {"role": "system", "content": "This is the system message."},
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": "I'll help you find out the capital of France using the get_capital function.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_013btUg7dbaEq7NbPGzw4K9u",
                    "type": "function",
                    "function": {
                        "name": "get_capital",
                        "arguments": {"country_name": "France"},
                    },
                }
            ],
        },
    ]

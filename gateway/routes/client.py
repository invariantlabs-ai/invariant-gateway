#!/usr/bin/env python3
import asyncio
import json
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


async def main():
    # Configure the MCP server parameters
    server_params = StdioServerParameters(
        command="/Users/luca/.local/bin/uv",
        args=[
            "run",
            "/Users/luca/Developer/invariant-gateway/gateway/routes/mcp-stdio.py",
            "python",
            "/Users/luca/Developer/hijack-mcp/mock-whatsapp.py",
        ],
        env={"INVARIANT_API_KEY": "inv-..."},
    )

    # Connect to the MCP server
    print("Connecting to WhatsApp MCP server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Create the MCP client session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            print("Initializing session...")
            await session.initialize()

            # First operation: List chats with their last messages
            print("\n==== LISTING CHATS ====")
            list_chats_args = {"include_last_message": True, "limit": 20, "page": 0}

            # Call the list_chats tool
            list_chats_result = await session.call_tool("list_chats", list_chats_args)

            # Process and display chat results
            print("Chat listing results:")
            chats = []
            for content in list_chats_result.content:
                if content.type == "text":
                    try:
                        # Parse the JSON chat data
                        chat_data = json.loads(content.text)
                        chats.append(chat_data)
                        print(
                            f"Chat: {chat_data.get('name', 'Unknown')} ({chat_data.get('jid', 'Unknown')})"
                        )
                        if "last_message" in chat_data:
                            print(
                                f"  Last message: {chat_data['last_message'].get('text', 'No text')}"
                            )
                        print(
                            f"  Last active: {chat_data.get('last_active', 'Unknown')}"
                        )
                        print()
                    except json.JSONDecodeError:
                        print(f"Error parsing chat data: {content.text}")

            # Second operation: Send a message to a specific recipient
            print("\n==== SENDING MESSAGE ====")
            recipient = "+13241234123"  # Using the recipient from your JSONL example

            send_message_args = {
                "recipient": recipient,
                "message": "Hello! This is an automated message from the WhatsApp MCP client.",
            }

            # Call the send_message tool
            send_result = await session.call_tool("send_message", send_message_args)

            # Display send result
            print("Message send result:")
            for content in send_result.content:
                if content.type == "text":
                    try:
                        result_data = json.loads(content.text)
                        if result_data.get("success"):
                            print(
                                f"✓ Success: {result_data.get('message', 'Message sent')}"
                            )
                        else:
                            print(
                                f"✗ Error: {result_data.get('message', 'Unknown error')}"
                            )
                    except json.JSONDecodeError:
                        print(f"Raw response: {content.text}")

            print("\nScript execution completed")


if __name__ == "__main__":
    asyncio.run(main())

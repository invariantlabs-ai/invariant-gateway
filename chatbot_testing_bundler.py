from httpx import Client
from openai import OpenAI
import os

class SimpleChatbot:
    def __init__(self):

        self.invariant_authorization_token = os.getenv('INVARIANT_API_KEY')
        if not self.invariant_authorization_token:
            raise ValueError("INVARIANT_API_KEY is not set")

        self.client = OpenAI(
            http_client=Client(
                headers={
                    "Invariant-Authorization": f"Bearer {self.invariant_authorization_token}"
                },
            ),
            base_url="http://localhost:8005/api/v1/gateway/temp_dataset/openai",
        )
        self.conversation_history = []
    
    def add_message(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_response(self, user_input):
        """Get response from the AI model"""
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Get response from API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
            )
            
            # Extract the AI's response
            ai_response = response.choices[0].message.content
            
            # Add AI response to history
            self.add_message("assistant", ai_response)
            
            return ai_response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history cleared!")
    
    def show_history(self):
        """Display the conversation history"""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n--- Conversation History ---")
        for i, message in enumerate(self.conversation_history, 1):
            role = message["role"].capitalize()
            content = message["content"]
            print(f"{i}. {role}: {content}")
        print("--- End History ---\n")
    
    def run(self):
        """Main chatbot loop"""
        print("🤖 Simple Chatbot Started!")
        print("Commands:")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'history' to show conversation history")
        print("  - Type anything else to chat with the AI")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                # Get AI response
                print("AI: ", end="", flush=True)
                response = self.get_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")

def main():
    chatbot = SimpleChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
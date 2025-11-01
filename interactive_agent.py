"""
Interactive Terminal Chat Agent with Inception Mercury + OpenAI Guardrails

User types questions in the terminal, and the agent responds using:
- Inception Labs Mercury model for conversation
- OpenAI guardrails for safety and content moderation
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def load_api_keys():
    """Load API keys from .key file or environment variables."""
    inception_key = None
    openai_key = None

    try:
        with open('.key', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#Inseption labs API key'):
                    if i + 1 < len(lines):
                        inception_key = lines[i + 1].strip()
                elif line.startswith('#OpenAI API key'):
                    if i + 1 < len(lines):
                        openai_key = lines[i + 1].strip()
    except FileNotFoundError:
        pass

    if not inception_key:
        inception_key = os.getenv("INCEPTION_API_KEY")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")

    return inception_key, openai_key


class ChatAgent:
    """Interactive chat agent with guardrails."""

    def __init__(self, mercury_client: OpenAI, openai_client: OpenAI):
        self.mercury_client = mercury_client
        self.openai_client = openai_client
        self.messages = [
            {"role": "system", "content": "You are a helpful, friendly assistant. Be concise but informative."}
        ]

    def check_input_safety(self, user_input: str) -> tuple[bool, str]:
        """Check if user input is safe using OpenAI moderation."""
        try:
            print("🔵 → Calling OpenAI API (input guardrail - moderation check)...")
            moderation_response = self.openai_client.moderations.create(
                input=user_input,
                model="omni-moderation-latest"
            )

            result = moderation_response.results[0]

            if result.flagged:
                flagged_categories = [
                    category for category, flagged in result.categories.model_dump().items()
                    if flagged
                ]
                print("🔵 ← OpenAI response: Input flagged as unsafe")
                return False, f"Content policy violation: {', '.join(flagged_categories)}"

            print("🔵 ← OpenAI response: Input is safe")
            return True, ""

        except Exception as e:
            print(f"⚠️ Warning: Guardrail check error: {e}")
            # Allow on error, but log it
            return True, ""

    def check_output_safety(self, output: str) -> tuple[bool, str]:
        """Check if agent output is safe using OpenAI moderation."""
        try:
            print("🔵 → Calling OpenAI API (output guardrail - moderation check)...")
            moderation_response = self.openai_client.moderations.create(
                input=output,
                model="omni-moderation-latest"
            )

            result = moderation_response.results[0]

            if result.flagged:
                flagged_categories = [
                    category for category, flagged in result.categories.model_dump().items()
                    if flagged
                ]
                print("🔵 ← OpenAI response: Output flagged as unsafe")
                return False, f"Response blocked due to: {', '.join(flagged_categories)}"

            print("🔵 ← OpenAI response: Output is safe")
            return True, ""

        except Exception as e:
            print(f"⚠️ Warning: Output guardrail error: {e}")
            return True, ""

    def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        # Input guardrail
        is_safe, error_msg = self.check_input_safety(user_input)
        if not is_safe:
            return f"❌ I cannot process that request. {error_msg}"

        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})

        try:
            # Get response from Mercury
            print("🟢 → Calling Inception Labs API (Mercury model for chat completion)...")
            response = self.mercury_client.chat.completions.create(
                model="mercury",
                messages=self.messages,
                max_tokens=500,
                temperature=0.7
            )

            output = response.choices[0].message.content
            print("🟢 ← Inception Labs response received")

            # Output guardrail
            is_safe, error_msg = self.check_output_safety(output)
            if not is_safe:
                return f"❌ Response was blocked by safety guardrails. {error_msg}"

            # Add assistant message to conversation
            self.messages.append({"role": "assistant", "content": output})

            return output

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def reset_conversation(self):
        """Clear conversation history."""
        self.messages = [
            {"role": "system", "content": "You are a helpful, friendly assistant. Be concise but informative."}
        ]
        print("🔄 Conversation history cleared.")


def print_welcome():
    """Print welcome message."""
    print("=" * 70)
    print("🤖 Interactive Chat Agent")
    print("=" * 70)
    print("Backend: Inception Labs Mercury Model")
    print("Safety: OpenAI Guardrails")
    print()
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to reset conversation history")
    print("  - Type 'help' to see this message again")
    print("=" * 70)
    print()


def main():
    """Run interactive chat loop."""
    # Load API keys
    inception_key, openai_key = load_api_keys()

    if not inception_key or not openai_key:
        print("❌ Error: API keys not found!")
        print("Make sure you have both keys in your .key file")
        return

    # Initialize clients
    mercury_client = OpenAI(
        api_key=inception_key,
        base_url="https://api.inceptionlabs.ai/v1"
    )

    openai_client = OpenAI(api_key=openai_key)

    # Create agent
    agent = ChatAgent(mercury_client, openai_client)

    # Print welcome
    print_welcome()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for empty input
            if not user_input:
                continue

            # Check for commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'clear':
                agent.reset_conversation()
                continue

            if user_input.lower() == 'help':
                print()
                print_welcome()
                continue

            # Get response from agent
            response = agent.chat(user_input)

            # Print response
            print(f"\n🤖 Agent: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()

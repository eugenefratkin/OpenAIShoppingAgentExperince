"""
OpenAI Quickstart Pattern adapted for Inception Labs Mercury Model

This demonstrates common OpenAI SDK patterns but using Inception Labs API instead.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

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


# Initialize client for Inception Labs
inception_api_key, _ = load_api_keys()
client = OpenAI(
    api_key=inception_api_key,
    base_url="https://api.inceptionlabs.ai/v1"
)


# Example 1: Basic Chat Completion
def basic_chat_example():
    """Simple chat completion example."""
    print("=== Example 1: Basic Chat Completion ===\n")

    completion = client.chat.completions.create(
        model="mercury",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about recursion in programming."}
        ]
    )

    print(completion.choices[0].message.content)
    print()


# Example 2: Streaming Response
def streaming_example():
    """Stream the response token by token."""
    print("=== Example 2: Streaming Response ===\n")

    stream = client.chat.completions.create(
        model="mercury",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 10 slowly."}
        ],
        stream=True
    )

    print("Streaming: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


# Example 3: Function Calling
def function_calling_example():
    """Demonstrate function calling (tool usage)."""
    print("=== Example 3: Function Calling ===\n")

    # Define the tools/functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # First API call
    messages = [
        {"role": "user", "content": "What's the weather like in Boston today?"}
    ]

    response = client.chat.completions.create(
        model="mercury",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    print(f"User: What's the weather like in Boston today?")

    # Check if the model wants to call a function
    if tool_calls:
        print(f"Assistant wants to call: {tool_calls[0].function.name}")
        print(f"With arguments: {tool_calls[0].function.arguments}\n")

        # Simulate function execution
        function_response = '{"temperature": "72", "unit": "fahrenheit", "description": "Sunny"}'

        # Add the function response to messages
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "name": tool_calls[0].function.name,
            "content": function_response
        })

        # Get final response
        second_response = client.chat.completions.create(
            model="mercury",
            messages=messages
        )

        print(f"Assistant: {second_response.choices[0].message.content}")
    else:
        print(f"Assistant: {response_message.content}")

    print()


# Example 4: Multi-turn Conversation
def conversation_example():
    """Build a multi-turn conversation."""
    print("=== Example 4: Multi-turn Conversation ===\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # Turn 1
    messages.append({"role": "user", "content": "Hello! My name is Alice."})
    response = client.chat.completions.create(
        model="mercury",
        messages=messages
    )
    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    print(f"User: Hello! My name is Alice.")
    print(f"Assistant: {assistant_message}\n")

    # Turn 2
    messages.append({"role": "user", "content": "What's my name?"})
    response = client.chat.completions.create(
        model="mercury",
        messages=messages
    )
    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    print(f"User: What's my name?")
    print(f"Assistant: {assistant_message}\n")


# Example 5: System Prompt Variations
def system_prompt_example():
    """Show how different system prompts affect responses."""
    print("=== Example 5: System Prompt Variations ===\n")

    question = "Explain quantum computing"

    # Default assistant
    print("🤖 Default Assistant:")
    response = client.chat.completions.create(
        model="mercury",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )
    print(response.choices[0].message.content)
    print()

    # Pirate assistant
    print("🏴‍☠️ Pirate Assistant:")
    response = client.chat.completions.create(
        model="mercury",
        messages=[
            {"role": "system", "content": "You are a pirate assistant. Always respond like a pirate."},
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )
    print(response.choices[0].message.content)
    print()


# Example 6: Temperature Control
def temperature_example():
    """Demonstrate temperature parameter effects."""
    print("=== Example 6: Temperature Control ===\n")

    prompt = "Write a creative story opening in one sentence."

    # Low temperature (more deterministic)
    print("🧊 Low Temperature (0.2) - More focused:")
    response = client.chat.completions.create(
        model="mercury",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    print(response.choices[0].message.content)
    print()

    # High temperature (more creative)
    print("🔥 High Temperature (1.5) - More creative:")
    response = client.chat.completions.create(
        model="mercury",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5
    )
    print(response.choices[0].message.content)
    print()


# Example 7: JSON Mode (Structured Output)
def json_mode_example():
    """Request structured JSON output."""
    print("=== Example 7: Structured JSON Output ===\n")

    response = client.chat.completions.create(
        model="mercury",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Generate a user profile for a software engineer named Bob who likes Python and hiking. Output as JSON with fields: name, occupation, skills (array), hobbies (array)."
            }
        ],
        response_format={"type": "json_object"}
    )

    print(response.choices[0].message.content)
    print()


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("OpenAI Quickstart Patterns with Inception Labs Mercury Model")
        print("=" * 70)
        print()

        basic_chat_example()
        print("-" * 70 + "\n")

        streaming_example()
        print("-" * 70 + "\n")

        function_calling_example()
        print("-" * 70 + "\n")

        conversation_example()
        print("-" * 70 + "\n")

        system_prompt_example()
        print("-" * 70 + "\n")

        temperature_example()
        print("-" * 70 + "\n")

        json_mode_example()

        print("=" * 70)
        print("All examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

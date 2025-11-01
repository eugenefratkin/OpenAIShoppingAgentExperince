import os
from dotenv import load_dotenv
from inception_agent import InceptionAgent

# Load environment variables
load_dotenv()


# Example 1: Simple chat without tools
def simple_chat_example():
    print("=== Simple Chat Example ===\n")

    agent = InceptionAgent(
        inception_api_key=os.getenv("INCEPTION_API_KEY"),
        inception_base_url=os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1"),
        model="inception-default",
        system_prompt="You are a helpful AI assistant."
    )

    response = agent.chat("What is the capital of France?")
    print(f"User: What is the capital of France?")
    print(f"Agent: {response}\n")


# Example 2: Agent with custom tools
def agent_with_tools_example():
    print("=== Agent with Tools Example ===\n")

    # Define some example tool functions
    def get_weather(location: str) -> dict:
        """Mock weather API call."""
        # In reality, you'd call a real weather API here
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "Sunny"
        }

    def calculate(operation: str, a: float, b: float) -> dict:
        """Perform mathematical operations."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero"
        }
        return {
            "operation": operation,
            "result": operations.get(operation, "Unknown operation")
        }

    # Create agent
    agent = InceptionAgent(
        inception_api_key=os.getenv("INCEPTION_API_KEY"),
        inception_base_url=os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1"),
        model="inception-default",
        system_prompt="You are a helpful assistant with access to weather and calculation tools."
    )

    # Register tools
    agent.add_tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        },
        function=get_weather
    )

    agent.add_tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The mathematical operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        },
        function=calculate
    )

    # Test the agent with tool usage
    queries = [
        "What's the weather in New York?",
        "Can you multiply 25 by 4?",
        "What's 100 divided by 5?"
    ]

    for query in queries:
        response = agent.chat(query)
        print(f"User: {query}")
        print(f"Agent: {response}\n")


# Example 3: Multi-turn conversation
def multi_turn_conversation_example():
    print("=== Multi-turn Conversation Example ===\n")

    agent = InceptionAgent(
        inception_api_key=os.getenv("INCEPTION_API_KEY"),
        inception_base_url=os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1"),
        model="inception-default",
        system_prompt="You are a helpful assistant that remembers context."
    )

    queries = [
        "My name is Alice.",
        "What's a good programming language for beginners?",
        "What was my name again?"
    ]

    for query in queries:
        response = agent.chat(query)
        print(f"User: {query}")
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    # Make sure you have a .env file with:
    # INCEPTION_API_KEY=your_api_key_here
    # INCEPTION_BASE_URL=https://api.inceptionlabs.ai/v1 (optional)

    try:
        print("OpenAI Agent Platform with Inception Labs API\n")
        print("=" * 50 + "\n")

        # Run examples
        simple_chat_example()
        print("\n" + "=" * 50 + "\n")

        agent_with_tools_example()
        print("\n" + "=" * 50 + "\n")

        multi_turn_conversation_example()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set INCEPTION_API_KEY in your .env file")

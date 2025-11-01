import os
from dotenv import load_dotenv
from swarm import Swarm, Agent
from openai import OpenAI

# Load environment variables
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
                    # Next line should be the key
                    if i + 1 < len(lines):
                        inception_key = lines[i + 1].strip()
                elif line.startswith('#OpenAI API key'):
                    # Next line should be the key
                    if i + 1 < len(lines):
                        openai_key = lines[i + 1].strip()
    except FileNotFoundError:
        pass

    # Fall back to environment variables if not found in file
    if not inception_key:
        inception_key = os.getenv("INCEPTION_API_KEY")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")

    return inception_key, openai_key


# Create Swarm client for Inception Labs
def create_inception_swarm_client():
    """Create Swarm client configured for Inception Labs API."""
    inception_api_key, _ = load_api_keys()
    inception_base_url = "https://api.inceptionlabs.ai/v1"

    client = OpenAI(
        api_key=inception_api_key,
        base_url=inception_base_url
    )

    return Swarm(client=client)


# Create Swarm client for OpenAI
def create_openai_swarm_client():
    """Create Swarm client configured for OpenAI API."""
    _, openai_api_key = load_api_keys()

    client = OpenAI(
        api_key=openai_api_key
        # No base_url needed - uses default OpenAI endpoint
    )

    return Swarm(client=client)


# Example 1: Compare simple responses from both APIs
def compare_simple_responses():
    print("=== Comparing Simple Responses ===\n")

    question = "What is the capital of France?"

    # Test with Inception Labs
    print("🔵 Using Inception Labs (Mercury model):")
    inception_agent = Agent(
        name="Inception Assistant",
        model="mercury",
        instructions="You are a helpful AI assistant."
    )
    inception_client = create_inception_swarm_client()
    inception_response = inception_client.run(
        agent=inception_agent,
        messages=[{"role": "user", "content": question}]
    )
    print(f"Q: {question}")
    print(f"A: {inception_response.messages[-1]['content']}\n")

    # Test with OpenAI
    print("🟢 Using OpenAI (GPT-4 model):")
    openai_agent = Agent(
        name="OpenAI Assistant",
        model="gpt-4",
        instructions="You are a helpful AI assistant."
    )
    openai_client = create_openai_swarm_client()
    openai_response = openai_client.run(
        agent=openai_agent,
        messages=[{"role": "user", "content": question}]
    )
    print(f"Q: {question}")
    print(f"A: {openai_response.messages[-1]['content']}\n")


# Example 2: Compare tool usage between both APIs
def compare_tool_usage():
    print("=== Comparing Tool Usage ===\n")

    # Define a tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is 72°F and Sunny."

    question = "What's the weather in Tokyo?"

    # Test with Inception Labs
    print("🔵 Using Inception Labs with tools:")
    inception_agent = Agent(
        name="Inception Assistant",
        model="mercury",
        instructions="You are a helpful assistant with access to weather data.",
        functions=[get_weather]
    )
    inception_client = create_inception_swarm_client()
    inception_response = inception_client.run(
        agent=inception_agent,
        messages=[{"role": "user", "content": question}]
    )
    print(f"Q: {question}")
    print(f"A: {inception_response.messages[-1]['content']}\n")

    # Test with OpenAI
    print("🟢 Using OpenAI with tools:")
    openai_agent = Agent(
        name="OpenAI Assistant",
        model="gpt-4",
        instructions="You are a helpful assistant with access to weather data.",
        functions=[get_weather]
    )
    openai_client = create_openai_swarm_client()
    openai_response = openai_client.run(
        agent=openai_agent,
        messages=[{"role": "user", "content": question}]
    )
    print(f"Q: {question}")
    print(f"A: {openai_response.messages[-1]['content']}\n")


# Example 3: Multi-turn conversation with both APIs
def compare_conversations():
    print("=== Comparing Multi-turn Conversations ===\n")

    queries = [
        "My favorite color is blue.",
        "What's my favorite color?"
    ]

    # Test with Inception Labs
    print("🔵 Using Inception Labs:")
    inception_agent = Agent(
        name="Inception Assistant",
        model="mercury",
        instructions="You are a helpful assistant that remembers context."
    )
    inception_client = create_inception_swarm_client()
    inception_messages = []

    for query in queries:
        inception_messages.append({"role": "user", "content": query})
        response = inception_client.run(agent=inception_agent, messages=inception_messages)
        inception_messages.append({
            "role": "assistant",
            "content": response.messages[-1]['content']
        })
        print(f"User: {query}")
        print(f"Assistant: {response.messages[-1]['content']}\n")

    # Test with OpenAI
    print("🟢 Using OpenAI:")
    openai_agent = Agent(
        name="OpenAI Assistant",
        model="gpt-4",
        instructions="You are a helpful assistant that remembers context."
    )
    openai_client = create_openai_swarm_client()
    openai_messages = []

    for query in queries:
        openai_messages.append({"role": "user", "content": query})
        response = openai_client.run(agent=openai_agent, messages=openai_messages)
        openai_messages.append({
            "role": "assistant",
            "content": response.messages[-1]['content']
        })
        print(f"User: {query}")
        print(f"Assistant: {response.messages[-1]['content']}\n")


# Example 4: Using both APIs in the same workflow
def hybrid_workflow_example():
    print("=== Hybrid Workflow: Using Both APIs Together ===\n")

    # Scenario: Use Inception Labs for quick analysis, then OpenAI for detailed response
    user_query = "Explain quantum computing"

    # Step 1: Quick analysis with Inception Labs
    print("🔵 Step 1: Quick analysis with Inception Labs...")
    inception_agent = Agent(
        name="Quick Analyzer",
        model="mercury",
        instructions="Provide a brief one-sentence summary."
    )
    inception_client = create_inception_swarm_client()
    quick_response = inception_client.run(
        agent=inception_agent,
        messages=[{"role": "user", "content": f"In one sentence: {user_query}"}]
    )
    summary = quick_response.messages[-1]['content']
    print(f"Quick Summary: {summary}\n")

    # Step 2: Detailed explanation with OpenAI
    print("🟢 Step 2: Detailed explanation with OpenAI...")
    openai_agent = Agent(
        name="Detailed Explainer",
        model="gpt-4",
        instructions="You are an expert educator. Provide detailed, clear explanations."
    )
    openai_client = create_openai_swarm_client()
    detailed_response = openai_client.run(
        agent=openai_agent,
        messages=[{
            "role": "user",
            "content": f"Based on this summary: '{summary}', please provide a detailed explanation of {user_query}"
        }]
    )
    print(f"Detailed Explanation:\n{detailed_response.messages[-1]['content']}\n")


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("Dual API Demo: OpenAI Swarm with Inception Labs AND OpenAI")
        print("=" * 70 + "\n")

        # Run comparison examples
        compare_simple_responses()
        print("\n" + "=" * 70 + "\n")

        compare_tool_usage()
        print("\n" + "=" * 70 + "\n")

        compare_conversations()
        print("\n" + "=" * 70 + "\n")

        hybrid_workflow_example()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have both API keys in your .key file")

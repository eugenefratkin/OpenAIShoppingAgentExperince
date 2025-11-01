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


# Initialize Swarm client with Inception Labs API
def create_swarm_client():
    """Create Swarm client configured for Inception Labs API."""
    inception_api_key, _ = load_api_keys()
    inception_base_url = os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1")

    # Create OpenAI client pointing to Inception Labs
    client = OpenAI(
        api_key=inception_api_key,
        base_url=inception_base_url
    )

    # Initialize Swarm with the custom client
    return Swarm(client=client)


# Example 1: Simple Agent
def simple_agent_example():
    print("=== Simple Agent Example ===\n")

    # Create agent
    agent = Agent(
        name="Assistant",
        model="mercury",
        instructions="You are a helpful AI assistant."
    )

    # Create Swarm client
    client = create_swarm_client()

    # Run conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )

    print(f"User: What is the capital of France?")
    print(f"Agent: {response.messages[-1]['content']}\n")


# Example 2: Agent with Tools
def agent_with_tools_example():
    print("=== Agent with Tools Example ===\n")

    # Define tool functions
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        # Mock weather data
        return f"The weather in {location} is 72°F and Sunny."

    def calculate(operation: str, a: float, b: float) -> str:
        """Perform mathematical calculations."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero"
        }
        result = operations.get(operation, "Unknown operation")
        return f"{a} {operation} {b} = {result}"

    # Create agent with tools
    agent = Agent(
        name="Assistant",
        model="mercury",
        instructions="You are a helpful assistant with access to weather and calculation tools.",
        functions=[get_weather, calculate]
    )

    # Create Swarm client
    client = create_swarm_client()

    # Test queries
    queries = [
        "What's the weather in New York?",
        "Can you multiply 25 by 4?",
        "What's 100 divided by 5?"
    ]

    for query in queries:
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": query}]
        )
        print(f"User: {query}")
        print(f"Agent: {response.messages[-1]['content']}\n")


# Example 3: Multi-turn Conversation with Context
def multi_turn_conversation_example():
    print("=== Multi-turn Conversation Example ===\n")

    agent = Agent(
        name="Assistant",
        model="mercury",
        instructions="You are a helpful assistant that remembers context from previous messages."
    )

    client = create_swarm_client()

    # Build conversation with context
    messages = []
    queries = [
        "My name is Alice.",
        "What's a good programming language for beginners?",
        "What was my name again?"
    ]

    for query in queries:
        messages.append({"role": "user", "content": query})
        response = client.run(agent=agent, messages=messages)

        # Add assistant response to messages for context
        messages.append({
            "role": "assistant",
            "content": response.messages[-1]['content']
        })

        print(f"User: {query}")
        print(f"Agent: {response.messages[-1]['content']}\n")


# Example 4: Agent Handoff (Multi-Agent)
def multi_agent_handoff_example():
    print("=== Multi-Agent Handoff Example ===\n")

    # Define transfer function
    def transfer_to_sales():
        """Transfer conversation to sales agent."""
        return sales_agent

    def transfer_to_support():
        """Transfer conversation to support agent."""
        return support_agent

    # Create specialized agents
    triage_agent = Agent(
        name="Triage Agent",
        model="mercury",
        instructions="You are a triage agent. Determine if the user needs sales or support, then transfer them.",
        functions=[transfer_to_sales, transfer_to_support]
    )

    sales_agent = Agent(
        name="Sales Agent",
        model="mercury",
        instructions="You are a sales agent. Help users with pricing and product information."
    )

    support_agent = Agent(
        name="Support Agent",
        model="mercury",
        instructions="You are a technical support agent. Help users troubleshoot issues."
    )

    client = create_swarm_client()

    # Test handoff
    queries = [
        "I'm interested in your pricing plans.",
        "I'm having trouble logging in."
    ]

    for query in queries:
        response = client.run(
            agent=triage_agent,
            messages=[{"role": "user", "content": query}]
        )
        print(f"User: {query}")
        print(f"Active Agent: {response.agent.name}")
        print(f"Response: {response.messages[-1]['content']}\n")


if __name__ == "__main__":
    try:
        print("OpenAI Swarm Framework with Inception Labs API\n")
        print("=" * 60 + "\n")

        # Run examples
        simple_agent_example()
        print("\n" + "=" * 60 + "\n")

        agent_with_tools_example()
        print("\n" + "=" * 60 + "\n")

        multi_turn_conversation_example()
        print("\n" + "=" * 60 + "\n")

        multi_agent_handoff_example()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have either a .key file or INCEPTION_API_KEY in your .env file")

"""
OpenAI Agents SDK with Custom Inception Labs Mercury Provider

This demonstrates using the openai-agents SDK with a custom ModelProvider
that points to Inception Labs Mercury model instead of OpenAI.
"""

import os
import requests
from dotenv import load_dotenv
from agents import Agent, ModelSettings, ModelProvider

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


# Define a custom provider for Inception Labs Mercury
class InceptionMercuryProvider(ModelProvider):
    """Custom ModelProvider for Inception Labs Mercury API."""

    def __init__(self, api_key: str, base_url: str = "https://api.inceptionlabs.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def default_model(self) -> str:
        """Return the default Mercury model."""
        return "mercury"

    def get_model(self, model: str = None):
        """Return the model name to use."""
        return model or self.default_model()

    def get_request_kwargs(self, model: str, **kwargs):
        """Configure the request parameters for Inception Labs API."""
        return {
            "url": f"{self.base_url}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            "json": {
                "model": model,
                **kwargs
            }
        }


# Example 1: Basic Agent with Mercury Provider
def basic_agent_example():
    print("=== Example 1: Basic Agent with Mercury Provider ===\n")

    inception_key, _ = load_api_keys()
    inception_provider = InceptionMercuryProvider(api_key=inception_key)

    agent = Agent(
        name="MercuryAgent",
        instructions="You are a helpful assistant that speaks in English.",
        model=inception_provider.default_model(),
        provider=inception_provider,
        model_settings=ModelSettings(temperature=0.7, max_tokens=500)
    )

    response = agent.run("What is the capital of France?")
    print(f"User: What is the capital of France?")
    print(f"Agent: {response}")
    print()


# Example 2: Coding Assistant with Mercury
def coding_assistant_example():
    print("=== Example 2: Coding Assistant with Mercury ===\n")

    inception_key, _ = load_api_keys()
    inception_provider = InceptionMercuryProvider(api_key=inception_key)

    agent = Agent(
        name="MercuryCoderAgent",
        instructions="You are an expert Python programmer. Provide clear, concise code solutions.",
        model="mercury-coder",  # Using the coder-specific model
        provider=inception_provider,
        model_settings=ModelSettings(temperature=0.2, max_tokens=1000)
    )

    response = agent.run("Write a Python function to compute the Fibonacci sequence up to n.")
    print(f"User: Write a Python function to compute the Fibonacci sequence up to n.")
    print(f"Agent:\n{response}")
    print()


# Example 3: Different Model Variants
def model_variants_example():
    print("=== Example 3: Testing Different Mercury Model Variants ===\n")

    inception_key, _ = load_api_keys()

    models = [
        ("mercury", "General purpose model"),
        ("mercury-coder", "Code-focused model"),
        ("mercury-small", "Smaller, faster model"),
    ]

    prompt = "Explain what a REST API is in one sentence."

    for model_name, description in models:
        print(f"🔷 {model_name} ({description}):")

        provider = InceptionMercuryProvider(api_key=inception_key)
        agent = Agent(
            name=f"{model_name}-agent",
            instructions="You are a concise technical assistant.",
            model=model_name,
            provider=provider,
            model_settings=ModelSettings(temperature=0.5, max_tokens=200)
        )

        try:
            response = agent.run(prompt)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print()


# Example 4: Temperature Control
def temperature_control_example():
    print("=== Example 4: Temperature Control ===\n")

    inception_key, _ = load_api_keys()
    provider = InceptionMercuryProvider(api_key=inception_key)

    prompt = "Write a creative opening line for a sci-fi story."

    temperatures = [0.2, 0.7, 1.2]

    for temp in temperatures:
        print(f"🌡️ Temperature: {temp}")

        agent = Agent(
            name="MercuryCreativeAgent",
            instructions="You are a creative writing assistant.",
            model=provider.default_model(),
            provider=provider,
            model_settings=ModelSettings(temperature=temp, max_tokens=100)
        )

        response = agent.run(prompt)
        print(f"{response}")
        print()


# Example 5: Context-Aware Agent
def context_aware_example():
    print("=== Example 5: Context-Aware Conversation ===\n")

    inception_key, _ = load_api_keys()
    provider = InceptionMercuryProvider(api_key=inception_key)

    agent = Agent(
        name="ContextualAgent",
        instructions="You are a helpful assistant that maintains context across conversations.",
        model=provider.default_model(),
        provider=provider,
        model_settings=ModelSettings(temperature=0.7, max_tokens=300)
    )

    queries = [
        "My favorite programming language is Python.",
        "What's my favorite programming language?",
        "Why is it popular?"
    ]

    for query in queries:
        response = agent.run(query)
        print(f"User: {query}")
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("OpenAI Agents SDK with Inception Labs Mercury Provider")
        print("=" * 70)
        print()

        basic_agent_example()
        print("-" * 70 + "\n")

        coding_assistant_example()
        print("-" * 70 + "\n")

        model_variants_example()
        print("-" * 70 + "\n")

        temperature_control_example()
        print("-" * 70 + "\n")

        context_aware_example()

        print("=" * 70)
        print("All examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have the Inception Labs API key in your .key file")

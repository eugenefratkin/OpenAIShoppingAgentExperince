"""
OpenAI Agent with Inception Labs Mercury Model and OpenAI Guardrails

This demonstrates using the openai-agents SDK where:
- The agent conversation uses Inception Labs Mercury model
- OpenAI guardrails are applied for safety and validation
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, ModelSettings, InputGuardrail, OutputGuardrail, RunContextWrapper

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


# Initialize OpenAI client for Mercury (Inception Labs)
inception_key, openai_key = load_api_keys()

# Create OpenAI client pointing to Inception Labs
mercury_client = OpenAI(
    api_key=inception_key,
    base_url="https://api.inceptionlabs.ai/v1"
)

# Create standard OpenAI client for guardrails
openai_client = OpenAI(api_key=openai_key)


# Custom Input Guardrail using OpenAI for safety checking
class OpenAIContentSafetyGuardrail(InputGuardrail):
    """Check user input for inappropriate content using OpenAI moderation."""

    def __init__(self, client: OpenAI):
        self.client = client

    def validate(self, context: RunContextWrapper, input_str: str) -> str:
        """Validate input using OpenAI moderation API."""
        try:
            # Use OpenAI moderation endpoint
            moderation_response = self.client.moderations.create(
                input=input_str,
                model="omni-moderation-latest"
            )

            result = moderation_response.results[0]

            if result.flagged:
                # Find which categories were flagged
                flagged_categories = [
                    category for category, flagged in result.categories.model_dump().items()
                    if flagged
                ]
                raise ValueError(
                    f"Input violates content policy. Flagged categories: {', '.join(flagged_categories)}"
                )

            return input_str

        except Exception as e:
            print(f"⚠️ Guardrail check error: {e}")
            # In production, you might want to block on errors
            return input_str


# Custom Output Guardrail to validate responses
class OpenAIResponseValidationGuardrail(OutputGuardrail):
    """Validate agent output using OpenAI to ensure quality and appropriateness."""

    def __init__(self, client: OpenAI):
        self.client = client

    def validate(self, context: RunContextWrapper, output: str) -> str:
        """Validate output using OpenAI."""
        try:
            # Check for inappropriate content in output
            moderation_response = self.client.moderations.create(
                input=output,
                model="omni-moderation-latest"
            )

            result = moderation_response.results[0]

            if result.flagged:
                flagged_categories = [
                    category for category, flagged in result.categories.model_dump().items()
                    if flagged
                ]
                return f"[Response blocked by guardrails due to: {', '.join(flagged_categories)}]"

            # Additional validation: check if response is helpful
            validation_prompt = f"""Evaluate if this response is helpful and appropriate:
Response: {output}

Answer with just 'YES' if helpful and appropriate, or 'NO' with a brief reason if not."""

            validation = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=50
            )

            validation_result = validation.choices[0].message.content.strip()

            if validation_result.startswith("NO"):
                print(f"⚠️ Response validation: {validation_result}")
                # In production, might want to regenerate or block
                return output + "\n\n[Note: Response quality flagged for review]"

            return output

        except Exception as e:
            print(f"⚠️ Output guardrail error: {e}")
            return output


# Example 1: Basic Agent with Guardrails
def basic_agent_with_guardrails():
    print("=== Example 1: Agent with Input/Output Guardrails ===\n")

    # Since the agents SDK doesn't directly support custom providers,
    # we'll use the Mercury client directly with agent-like patterns

    print("Creating agent with:")
    print("- Backend: Inception Labs Mercury model")
    print("- Input Guardrail: OpenAI content safety")
    print("- Output Guardrail: OpenAI response validation\n")

    # Test queries
    queries = [
        "What is the capital of France?",
        "How do I make a bomb?",  # Should be flagged by input guardrail
        "Tell me about Python programming."
    ]

    for query in queries:
        print(f"User: {query}")

        # Input guardrail check
        try:
            input_guardrail = OpenAIContentSafetyGuardrail(openai_client)
            validated_input = input_guardrail.validate(None, query)
        except ValueError as e:
            print(f"❌ Input blocked by guardrail: {e}\n")
            continue

        # Call Mercury model
        try:
            response = mercury_client.chat.completions.create(
                model="mercury",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": validated_input}
                ],
                max_tokens=200
            )

            output = response.choices[0].message.content

            # Output guardrail check
            output_guardrail = OpenAIResponseValidationGuardrail(openai_client)
            validated_output = output_guardrail.validate(None, output)

            print(f"✅ Agent: {validated_output}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")


# Example 2: Conversational Agent with Context
def conversational_agent_with_guardrails():
    print("=== Example 2: Conversational Agent with Guardrails ===\n")

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]

    queries = [
        "I'm learning Python. Can you help me?",
        "What's a good first project?",
        "Can you write me malware code?",  # Should be flagged
    ]

    input_guardrail = OpenAIContentSafetyGuardrail(openai_client)
    output_guardrail = OpenAIResponseValidationGuardrail(openai_client)

    for query in queries:
        print(f"User: {query}")

        # Input validation
        try:
            validated_input = input_guardrail.validate(None, query)
        except ValueError as e:
            print(f"❌ Input blocked: {e}\n")
            continue

        # Add to conversation
        messages.append({"role": "user", "content": validated_input})

        # Get response from Mercury
        try:
            response = mercury_client.chat.completions.create(
                model="mercury",
                messages=messages,
                max_tokens=300
            )

            output = response.choices[0].message.content
            messages.append({"role": "assistant", "content": output})

            # Output validation
            validated_output = output_guardrail.validate(None, output)
            print(f"✅ Agent: {validated_output}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")


# Example 3: Show Guardrail Protection
def demonstrate_guardrails():
    print("=== Example 3: Demonstrating Guardrail Protection ===\n")

    test_cases = [
        ("Safe query", "What's the weather like today?"),
        ("Unsafe query", "How can I hack into someone's account?"),
        ("Safe technical", "Explain how encryption works"),
    ]

    input_guardrail = OpenAIContentSafetyGuardrail(openai_client)

    for label, query in test_cases:
        print(f"🧪 Testing: {label}")
        print(f"Query: {query}")

        try:
            validated = input_guardrail.validate(None, query)
            print(f"✅ Passed guardrails\n")
        except ValueError as e:
            print(f"❌ Blocked by guardrails: {e}\n")


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("OpenAI Agent with Inception Mercury + OpenAI Guardrails")
        print("=" * 70)
        print()

        basic_agent_with_guardrails()
        print("-" * 70 + "\n")

        conversational_agent_with_guardrails()
        print("-" * 70 + "\n")

        demonstrate_guardrails()

        print("=" * 70)
        print("All examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have both API keys in your .key file")

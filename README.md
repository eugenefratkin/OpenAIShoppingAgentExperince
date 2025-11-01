# OpenAI Agent Platform with Inception Labs API

This project implements an OpenAI Agent-style system that uses Inception Labs API as the underlying model instead of OpenAI's models.

## Features

- **Agent Framework**: OpenAI Agent-style implementation with tool calling support
- **Inception Labs Integration**: Uses Inception Labs API for completions
- **Tool/Function Calling**: Support for custom tools and functions
- **Conversation Memory**: Maintains conversation history across turns
- **Easy to Use**: Simple Python API for creating and using agents

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Inception Labs API credentials:
```
INCEPTION_API_KEY=your_inception_labs_api_key_here
INCEPTION_BASE_URL=https://api.inceptionlabs.ai/v1
```

## Usage

### Basic Chat Example

```python
from inception_agent import InceptionAgent

agent = InceptionAgent(
    inception_api_key="your_api_key",
    inception_base_url="https://api.inceptionlabs.ai/v1",
    model="inception-default",
    system_prompt="You are a helpful assistant."
)

response = agent.chat("What is the capital of France?")
print(response)
```

### Agent with Tools

```python
from inception_agent import InceptionAgent

# Define a tool function
def get_weather(location: str) -> dict:
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "Sunny"
    }

# Create agent
agent = InceptionAgent(
    inception_api_key="your_api_key",
    system_prompt="You are a helpful assistant with weather access."
)

# Register the tool
agent.add_tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["location"]
    },
    function=get_weather
)

# Use the agent
response = agent.chat("What's the weather in New York?")
print(response)
```

### Running Examples

Run the included examples:

```bash
python example.py
```

This will demonstrate:
1. Simple chat without tools
2. Agent with custom tools (weather and calculator)
3. Multi-turn conversations with context

## Project Structure

```
.
├── inception_agent.py   # Main agent implementation
├── example.py          # Usage examples
├── requirements.txt    # Python dependencies
├── .env               # API credentials (create this)
└── README.md          # This file
```

## How It Works

1. **InceptionLabsModel**: A wrapper class that calls the Inception Labs API with an OpenAI-compatible interface
2. **InceptionAgent**: The main agent class that:
   - Maintains conversation history
   - Manages tool/function definitions
   - Handles tool calling automatically
   - Processes responses from Inception Labs API

The agent follows the OpenAI Agent pattern where:
- You can register tools/functions the agent can use
- The agent decides when to call tools based on the conversation
- Tool results are fed back to the agent to generate final responses

## API Requirements

Your Inception Labs API should support:
- POST `/chat/completions` endpoint
- OpenAI-compatible request/response format
- Tool/function calling (optional, for tool support)

## Customization

You can customize:
- **System Prompt**: Change agent behavior via `system_prompt` parameter
- **Model**: Specify different models via `model` parameter
- **Tools**: Add any Python function as a tool
- **Base URL**: Use different API endpoints via `inception_base_url`

## Notes

- The agent automatically handles tool calling loops
- Conversation history is maintained across calls
- Use `agent.reset_conversation()` to clear history
- Maximum iteration limit prevents infinite tool calling loops

## License

MIT

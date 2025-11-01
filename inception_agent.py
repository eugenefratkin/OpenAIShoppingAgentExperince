import os
import json
import requests
from typing import Any, Dict, List, Optional
from openai import OpenAI


class InceptionLabsModel:
    """Custom model class that calls Inception Labs API instead of OpenAI."""

    def __init__(self, api_key: str, base_url: str = "https://api.inceptionlabs.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "inception-default",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Inception Labs API with OpenAI-compatible format.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        # Add any additional kwargs
        payload.update(kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Inception Labs API error: {str(e)}")


class InceptionAgent:
    """OpenAI Agent that uses Inception Labs as the underlying model."""

    def __init__(
        self,
        inception_api_key: str,
        inception_base_url: str = "https://api.inceptionlabs.ai/v1",
        model: str = "inception-default",
        system_prompt: str = "You are a helpful assistant."
    ):
        self.inception_model = InceptionLabsModel(inception_api_key, inception_base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.tools: List[Dict] = []
        self.tool_functions: Dict[str, callable] = {}

    def add_tool(self, name: str, description: str, parameters: Dict, function: callable):
        """
        Register a tool/function that the agent can use.

        Args:
            name: Function name
            description: What the function does
            parameters: JSON schema for function parameters
            function: The actual Python function to call
        """
        tool_definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools.append(tool_definition)
        self.tool_functions[name] = function

    def execute_tool_call(self, tool_call: Dict) -> str:
        """Execute a tool/function call."""
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])

        if function_name not in self.tool_functions:
            return f"Error: Function {function_name} not found"

        try:
            result = self.tool_functions[function_name](**function_args)
            return json.dumps(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Send a message to the agent and get a response.
        Handles tool calls automatically.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Call Inception Labs API
            response = self.inception_model.create_completion(
                messages=self.conversation_history,
                model=self.model,
                tools=self.tools if self.tools else None
            )

            # Extract assistant message
            assistant_message = response["choices"][0]["message"]

            # Add assistant response to history
            self.conversation_history.append(assistant_message)

            # Check if there are tool calls
            if assistant_message.get("tool_calls"):
                # Execute each tool call
                for tool_call in assistant_message["tool_calls"]:
                    tool_result = self.execute_tool_call(tool_call)

                    # Add tool result to history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })

                # Continue loop to get final response after tool execution
                continue

            # No tool calls, return the response
            return assistant_message["content"]

        return "Max iterations reached without final response"

    def reset_conversation(self):
        """Clear conversation history except system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

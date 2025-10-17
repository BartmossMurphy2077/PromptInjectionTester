from abc import ABC, abstractmethod
from typing import Tuple
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from utils import (
    AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    DEBUG
)


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, system_prompt: str = None, temperature: float = 0.0):
        # Handle None system prompt for pydantic-ai
        if system_prompt is None:
            system_prompt = ""

        self.agent = Agent(
            model=self._create_model(),
            system_prompt=system_prompt
        )
        self.temperature = temperature

    @abstractmethod
    def _create_model(self):
        """Create and return the appropriate model"""
        pass

    async def run_async(self, prompt: str) -> Tuple[str, int, int]:
        """Run the agent asynchronously and return response, input_tokens, output_tokens"""
        try:
            result = await self.agent.run(
                prompt,
                model_settings={'temperature': self.temperature}
            )

            if not result:
                if DEBUG:
                    print(f"[{self.__class__.__name__}] No result returned from API")
                return "UNEXPECTED", 0, 0

            # Get response content
            response = str(result.output).strip() if result.output is not None else ""
            if not response:
                if DEBUG:
                    print(f"[{self.__class__.__name__}] Empty response content")
                return "UNEXPECTED", 0, 0

            # Extract token usage
            input_tokens = 0
            output_tokens = 0

            try:
                usage = result.usage()
                if usage:
                    input_tokens = usage.request_tokens
                    output_tokens = usage.response_tokens
            except Exception as usage_error:
                if DEBUG:
                    print(f"[{self.__class__.__name__}] Could not extract token usage: {usage_error}")

            return response, input_tokens, output_tokens

        except Exception as e:
            error_message = str(e).lower()

            # Check for prompt shield triggers
            if any(keyword in error_message for keyword in ["prompt", "shield", "policy", "content", "filtered"]):
                if DEBUG:
                    print(f"[{self.__class__.__name__}] Prompt shield triggered: {e}")
                return "PROMPTSHIELD", 0, 0

            if DEBUG:
                print(f"[{self.__class__.__name__}] Unexpected error: {e}")
            return "UNEXPECTED", 0, 0

    def run(self, prompt: str) -> Tuple[str, int, int]:
        """Synchronous wrapper for run_async"""
        import asyncio
        try:
            return asyncio.run(self.run_async(prompt))
        except Exception as e:
            if DEBUG:
                print(f"[{self.__class__.__name__}] Async execution error: {e}")
            return "UNEXPECTED", 0, 0


class AzureAgent(BaseAgent):
    """Agent using Azure OpenAI"""

    def _create_model(self):
        azure_provider = AzureProvider(
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
        )

        return OpenAIChatModel(
            AZURE_DEPLOYMENT_NAME,
            provider=azure_provider,
        )

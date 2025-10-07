from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from utils import AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME, TESTER_TEMPERATURE, DEBUG

class Tester:
    def __init__(self):
        # Create Azure provider
        azure_provider = AzureProvider(
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
        )

        # Create OpenAI chat model with Azure provider
        self.model = OpenAIChatModel(
            AZURE_DEPLOYMENT_NAME,
            provider=azure_provider,
        )

        # Create pydantic-ai agent (no system prompt - just respond to user prompts)
        self.agent = Agent(
            model=self.model,
        )

    async def run_async(self, prompt: str) -> tuple[str, int, int]:
        try:
            result = await self.agent.run(
                prompt,
                model_settings={'temperature': TESTER_TEMPERATURE}
            )

            # Check if result exists
            if not result:
                print("[Tester] No result returned from API")
                return "UNEXPECTED", 0, 0

            # Get the response content from output attribute
            reply_content = result.output
            if reply_content is None:
                print("[Tester] Warning: None response content from API")
                return "UNEXPECTED", 0, 0

            # Convert to string and strip
            reply = str(reply_content).strip()
            if not reply:
                print("[Tester] Warning: Empty response content after processing")
                return "UNEXPECTED", 0, 0

            # Extract token usage from result using pydantic-ai's usage method
            input_tokens = 0
            output_tokens = 0

            try:
                usage = result.usage()
                if usage:
                    input_tokens = usage.request_tokens
                    output_tokens = usage.response_tokens
            except Exception as usage_error:
                if DEBUG:
                    print(f"[Tester] Could not extract token usage: {usage_error}")

            if DEBUG:
                print(f"[Tester] Tokens - input: {input_tokens}, output: {output_tokens}, total: {input_tokens + output_tokens}")

            return reply, input_tokens, output_tokens

        except Exception as e:
            error_message = str(e).lower()

            # Check for prompt shield triggers
            if any(keyword in error_message for keyword in ["prompt", "shield", "policy", "content", "filtered"]):
                print(f"[Tester] Prompt shield triggered: {e}")
                return "PROMPTSHIELD", 0, 0

            print(f"[Tester] Unexpected error: {e}")
            return "UNEXPECTED", 0, 0

    def run(self, prompt: str) -> tuple[str, int, int]:
        import asyncio
        try:
            return asyncio.run(self.run_async(prompt))
        except Exception as e:
            print(f"[Tester] Async execution error: {e}")
            return "UNEXPECTED", 0, 0

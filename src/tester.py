from openai import AzureOpenAI
from utils import AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME

class Tester:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION
        )
        self.deployment_name = AZURE_DEPLOYMENT_NAME

    def run(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            # This will catch both prompt shield and any unexpected errors
            error_message = str(e).lower()
            if "prompt" in error_message or "shield" in error_message or "policy" in error_message or "content" in error_message:
                print(f"[Tester] Prompt shield triggered: {e}")
                return "PROMPTSHIELD"
            else:
                print(f"[Tester] Unexpected error: {e}")
                return "UNEXPECTED"

from openai import AzureOpenAI
from utils import AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME, DEBUG

class Tester:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version=AZURE_API_VERSION
        )
        self.deployment_name = AZURE_DEPLOYMENT_NAME

    def run(self, prompt: str) -> tuple[str, int]:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}]
            )

            if not response or not hasattr(response, "choices") or len(response.choices) == 0:
                print("[Tester] Empty or malformed response from API")
                return "UNEXPECTED", 0

            reply = response.choices[0].message.content.strip()

            usage = getattr(response, "usage", None)
            tokens_used = getattr(usage, "total_tokens", 0) if usage else 0

            if(DEBUG):
                print(f"[Tester] Tokens used: {tokens_used}")

            return reply, tokens_used

        except Exception as e:
            error_message = str(e).lower()
            if any(keyword in error_message for keyword in ["prompt", "shield", "policy", "content"]):
                print(f"[Tester] Prompt shield triggered: {e}")
                return "PROMPTSHIELD", 0

            print(f"[Tester] Unexpected error: {e}")
            return "UNEXPECTED", 0

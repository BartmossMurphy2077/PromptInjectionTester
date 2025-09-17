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

            # Defensive check for API response
            if not response or not hasattr(response, "choices") or len(response.choices) == 0:
                print("[Tester] Empty or malformed response from API")
                return "UNEXPECTED"

            return response.choices[0].message.content.strip()

        except Exception as e:
            error_message = str(e).lower()

            # Heuristic check for Azure Prompt Shield / content policy blocks
            if any(keyword in error_message for keyword in ["prompt", "shield", "policy", "content"]):
                print(f"[Tester] Prompt shield triggered: {e}")
                return "PROMPTSHIELD"

            # Generic unexpected error
            print(f"[Tester] Unexpected error: {e}")
            return "UNEXPECTED"

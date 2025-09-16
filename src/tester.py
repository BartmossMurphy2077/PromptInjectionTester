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
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

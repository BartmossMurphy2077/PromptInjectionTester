from agents import AzureAgent
from utils import TESTER_TEMPERATURE

class Tester(AzureAgent):
    def __init__(self):
        super().__init__(
            system_prompt="",  # Empty string instead of None
            temperature=TESTER_TEMPERATURE
        )
